#include "ggml.h"
#include "template.h"
#include "ggml-backend.h"
#include "utils.cpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <limits.h>
#include <inttypes.h>

/**
 * @brief Loads the weights of a module from the ggml_context.
 *
 * This function loads the weights of a module from the ggml_context by retrieving
 * the tensors with the specified names and assigning them to the corresponding
 * fields of the module.
 *
 * @param model The module to load the weights into.
 */
void load_weights(module &model)
{
    model.fc_w = get_tensor(model.ctx, "fc.weight");
    model.bias = get_tensor(model.ctx, "bias");
}

/**
 * @brief Loads the hyperparameters of a module from the gguf_context.
 *
 * This function loads the hyperparameters of a module from the gguf_context by
 * retrieving the values of the specified keys and assigning them to the corresponding
 * fields of the module's hparams struct.
 *
 * @param model The module to load the hyperparameters into.
 * @param ctx The gguf_context.
 */
void load_hparams(module &model, gguf_context *ctx)
{
    auto &hparams = model.hparams;
    hparams.in_channels = get_i32(ctx, "in_channels");
    hparams.bias_size = get_i32(ctx, "bias_size");
    printf("%s: in_channels = %d\n", __func__, hparams.in_channels);
}

/**
 * @brief Creates an input tensor from a vector of floats.
 *
 * This function creates a 1-dimensional input tensor from a vector of floats.
 * It allocates memory for the tensor, copies the input data into the tensor,
 * and sets the tensor's name.
 *
 * @param input The input data as a vector of floats.
 * @param ctx The ggml_context.
 * @param shape The shape of the tensor.
 * @return A pointer to the created input tensor.
 */
struct ggml_tensor *create_input_tensor(const std::vector<float> &input, struct ggml_context *ctx, int32_t shape)
{
    struct ggml_tensor *input_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, shape);
    memcpy(input_tensor->data, input.data(), ggml_nbytes(input_tensor));
    ggml_set_name(input_tensor, "input_tensor");

    return input_tensor;
}

/**
 * @brief Performs the forward pass of a module.
 *
 * This function performs the forward pass of a module by multiplying the input tensor
 * with the module's weight tensor and adding the bias tensor. It creates a new graph,
 * adds the multiplication and addition operations to the graph, computes the result,
 * and returns it.
 *
 * @param input_tensor The input tensor.
 * @param ctx The ggml_context.
 * @param model The module.
 * @return The result tensor.
 */
struct ggml_tensor *forward(ggml_tensor *input_tensor, struct ggml_context *ctx, const module &model)
{
    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    struct ggml_tensor *result = ggml_add(
        ctx,
        ggml_mul_mat(ctx, model.fc_w, input_tensor),
        model.bias);

    ggml_set_name(result, "result");
    ggml_build_forward_expand(gf, result);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    return result;
}

/**
 * Computes the result of a given model on the input data.
 *
 * @param model The module representing the model.
 * @param input The input data as a vector of floats.
 * @return The computed result as a ggml_tensor pointer.
 */
struct ggml_tensor *compute(const module &model, const std::vector<float> &input)
{
    int32_t shape = model.hparams.in_channels;

    static size_t buf_size = shape * sizeof(float) * 1024 * 1024;
    static void *buf = malloc(buf_size);
    struct ggml_init_params params = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/buf,
        /*.no_alloc   =*/false,
    };
    struct ggml_context *ctx = ggml_init(params);
    struct ggml_tensor *input_tensor = create_input_tensor(input, ctx, shape);
    struct ggml_tensor *result = forward(input_tensor, ctx, model);

    // ggml_graph_print(gf);

    ggml_free(ctx);
    return result;
}

void load_model(const std::string &fname, module &model)
{
    fprintf(stderr, "%s: loading model from '%s'\n", __func__, fname.c_str());

    struct ggml_context *meta = NULL;

    struct gguf_init_params gguf_params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &meta,
    };

    struct gguf_context *ctx = gguf_init_from_file(fname.c_str(), gguf_params);

    if (!ctx)
    {
        throw std::runtime_error(format("%s: failed to open '%s'\n", __func__, fname.c_str()));
    }

    const int n_tensors = gguf_get_n_tensors(ctx);
    const int n_kv = gguf_get_n_kv(ctx);
    printf("%s: loaded meta data with %d key-value pairs and %d tensors from %s\n", __func__, n_kv, n_tensors, fname.c_str());

    // Evaluate context size
    size_t model_size = 0;

    for (int i = 0; i < n_tensors; ++i)
    {
        const char *name = gguf_get_tensor_name(ctx, i);
        const size_t offset = gguf_get_tensor_offset(ctx, i);
        enum ggml_type type = gguf_get_tensor_type(ctx, i);
        struct ggml_tensor *cur = ggml_get_tensor(meta, name);
        size_t tensor_size = ggml_nbytes(cur);
        model_size += tensor_size;

        printf("%s: tensor[%d]: n_dims = %d, name = %s, tensor_size=%zu, offset=%zu, shape: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "], type = %s\n",
               __func__, i, ggml_n_dims(cur), cur->name, tensor_size, offset, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3], ggml_type_name(type));
    }

    // Init model backend - currently CPU only
    model.backend = ggml_backend_cpu_init();
    if (!model.backend)
    {
        throw std::runtime_error(format("%s: ggml_backend_cpu_init() failed\n", __func__));
    }

    // Init model context
    struct ggml_init_params ggml_params = {
        /* .mem_size   = */ (n_tensors + 1) * ggml_tensor_overhead(),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true,
    };

    model.ctx = ggml_init(ggml_params);

    if (!model.ctx)
    {
        throw std::runtime_error(format("%s: ggml_init() failed\n", __func__));
    }

    // Add tensors to context
    for (int i = 0; i < n_tensors; ++i)
    {
        const char *name = gguf_get_tensor_name(ctx, i);
        struct ggml_tensor *t = ggml_get_tensor(meta, name);
        struct ggml_tensor *cur = ggml_dup_tensor(model.ctx, t);
        ggml_set_name(cur, name);
    }

    // Allocate model buffer
    auto fin = std::ifstream(fname, std::ios::binary);

    if (!fin)
    {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
    }

    model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx, model.backend);
    if (!model.buffer)
    {
        throw std::runtime_error(format("%s: failed to allocate memory for the model\n", __func__));
    }

    for (int i = 0; i < n_tensors; ++i)
    {
        const char *name = gguf_get_tensor_name(ctx, i);
        struct ggml_tensor *cur = ggml_get_tensor(model.ctx, name);
        const size_t offset = gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, i);
        fin.seekg(offset, std::ios::beg);
        if (!fin)
        {
            gguf_free(ctx);
            throw std::runtime_error(format("%s: failed to seek for tensor %s\n", __func__, name));
        }
        int num_bytes = ggml_nbytes(cur);
        if (ggml_backend_buffer_is_host(model.buffer))
        {
            // for the CPU and Metal backend, we can read directly into the tensor
            fin.read(reinterpret_cast<char *>(cur->data), num_bytes);
        }
    }
    fin.close();

    // Load hparams and weights into model params
    load_hparams(model, ctx);
    load_weights(model);
}