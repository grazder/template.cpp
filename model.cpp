#include "ggml.h"
#include "ggml-backend.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#define FILE_MAGIC 'ggml'

static const size_t MB = 1024 * 1024;

template <typename T>
static void read_safe(std::ifstream &infile, T &dest)
{
    infile.read(reinterpret_cast<char *>(&dest), sizeof(T));
}

struct model_hparams
{
    int32_t in_channels = 1;
    int32_t bias_size = 1;
};

struct module
{
    model_hparams hparams;

    ggml_backend_t backend = NULL;
    ggml_backend_buffer_t buffer_w;

    // weights
    struct ggml_tensor *fc_w;
    struct ggml_tensor *bias;

    // the context to define the tensor information (dimensions, size, memory data)
    struct ggml_context *ctx;

    std::map<std::string, struct ggml_tensor *> tensors;

    ggml_backend_buffer_t buffer;
};

void create_model_weight_tensors(module &model)
{
    const auto &hparams = model.hparams;

    model.fc_w = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, hparams.in_channels);
    model.bias = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, hparams.bias_size);

    model.tensors["fc.weight"] = model.fc_w;
    model.tensors["bias"] = model.bias;
}

bool verify_magic(std::ifstream &infile)
{
    uint32_t magic;
    read_safe(infile, magic);
    if (magic != FILE_MAGIC)
    {
        fprintf(stderr, "%s: invalid model file (bad magic)\n", __func__);
        return false;
    }
    return true;
}

void load_hparams(std::ifstream &infile, module &model)
{
    auto &hparams = model.hparams;
    read_safe(infile, hparams.in_channels);
    printf("%s: in_channels = %d\n", __func__, hparams.in_channels);
}

size_t evaluate_context_size(module &model)
{
    const auto &hparams = model.hparams;

    size_t ctx_size = 0;

    const int32_t in_channels = hparams.in_channels;
    const int32_t bias_size = hparams.bias_size;

    ctx_size += in_channels * ggml_type_size(GGML_TYPE_F32); // fp32 linear no bias
    ctx_size += bias_size * ggml_type_size(GGML_TYPE_F32);   // fp32 bias

    ctx_size += 10ull * MB;                 // object overhead
    ctx_size += 2 * ggml_tensor_overhead(); // one linear and one bias

    printf("%s: ggml tensor size    = %d bytes\n", __func__, (int)sizeof(ggml_tensor));
    printf("%s: backend buffer size = %6.2f MB\n", __func__, ctx_size / (1024.0 * 1024.0));

    return ctx_size;
}

bool init_model_context(module &model, size_t ctx_size)
{
    struct ggml_init_params params = {
        /* .mem_size   = */ ctx_size,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true,
    };

    model.ctx = ggml_init(params);

    if (!model.ctx)
    {
        fprintf(stderr, "%s: ggml_init() failed\n", __func__);
        return false;
    }
    return true;
}

bool init_model_backend(module &model)
{
    model.backend = ggml_backend_cpu_init();
    if (!model.backend)
    {
        fprintf(stderr, "%s: ggml_backend_cpu_init() failed\n", __func__);
        return false;
    }
    return true;
}

bool allocate_model_buffer(module &model)
{
    model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx, model.backend);
    if (!model.buffer)
    {
        printf("%s: failed to allocate memory for the model\n", __func__);
        return false;
    }
    return true;
}

bool load_weights(std::ifstream &infile, module &model)
{
    size_t total_size = 0;

    while (true)
    {
        int32_t n_dims;
        int32_t length;
        int32_t ftype;

        read_safe(infile, n_dims);
        read_safe(infile, length);
        read_safe(infile, ftype);

        if (infile.eof())
        {
            break;
        }

        int64_t nelements = 1;
        // TODO: FIX TO MULTIPLE SHAPES
        int64_t ne[1] = {1};
        for (int i = 0; i < n_dims; i++)
        {
            int32_t ne_cur;
            read_safe(infile, ne_cur);
            ne[i] = ne_cur;
            nelements *= ne[i];
        }

        std::string name(length, 0);
        infile.read(&name[0], length);

        if (model.tensors.find(name.data()) == model.tensors.end())
        {
            fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.data());
            return false;
        }
        else
        {
            fprintf(stderr, "%s: found tensor '%s' in model file\n", __func__, name.data());
        }

        auto tensor = model.tensors[name.data()];
        ggml_set_name(tensor, name.c_str());

        if (ggml_nelements(tensor) != nelements)
        {
            fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
            return false;
        }

        // TODO: FIX TO MULTIPLE SHAPES
        if (tensor->ne[0] != ne[0])
        {
            fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d], expected [%d]\n",
                    __func__, name.data(), (int)tensor->ne[0], (int)ne[0]);
            return false;
        }

        const size_t bpe = ggml_type_size(ggml_type(ftype));

        if ((nelements * bpe) / ggml_blck_size(tensor->type) != ggml_nbytes(tensor))
        {
            fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                    __func__, name.data(), ggml_nbytes(tensor), (size_t)nelements * bpe);
            return false;
        }

        infile.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));

        total_size += ggml_nbytes(tensor);
    }

    printf("%s: model size = %8.2f MB\n", __func__, total_size / 1024.0 / 1024.0);
    return true;
}

bool load_model(const std::string &fname, module &model)
{
    fprintf(stderr, "%s: loading model from '%s'\n", __func__, fname.c_str());

    auto infile = std::ifstream(fname, std::ios::binary);
    if (!infile)
    {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }

    if (!verify_magic(infile))
    {
        return false;
    }

    load_hparams(infile, model);
    size_t ctx_size = evaluate_context_size(model);

    if (!init_model_context(model, ctx_size))
    {
        return false;
    }

    create_model_weight_tensors(model);

    if (!init_model_backend(model))
    {
        return false;
    }

    if (!allocate_model_buffer(model))
    {
        return false;
    }

    if (!load_weights(infile, model))
    {
        return false;
    }

    infile.close();
    return true;
}

struct ggml_tensor *create_input_tensor(const std::vector<float> &input, struct ggml_context *ctx, int32_t shape)
{
    struct ggml_tensor *input_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, shape);
    memcpy(input_tensor->data, input.data(), ggml_nbytes(input_tensor));
    ggml_set_name(input_tensor, "input_tensor");

    return input_tensor;
}

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

// compute with backend
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

int main(void)
{
    ggml_time_init();

    module model;
    load_model("../ggml-model.bin", model);

    // perform computation in cpu
    std::vector<float> input = {1, 2, 3, 4, 5};
    struct ggml_tensor *result = compute(model, input);

    // get the result data pointer as a float array to print
    std::vector<float> out_data(ggml_nelements(result));
    memcpy(out_data.data(), result->data, ggml_nbytes(result));

    printf("Result: [");
    for (int i = 0; i < result->ne[0]; i++)
    {
        printf("%.2f, ", out_data[i]);
    }
    printf("]\n");

    ggml_free(model.ctx);
    return 0;
}