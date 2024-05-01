#include "ggml.h"
#include "template.h"
#include "ggml-backend.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

template <typename T>
static void read_safe(std::ifstream &infile, T &dest)
{
    infile.read(reinterpret_cast<char *>(&dest), sizeof(T));
}

void add_tensors_to_context(module &model, const int n_tensors, gguf_context *ctx, ggml_context *meta)
{
    for (int i = 0; i < n_tensors; ++i)
    {
        const char *name = gguf_get_tensor_name(ctx, i);
        struct ggml_tensor *t = ggml_get_tensor(meta, name);
        struct ggml_tensor *cur = ggml_dup_tensor(model.ctx, t);
        ggml_set_name(cur, name);
    }
}

static std::string format(const char *fmt, ...)
{
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), buf.size());
}

static int get_key_idx(const gguf_context *ctx, const char *key)
{
    int i = gguf_find_key(ctx, key);
    if (i == -1)
    {
        printf("key %s not found in file\n", key);
    }

    return i;
}

static int32_t get_i32(const gguf_context *ctx, const std::string &key)
{
    const int i = get_key_idx(ctx, key.c_str());

    return gguf_get_val_i32(ctx, i);
}

static struct ggml_tensor *get_tensor(struct ggml_context *ctx, const std::string &name)
{
    struct ggml_tensor *cur = ggml_get_tensor(ctx, name.c_str());
    if (!cur)
    {
        throw std::runtime_error(format("%s: unable to find tensor %s\n", __func__, name.c_str()));
    }

    return cur;
}

void load_weights(module &model)
{
    model.fc_w = get_tensor(model.ctx, "fc.weight");
    model.bias = get_tensor(model.ctx, "bias");
}

void load_hparams(module &model, gguf_context *ctx)
{
    auto &hparams = model.hparams;
    hparams.in_channels = get_i32(ctx, "in_channels");
    printf("%s: in_channels = %d\n", __func__, hparams.in_channels);
}

size_t evaluate_context_size(module &model, const int n_tensors, gguf_context *ctx, ggml_context *meta)
{
    size_t model_size = 0;

    for (int i = 0; i < n_tensors; ++i)
    {
        const char *name = gguf_get_tensor_name(ctx, i);
        const size_t offset = gguf_get_tensor_offset(ctx, i);
        enum ggml_type type = gguf_get_tensor_type(ctx, i);
        struct ggml_tensor *cur = ggml_get_tensor(meta, name);
        size_t tensor_size = ggml_nbytes(cur);
        model_size += tensor_size;

        printf("%s: tensor[%d]: n_dims = %d, name = %s, tensor_size=%zu, offset=%zu, shape: [%d, %d, %d, %d], type = %s\n",
               __func__, i, ggml_n_dims(cur), cur->name, tensor_size, offset, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3], ggml_type_name(type));
    }

    return model_size;
}

void init_model_context(module &model, const int n_tensors)
{
    struct ggml_init_params params = {
        /* .mem_size   = */ (n_tensors + 1) * ggml_tensor_overhead(),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true,
    };

    model.ctx = ggml_init(params);

    if (!model.ctx)
    {
        throw std::runtime_error(format("%s: ggml_init() failed\n", __func__));
    }
}

void init_model_backend(module &model)
{
    model.backend = ggml_backend_cpu_init();
    if (!model.backend)
    {
        throw std::runtime_error(format("%s: ggml_backend_cpu_init() failed\n", __func__));
    }
}

void allocate_model_buffer(module &model, const int n_tensors, gguf_context *ctx, ggml_context *meta, const std::string &fname)
{
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
}

void load_model(const std::string &fname, module &model)
{
    fprintf(stderr, "%s: loading model from '%s'\n", __func__, fname.c_str());

    struct ggml_context *meta = NULL;

    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &meta,
    };

    struct gguf_context *ctx = gguf_init_from_file(fname.c_str(), params);

    if (!ctx)
    {
        throw std::runtime_error(format("%s: failed to open '%s'\n", __func__, fname.c_str()));
    }

    const int n_tensors = gguf_get_n_tensors(ctx);
    const int n_kv = gguf_get_n_kv(ctx);
    printf("%s: loaded meta data with %d key-value pairs and %d tensors from %s\n", __func__, n_kv, n_tensors, fname.c_str());

    size_t model_size = evaluate_context_size(model, n_tensors, ctx, meta);

    init_model_backend(model);
    init_model_context(model, n_tensors);
    add_tensors_to_context(model, n_tensors, ctx, meta);
    allocate_model_buffer(model, n_tensors, ctx, meta, fname);
    load_hparams(model, ctx);
    load_weights(model);
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