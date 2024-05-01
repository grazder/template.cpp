#include "ggml.h"
#include "ggml-backend.h"

#include <map>
#include <string>
#include <vector>

#define GGUF_FILE_MAGIC 0x46554747 // "GGUF"

static const size_t MB = 1024 * 1024;

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

void load_model(const std::string &fname, module &model);
struct ggml_tensor *compute(const module &model, const std::vector<float> &input);