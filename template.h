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

/**
 * @brief Loads a model from a file and initializes the model context.
 *
 * This function loads a model from the specified file and initializes the model context
 * with the tensors and metadata extracted from the file. It also allocates memory for the model
 * and loads the tensor data from the file into the model buffer.
 *
 * @param fname The path to the model file.
 * @param model The model object to load the model into.
 *
 * @throws std::runtime_error if the model file fails to open, fails to allocate memory for the model,
 * or fails to seek for a tensor in the file.
 */
void load_model(const std::string &fname, module &model);

/**
 * Computes the result of a given model on the input data.
 *
 * @param model The module representing the model.
 * @param input The input data as a vector of floats.
 * @return The computed result as a ggml_tensor pointer.
 */
struct ggml_tensor *compute(const module &model, const std::vector<float> &input);