#include "template.h"
#include <string.h>

int main(void)
{
    ggml_time_init();

    // Create example tensor
    std::vector<float> input = {1, 2, 3, 4, 5};

    // Load model and run forward
    module model;
    load_model("../example.gguf", model);
    // struct ggml_tensor *result = compute(model, input);

    // // Printing
    // std::vector<float> out_data(ggml_nelements(result));
    // memcpy(out_data.data(), result->data, ggml_nbytes(result));

    // printf("Result: [");
    // for (int i = 0; i < result->ne[0]; i++)
    // {
    //     printf("%.2f, ", out_data[i]);
    // }
    // printf("]\n");

    ggml_free(model.ctx);
    return 0;
}