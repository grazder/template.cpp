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

#define FILE_MAGIC   'ggml'

static const size_t MB = 1024*1024;

struct model_hparams {
    int32_t in_channels = 1;
    int32_t bias_size = 1;
};

// This is a simple model with two tensors a and b
struct simple_model {
    model_hparams hparams;

    ggml_backend_t backend = NULL;
    ggml_backend_buffer_t buffer_w;

    // weights
    struct ggml_tensor * fc_w;
    struct ggml_tensor * bias;

    // the context to define the tensor information (dimensions, size, memory data)
    struct ggml_context * ctx;

    std::map<std::string, struct ggml_tensor *> tensors;

    ggml_backend_buffer_t buffer;
};

template<typename T>
static void read_safe(std::ifstream& infile, T& dest) {
    infile.read(reinterpret_cast<char *>(&dest), sizeof(T));
}

// initialize the tensors of the model in this case two matrices 2x2
bool load_model(const std::string & fname, simple_model & model) {
    fprintf(stderr, "%s: loading model from '%s'\n", __func__, fname.c_str());

    auto infile = std::ifstream(fname, std::ios::binary);
    if (!infile) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }

    // verify magic (i.e. ggml signature in hex format)
    {
        uint32_t magic;
        read_safe(infile, magic);
        if (magic != FILE_MAGIC) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
            return false;
        }
    }

    // load hparams
    {
        auto & hparams = model.hparams;
        read_safe(infile, hparams.in_channels);
        printf("%s: in_channels = %d\n", __func__, hparams.in_channels);
    }

    auto & ctx = model.ctx;

    size_t ctx_size = 0;

    // Evaluating context size
    {
        const auto & hparams = model.hparams;

        const int32_t in_channels = hparams.in_channels;
        const int32_t bias_size = hparams.bias_size;

        ctx_size += in_channels * ggml_type_size(GGML_TYPE_F32); // fp32 linear no bias
        ctx_size += bias_size * ggml_type_size(GGML_TYPE_F32); // fp32 bias

        ctx_size += 10ull*MB;  // object overhead
        ctx_size += 2 * ggml_tensor_overhead(); // one linear and one bias

        printf("%s: ggml tensor size    = %d bytes\n", __func__, (int) sizeof(ggml_tensor));
        printf("%s: backend buffer size = %6.2f MB\n", __func__, ctx_size/(1024.0*1024.0));
    }

    // create the ggml context
    {
        struct ggml_init_params params = {
            /* .mem_size   = */   ctx_size,
            /* .mem_buffer = */   NULL,
            /* .no_alloc   = */   true,
        };

        model.ctx = ggml_init(params);
        if(!model.ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }
    
    {
        const auto & hparams = model.hparams;

        const int in_channels = hparams.in_channels;
        const int bias_size = hparams.bias_size;

        model.fc_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, in_channels);
        model.bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, bias_size);

        model.tensors["fc.weight"] = model.fc_w;
        model.tensors["bias"] = model.bias;
    }

    model.backend = ggml_backend_cpu_init();
    if (!model.backend) {
        fprintf(stderr, "%s: ggml_backend_cpu_init() failed\n", __func__);
        return false;
    }

    // allocate tensors in the backend buffers
    model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx, model.backend);
    if (!model.buffer) {
        printf("%s: failed to allocate memory for the model\n", __func__);
        return false;
    }

    // load weights
    {
        int n_tensors = 0;
        size_t total_size = 0;

        while(true) {
            int32_t n_dims;
            int32_t length;
            int32_t ftype;

            read_safe(infile, n_dims);
            read_safe(infile, length);
            read_safe(infile, ftype);

            if (infile.eof()) {
                break;
            }

            int64_t nelements = 1;
            int64_t ne[1] = {1};
            for (int i = 0; i < n_dims; i++) {
                int32_t ne_cur;
                read_safe(infile, ne_cur);
                ne[i] = ne_cur;
                nelements *= ne[i];
            }

            std::string name(length, 0);
            infile.read(&name[0], length);

            if (model.tensors.find(name.data()) == model.tensors.end()) {
                fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.data());
                return false;
            }
            else {
                fprintf(stderr, "%s: found tensor '%s' in model file\n", __func__, name.data());
            }

            auto tensor = model.tensors[name.data()];
            ggml_set_name(tensor, name.c_str());

            if (ggml_nelements(tensor) != nelements) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                return false;
            }

            if (tensor->ne[0] != ne[0]) {
                fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d], expected [%d]\n",
                        __func__, name.data(), (int) tensor->ne[0], (int) ne[0]);
                return false;
            }

            const size_t bpe = ggml_type_size(ggml_type(ftype));

            if ((nelements*bpe)/ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                        __func__, name.data(), ggml_nbytes(tensor), (size_t) nelements*bpe);
                return false;
            }

            infile.read(reinterpret_cast<char *>(tensor -> data), ggml_nbytes(tensor));

            total_size += ggml_nbytes(tensor);
            n_tensors++;
        }

        // ggml_allocr_free(alloc);
        // printf("%s: model size = %8.2f MB\n", __func__, total_size/1024.0/1024.0);
    }

    // infile.close();
    return true;
}

// // build the compute graph to perform a matrix multiplication
// struct ggml_cgraph * build_graph(const simple_model& model) {
//     struct ggml_cgraph  * gf = ggml_new_graph(model.ctx);

//     // result = a*b^T
//     struct ggml_tensor * result = ggml_mul_mat(model.ctx, model.a, model.b);

//     ggml_build_forward_expand(gf, result);
//     return gf;
// }

// // compute with backend
// struct ggml_tensor * compute(const simple_model & model) {
//     struct ggml_cgraph * gf = build_graph(model);

//     int n_threads = 1; // number of threads to perform some operations with multi-threading

//     ggml_graph_compute_with_ctx(model.ctx, gf, n_threads);

//     // in this case, the output tensor is the last one in the graph
//     return gf->nodes[gf->n_nodes - 1];
// }

int main(void) {
    ggml_time_init();

    // // initialize data of matrices to perform matrix multiplication
    // const int rows_A = 4, cols_A = 1;

    // float matrix_A[rows_A * cols_A] = {
    //     5,
    //     5,
    //     4,
    //     8,
    // };

    // const int rows_B = 3, cols_B = 1;
    // /* Transpose([
    //     10, 9, 5,
    //     5, 9, 4
    // ]) 2 rows, 3 cols */
    // float matrix_B[rows_B * cols_B] = {
    //     20,
    //     9,
    //     5,
    // };

    simple_model model;
    load_model("../ggml-model.bin", model);

    // // perform computation in cpu
    // struct ggml_tensor * result = compute(model);

    // // get the result data pointer as a float array to print
    // std::vector<float> out_data(ggml_nelements(result));
    // memcpy(out_data.data(), result->data, ggml_nbytes(result));

    // printf("mul mat (%d x %d) (transposed result):\n[", (int) result->ne[0], (int) result->ne[1]);
    // for (int j = 0; j < result->ne[1] /* rows */; j++) {
    //     if (j > 0) {
    //         printf("\n");
    //     }

    //     for (int i = 0; i < result->ne[0] /* cols */; i++) {
    //         printf(" %.2f", out_data[j * result->ne[0] + i]);
    //     }
    // }
    // printf(" ]\n");

    // free memory
    ggml_free(model.ctx);
    return 0;
}