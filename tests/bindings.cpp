#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "template.h" // Include your header file

namespace py = pybind11;

PYBIND11_MODULE(bindings, m) {
    m.doc() = "Your module documentation string";

    py::class_<module>(m, "Module")
        .def(py::init<>())
        .def_readwrite("hparams", &module::hparams)
        // .def_readwrite("backend", &module::backend)
        // .def_readwrite("buffer_w", &module::buffer_w)
        .def_readwrite("fc_w", &module::fc_w)
        .def_readwrite("bias", &module::bias)
        // .def_readwrite("ctx", &module::ctx)
        .def_readwrite("tensors", &module::tensors);
    
    m.def("load_model", [](const std::string& fname) -> py::object {
        module model;
        load_model(fname, model);
        return py::cast(model);
    }, "A function to load the model");
    
    m.def("compute", [](const module& model, const std::vector<float>& input) {
        struct ggml_tensor* result = compute(model, input);
        std::vector<float> out_data(ggml_nelements(result));
        memcpy(out_data.data(), result->data, ggml_nbytes(result));
        // ggml_free(result);
        return out_data;
    }, "A function to compute using the loaded model");
}
