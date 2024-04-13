#include "template.h"
#include <pybind11/pybind11.h>

int add(int i, int j)
{
    return i + j;
}

namespace py = pybind11;

PYBIND11_MODULE(py_bindings, m)
{
    m.def("add", &add);
}