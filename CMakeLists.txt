cmake_minimum_required(VERSION 3.12)
project(template.cpp)
set(TEMPLATE_LIB template)
set(BINDINGS_LIB py_bindings)

add_subdirectory(ggml)
add_library(${TEMPLATE_LIB} STATIC template.cpp template.h)

add_subdirectory(example)
add_subdirectory(tests/pybind11)
pybind11_add_module(${BINDINGS_LIB} tests/py_bindings.cpp)

target_link_libraries(${BINDINGS_LIB} PRIVATE ${TEMPLATE_LIB})
target_link_libraries(${TEMPLATE_LIB} PUBLIC ggml)
target_include_directories(${TEMPLATE_LIB} PUBLIC .)
target_compile_features(${TEMPLATE_LIB} PUBLIC cxx_std_11)