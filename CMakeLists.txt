cmake_minimum_required(VERSION 3.12)
project(template.cpp)
set(TEST_TARGET template)

add_subdirectory(ggml)
add_executable(${TEST_TARGET} template.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
target_compile_features(${TEST_TARGET} PUBLIC cxx_std_11)

add_subdirectory(pybind11)
pybind11_add_module(cmake_example bindings.cpp)