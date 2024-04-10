cmake_minimum_required(VERSION 3.12)
project(simple_ggml_example)

add_subdirectory(ggml)

set(TEST_TARGET model)
add_executable(${TEST_TARGET} model.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
target_compile_features(${TEST_TARGET} PUBLIC cxx_std_11)