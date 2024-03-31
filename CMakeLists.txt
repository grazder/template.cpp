cmake_minimum_required(VERSION 3.12)
project(simple_ggml_example)

add_subdirectory(ggml)

# simple-ctx

set(TEST_TARGET simple-ctx)
add_executable(${TEST_TARGET} simple-ctx.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
target_compile_features(${TEST_TARGET} PUBLIC cxx_std_14)