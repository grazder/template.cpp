set(TARGET main)
add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE template)
target_compile_features(${TARGET} PRIVATE cxx_std_11)