cmake_minimum_required(VERSION 3.14)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(DenseNet LANGUAGES CXX CUDA)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_30,code=sm_30  -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=[sm_50,compute_50] -gencode arch=compute_52,code=[sm_52,compute_52]")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
MESSAGE("${CMAKE_CUDA_HOST_COMPILER}")

MESSAGE(STATUS ${PROJECT_INCLUDE_DIRS})

link_directories(/usr/local/cuda/lib64)
include_directories(${PROJECT_INCLUDE_DIRS} /usr/local/cuda/include)


add_executable(GPU main.cpp CLI11.hpp CSV.h debug.h default.h kernels.cuh Perceptron.h)

target_compile_features(GPU PUBLIC cxx_std_11)

target_link_libraries(GPU ${PROJECT_LIBRARIES} -lcuda -lcudart -lcublas -lcurand -lm -pthread -lstdc++)