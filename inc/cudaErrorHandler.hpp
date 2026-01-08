#ifndef CUDA_ERROR_HANDLER_HPP
#define CUDA_ERROR_HANDLER_HPP

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

inline void checkCudaError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error at " << file << ":" << line << " - "
                  << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(err) checkCudaError(err, __FILE__, __LINE__)

#endif // CUDA_ERROR_HANDLER_HPP