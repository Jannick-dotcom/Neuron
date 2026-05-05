#ifndef activationFun_h
#define activationFun_h

#include <cmath>
#include <array>
#include "neuronTypes.hpp"
#ifdef useGPU
#include <cuda_runtime.h>
#endif

typedef enum {
    LINEAR = 0,
    SIGMOID,
    TANH,
    RELU,
    LEAKYRELU,
    NONE,
    COUNT
} ActivationFunctionType;

constexpr std::array<const char*, static_cast<size_t>(ActivationFunctionType::COUNT)> activationFunctionName = {
    "LINEAR",
    "SIGMOID",
    "TANH",
    "RELU",
    "LEAKYRELU",
    "NONE"
};

#ifdef useGPU
__host__ __device__ 
#endif
in_out_t activationFunction(ActivationFunctionType type, in_out_t input);
#ifdef useGPU
__host__ __device__ 
#endif
in_out_t activationFunctionDerivative(ActivationFunctionType type, in_out_t input);

#endif