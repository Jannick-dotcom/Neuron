#ifndef activationFun_h
#define activationFun_h

#include <cmath>
#include "neuronTypes.hpp"

typedef enum {
    LINEAR,
    SIGMOID,
    TANH,
    RELU,
    LEAKYRELU,
    NONE
} ActivationFunctionType;

#ifdef useGPU
__host__ __device__ 
#endif
in_out_t activationFunction(ActivationFunctionType type, in_out_t input);
in_out_t activationFunctionDerivative(ActivationFunctionType type, in_out_t input);

#endif