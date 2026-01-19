#ifndef layerv2_h
#define layerv2_h

#include "neuronTypes.hpp"
#include "activationFun.hpp"
#include <fstream>

#ifdef useGPU
#include <cuda_runtime.h>
#endif

class LayerV2
{
public:
    LayerV2 *next;
    weight_t **weights; //incoming weights
    weight_t *biases; //incoming biases
    ActivationFunctionType *actiFun; //activation functions
    in_out_t *activations; //current activations
    count_t size = 0;
    count_t prevLayerSize = 0;
    LayerV2(count_t size, count_t prevLayerSize, ActivationFunctionType activationFunction);
    ~LayerV2();
    void exportToFile(std::ofstream &file, bool humanReadable);
    LayerV2* deepCopy();
    void addNeuron(ActivationFunctionType type);
    void removeNeuron(count_t neuronIndex);
    void mutate(weight_t mutationRate);
    #ifdef useGPU
    __global__ friend void feedThroughGPU(weight_t **weights, weight_t *biases, in_out_t *inputs, in_out_t *activations, count_t prevLayerSize, ActivationFunctionType *actiFun);
    #endif

    #ifdef useGPU
    void feedThrough(in_out_t *inputs, cudaStream_t stream);
    #else
    void feedThrough(in_out_t *inputs);
    #endif
};

#endif