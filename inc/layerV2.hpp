#ifndef layerv2_h
#define layerv2_h

#include "neuronTypes.hpp"
#include "activationFun.hpp"
#include <fstream>

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
    void addNeuron(ActivationFunctionType type);
    void removeNeuron(count_t neuronIndex);
    void mutate(weight_t mutationRate);
    #ifdef useGPU
    __global__ friend void feedThroughGPU(LayerV2 *currentLayer, in_out_t *inputs);
    #endif

    #ifdef useGPU
    void feedThrough(in_out_t *inputs, cudaStream_t stream);
    #else
    void feedThrough(in_out_t *inputs);
    #endif
};

#endif