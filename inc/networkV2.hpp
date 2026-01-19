#ifndef networkv2_h
#define networkv2_h

#include "neuronTypes.hpp"
#include "activationFun.hpp"
#include <fstream>
#include "layerV2.hpp"

#ifdef useGPU
#include <cuda_runtime.h>
__global__ extern void feedThroughGPU(weight_t **weights, weight_t *biases, in_out_t *inputs, in_out_t *activations, count_t prevLayerSize, ActivationFunctionType *actiFun);
#endif

class NetworkV2
{
public:
    LayerV2 *firstLayer;
    LayerV2 *lastLayer;
    count_t ctLayers;
    #ifdef useGPU
    cudaStream_t currentInstance;
    #endif
    NetworkV2();
    ~NetworkV2();
    LayerV2 *addLayer(count_t size, ActivationFunctionType activationFunction);
    void feedThrough(in_out_t *inputs);
    void exportNetwork(std::string fileName, bool humanReadable = false);
    NetworkV2* deepCopy();
    //Import a network from a file
    void getConnections(std::string str, LayerV2 *currentLayer);
    LayerV2 *parseLayer(std::string str);
    void importNetwork(std::string fileName);

    void mutate(double mutationRate);
};
#endif