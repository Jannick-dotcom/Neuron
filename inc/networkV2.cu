#include "networkV2.hpp"
#include "cudaErrorHandler.hpp"
#include <cstring>

NetworkV2::NetworkV2()
{
    firstLayer = nullptr;
    lastLayer = nullptr;
    ctLayers = 0;

    currentInstance = 0;
    CUDA_CHECK(cudaStreamCreate(&currentInstance));
}

NetworkV2::~NetworkV2()
{
    LayerV2 *currentLayer = firstLayer;
    while(currentLayer != nullptr)
    {
        LayerV2 *nextLayer = currentLayer->next;
        delete currentLayer;
        currentLayer = nextLayer;
    }
    CUDA_CHECK(cudaStreamSynchronize(currentInstance));
    CUDA_CHECK(cudaStreamDestroy(currentInstance));
}

LayerV2 *NetworkV2::addLayer(count_t size, ActivationFunctionType activationFunction)
{
    ctLayers++;
    LayerV2 *tempLayer;
    if(firstLayer == nullptr)
    {
        tempLayer = new LayerV2(size, 0, activationFunction);
        firstLayer = tempLayer;
    }
    else
    {
        tempLayer = new LayerV2(size, lastLayer->size, activationFunction);
        lastLayer->next = tempLayer;
    }
    lastLayer = tempLayer;
    return tempLayer;
}

void NetworkV2::feedThrough(in_out_t *inputs)
{
    LayerV2 *currentLayer = firstLayer;
    in_out_t *outputsOfLastLayer = inputs;
    while(currentLayer != nullptr)
    {
        currentLayer->feedThrough(outputsOfLastLayer, currentInstance);
        outputsOfLastLayer = currentLayer->activations;
        currentLayer = currentLayer->next;
    }
}