#include "networkV2.hpp"
#include "cudaErrorHandler.hpp"
#include <cstring>

NetworkV2::NetworkV2()
{
    firstLayer = nullptr;
    lastLayer = nullptr;
    ctLayers = 0;

    currentInstance = 0;
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
    // in_out_t *tempinputs;
    // CUDA_CHECK(cudaMallocManaged(&tempinputs, sizeof(in_out_t) * firstLayer->size));
    // CUDA_CHECK(cudaMemcpy(tempinputs, inputs, sizeof(in_out_t) * firstLayer->size, cudaMemcpyHostToDevice));
    LayerV2 *currentLayer = firstLayer;
    in_out_t *outputsOfLastLayer = inputs;
    while(currentLayer != nullptr)
    {
        currentLayer->feedThrough(outputsOfLastLayer, currentInstance);
        outputsOfLastLayer = currentLayer->activations;
        currentLayer = currentLayer->next;
    }
}