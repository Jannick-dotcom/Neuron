#include "networkV2.hpp"
count_t NetworkV2::instances = 0;
__global__ void feedThroughGPU(LayerV2 *currentLayer, in_out_t *inputs, count_t sizeOfLastLayer)
{
    unsigned int row = blockDim.x*blockIdx.x + threadIdx.x;
    in_out_t weightedSum = 0;
    if(currentLayer->prevLayerSize > 0) 
    {
        weightedSum = currentLayer->biases[row];
        for(count_t iWeights = 0; iWeights < currentLayer->prevLayerSize; iWeights++)
        {
            weightedSum += inputs[iWeights] * currentLayer->weights[row][iWeights];
        }
    }
    else
    {
        weightedSum = inputs[row];
    }
    currentLayer->activations[row] = activationFunction(currentLayer->actiFun[row], weightedSum); //make ReLu
}

int main()
{
    NetworkV2 net;
    net.addLayer(1, LINEAR);
    net.addLayer(100, SIGMOID);
    net.addLayer(100, SIGMOID);
    net.addLayer(1, LINEAR);
    in_out_t *inputs = new in_out_t[5];
    for(count_t i = 0; i < 10000; i++)
    {
        net.feedThrough(inputs);
    }
}