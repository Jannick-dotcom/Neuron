#include "layerV2.hpp"
#include "neuronTypes.hpp"
#include "activationFun.hpp"
#include <cstring>
#include <cuda_runtime.h>
#include "cudaErrorHandler.hpp"

#ifdef useGPU
__global__ extern void feedThroughGPU(count_t size, weight_t **weights, weight_t *biases, in_out_t *inputs, in_out_t *activations, count_t prevLayerSize, ActivationFunctionType *actiFun);
#endif

__host__
LayerV2::LayerV2(count_t size, count_t prevLayerSize, ActivationFunctionType activationFunction)
{
    biases = nullptr;
    weights = nullptr;
    actiFun = nullptr;
    activations = nullptr;
    next = nullptr;
    CUDA_CHECK(cudaMallocManaged(&actiFun, sizeof(ActivationFunctionType) * size));
    for(count_t i = 0; i < size; i++) 
    {
        actiFun[i] = activationFunction;
    }
    if(prevLayerSize > 0)
    {
        CUDA_CHECK(cudaMallocManaged(&weights, sizeof(weight_t*) * size));
        for(count_t i = 0; i < size; i++) {
            CUDA_CHECK(cudaMallocManaged(&(weights[i]), sizeof(weight_t) * prevLayerSize));
            for(count_t j = 0; j < prevLayerSize; j++) 
            {
                weights[i][j] = weight_t(rand()) / weight_t(RAND_MAX) - weight_t(0.5);
            }
        }
        CUDA_CHECK(cudaMallocManaged(&biases, sizeof(weight_t) * size));
        for(count_t i = 0; i < size; i++) {
            biases[i] = weight_t(rand()) / weight_t(RAND_MAX) - weight_t(0.5);
        }
    }
    CUDA_CHECK(cudaMallocManaged(&activations, sizeof(in_out_t) * size));
    this->size = size;
    this->prevLayerSize = prevLayerSize;
    this->next = nullptr;
}

__host__
LayerV2::~LayerV2()
{
    if(prevLayerSize > 0)
    {
        for(count_t i = 0; i < size; i++)
        {
            if (weights[i] != NULL) CUDA_CHECK(cudaFree(weights[i]));
        }
        if (weights != NULL) CUDA_CHECK(cudaFree(weights));
        if (biases != NULL) CUDA_CHECK(cudaFree(biases));
    }
    if (actiFun != NULL) CUDA_CHECK(cudaFree(actiFun));
    if (activations != NULL) CUDA_CHECK(cudaFree(activations));
}

__host__
void LayerV2::addNeuron(ActivationFunctionType type)
{
    weight_t **newWeights;
    weight_t *newBiases;
    ActivationFunctionType *newActiFuns;
    CUDA_CHECK(cudaMallocManaged(&newWeights, sizeof(weight_t*) * (size+1)));
    CUDA_CHECK(cudaMallocManaged(&newBiases, sizeof(weight_t) * (size+1)));
    CUDA_CHECK(cudaMallocManaged(&newActiFuns, sizeof(ActivationFunctionType) * (size+1)));
    for(count_t i = 0; i < size; i++) //iterate over every old neuron
    {
        newWeights[i] = weights[i];
        newBiases[i] = biases[i];
        newActiFuns[i] = actiFun[i];
    }
    CUDA_CHECK(cudaMallocManaged(&newWeights[size], sizeof(weight_t) * prevLayerSize));

    for(count_t conn = 0; conn < prevLayerSize; conn++)
    {
        newWeights[size][conn] = weight_t(rand()) / weight_t(RAND_MAX) - weight_t(0.5);
    }

    newBiases[size] = weight_t(rand()) / weight_t(RAND_MAX) - weight_t(0.5);
    newActiFuns[size] = type;

    //fix connections of next layer
    for(count_t neuron = 0; neuron < next->size; neuron++)
    {
        weight_t *nextLayerNewWeights;
        CUDA_CHECK(cudaMallocManaged(&nextLayerNewWeights, sizeof(weight_t) * (size+1)));
        for(count_t conn = 0; conn < this->size; conn++)
        {
            nextLayerNewWeights[conn] = next->weights[neuron][conn];
        }
        nextLayerNewWeights[this->size] = weight_t(rand()) / weight_t(RAND_MAX) - weight_t(0.5);
        CUDA_CHECK(cudaFree(next->weights[neuron]));
        next->weights[neuron] = nextLayerNewWeights;
    }
    next->prevLayerSize = static_cast<count_t>(size+1);
    ////////////////////////////////


    in_out_t *newActivations;
    CUDA_CHECK(cudaMallocManaged(&newActivations, sizeof(in_out_t) * (size+1))); //No need to copy these
    CUDA_CHECK(cudaFree(weights));
    CUDA_CHECK(cudaFree(biases));
    CUDA_CHECK(cudaFree(actiFun));
    CUDA_CHECK(cudaFree(activations));
    weights = newWeights;
    biases = newBiases;
    actiFun = newActiFuns;
    activations = newActivations;
    size++;
}

__host__
void LayerV2::removeNeuron(count_t neuronIndex)
{
    if(size <= 1)
    {
        printf("Cannot remove neuron, layer size would be zero\n");
        return;
    }
    if(neuronIndex >= size)
    {
        printf("Cannot remove neuron, index out of bounds\n");
        return;
    }
    weight_t **newWeights;
    in_out_t *newActivations;
    weight_t *newBiases;
    ActivationFunctionType *newActiFuns;
    CUDA_CHECK(cudaMallocManaged(&newWeights, sizeof(weight_t*) * (size-1)));
    CUDA_CHECK(cudaMallocManaged(&newActivations, sizeof(in_out_t) * (size-1))); //No need to copy these
    CUDA_CHECK(cudaMallocManaged(&newBiases, sizeof(weight_t) * (size-1)));
    CUDA_CHECK(cudaMallocManaged(&newActiFuns, sizeof(ActivationFunctionType) * (size-1)));
    count_t newIndex = 0;
    for(count_t i = 0; i < size; i++) //iterate over every old neuron
    {
        if(i == neuronIndex) continue;
        newWeights[newIndex] = weights[i];
        newBiases[newIndex] = biases[i];
        newActiFuns[newIndex] = actiFun[i];
        newIndex++;
    }

    //fix connections of next layer
    for(count_t neuron = 0; neuron < next->size; neuron++)
    {
        weight_t *nextLayerNewWeights;
        CUDA_CHECK(cudaMallocManaged(&nextLayerNewWeights, sizeof(weight_t) * (size-1)));
        count_t newWeightIndex = 0;
        for(count_t conn = 0; conn < this->size; conn++)
        {
            if(conn == neuronIndex) continue;
            nextLayerNewWeights[newWeightIndex] = next->weights[neuron][conn];
            newWeightIndex++;
        }
        CUDA_CHECK(cudaFree(next->weights[neuron]));
        next->weights[neuron] = nextLayerNewWeights;
    }
    next->prevLayerSize = static_cast<count_t>(size-1);
    ////////////////////////////////

    if(weights == nullptr || biases == nullptr || actiFun == nullptr || activations == nullptr)
    {
        printf("Error\n");
    }
    CUDA_CHECK(cudaFree(weights));
    CUDA_CHECK(cudaFree(biases));
    CUDA_CHECK(cudaFree(actiFun));
    CUDA_CHECK(cudaFree(activations));
    weights = newWeights;
    biases = newBiases;
    actiFun = newActiFuns;
    activations = newActivations;
    size--;
}

__host__
void LayerV2::feedThrough(in_out_t *inputs, cudaStream_t stream)
{
    if(prevLayerSize > 0)
    {
        if(size == 0) 
        {
            printf("Layer size invalid\n");
            throw "Layer size invalid";
            return;
        }
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        feedThroughGPU<<<blocks, threads, 0, stream>>>(this->size, this->weights, this->biases, inputs, this->activations, this->prevLayerSize, this->actiFun);
        // CUDA_CHECK(cudaGetLastError());
        // CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    else 
    {
        for(count_t i = 0; i < size; i++)
        {
            in_out_t weightedSum = 0;
            weightedSum = inputs[i];
            activations[i] = activationFunction(actiFun[i], weightedSum); //make ReLu
        }
    }
}