#include "networkV2.hpp"

#include "activationFun.hpp"

#ifdef useGPU
__host__ __device__ 
#endif
in_out_t activationFunction(ActivationFunctionType type, in_out_t input)
{
    switch (type)
    {
    case LINEAR:
        return input;
        break;
    case SIGMOID:
        return (in_out_t)(1.0 / (1.0 + exp(-input)));
        break;
    case TANH:
        return (in_out_t)tanh(input);
        break;
    case RELU:
        return input > 0 ? input : 0;
        break;
    case LEAKYRELU:
        return (in_out_t)(input > 0 ? input : 0.01 * input);
        break;
    default:
        return input;
        break;
    }
}

in_out_t activationFunctionDerivative(ActivationFunctionType type, in_out_t input)
{
    switch (type)
    {
    case LINEAR:
        return 1.0;
        break;
    case SIGMOID:
        return activationFunction(type, input) * ((in_out_t)1.0 - activationFunction(type, input));
        break;
    case TANH:
        return (in_out_t)(1.0 / pow(cosh(input),2));
        break;
    case RELU:
        return input > 0 ? 1.0 : 0.0;
        break;
    case LEAKYRELU:
        return (in_out_t)(input > 0 ? 1.0 : 0.01);
        break;
    default:
        return 1.0;
        break;
    }
}

__global__ void feedThroughGPU(LayerV2 *currentLayer, in_out_t *inputs)
{
    count_t row = blockDim.x*blockIdx.x + threadIdx.x;
    in_out_t weightedSum = 0;
    weightedSum = currentLayer->biases[row];
    for(count_t iWeights = 0; iWeights < currentLayer->prevLayerSize; iWeights++)
    {
        weightedSum += inputs[iWeights] * currentLayer->weights[row][iWeights];
    }
    currentLayer->activations[row] = activationFunction(currentLayer->actiFun[row], weightedSum); //make ReLu
}