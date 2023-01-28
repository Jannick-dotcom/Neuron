#pragma once

#include <string>
#include <cmath>

typedef enum {
    LINEAR,
    SIGMOID,
    TANH,
    RELU,
    LEAKYRELU,
    NONE
} ActivationFunctionType;

double activationFunction(ActivationFunctionType type, double input)
{
    switch (type)
    {
    case LINEAR:
        return input;
        break;
    case SIGMOID:
        return 1.0 / (1.0 + exp(-input));
        break;
    case TANH:
        return tanh(input);
        break;
    case RELU:
        return input > 0 ? input : 0;
        break;
    case LEAKYRELU:
        return input > 0 ? input : 0.01 * input;
        break;
    default:
        return input;
        break;
    }
}

double activationFunctionDerivative(ActivationFunctionType type, double input)
{
    switch (type)
    {
    case LINEAR:
        return 1.0;
        break;
    case SIGMOID:
        return activationFunction(type, input) * (1.0 - activationFunction(type, input));
        break;
    case TANH:
        return 1.0 / pow(cosh(input),2);
        break;
    case RELU:
        return input > 0 ? 1.0 : 0.0;
        break;
    case LEAKYRELU:
        return input > 0 ? 1.0 : 0.01;
        break;
    default:
        return 1.0;
        break;
    }
}