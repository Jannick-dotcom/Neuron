#pragma once
#include <cmath>
#include "neuronTypes.hpp"

typedef enum {
    CostQUADRATIC,
    CostEXPONENTIAL,
    CostNONE
} CostFunctionType;

float costFunction(CostFunctionType type, in_out_t output, in_out_t target)
{
    switch (type)
    {
    case CostQUADRATIC:
        return 0.5 * pow(output - target, 2);
        break;
    case CostEXPONENTIAL:
        return exp(-pow(output - target, 2));
        break;
    default:
        return 0;
        break;
    }
}

float costFunctionDerivative(CostFunctionType type, in_out_t output, in_out_t target)
{
    switch (type)
    {
    case CostQUADRATIC:
        return output - target;
        break;
    case CostEXPONENTIAL:
        return -2 * (output - target) * exp(-pow(output - target, 2));
        break;
    default:
        return 0;
        break;
    }
}