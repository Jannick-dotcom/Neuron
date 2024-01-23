#pragma once
#include <cmath>
#include "neuronTypes.hpp"

typedef enum {
    CostQUADRATIC,
    CostEXPONENTIAL,
    CostNONE
} CostFunctionType;

in_out_t costFunction(CostFunctionType type, in_out_t output, in_out_t target)
{
    switch (type)
    {
    case CostQUADRATIC:
        return (in_out_t)(0.5 * pow(output - target, 2));
        break;
    case CostEXPONENTIAL:
        return (in_out_t)exp(-pow(output - target, 2));
        break;
    default:
        return 0;
        break;
    }
}

in_out_t costFunctionDerivative(CostFunctionType type, in_out_t output, in_out_t target)
{
    switch (type)
    {
    case CostQUADRATIC:
        return output - target;
        break;
    case CostEXPONENTIAL:
        return -2 * (output - target) * (in_out_t)exp(-pow(output - target, 2));
        break;
    default:
        return 0;
        break;
    }
}