#pragma once
#include <cmath>

typedef enum {
    CostQUADRATIC,
    CostEXPONENTIAL,
    CostNONE
} CostFunctionType;

double costFunction(CostFunctionType type, double output, double target)
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

double costFunctionDerivative(CostFunctionType type, double output, double target)
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