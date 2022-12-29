#pragma once

#include <cmath>
#include "connection.hpp"
#include "activationFun.hpp"

class Neuron
{
public:
    double inputVal;
    double outputVal;
    double bias;
    double gradientW;
    double gradientB;
    connection *connectionsIn;
    u_int32_t ctConnectionsIn;
    ActivationFunction *activationFunction;
    void updateInput()
    {
        if(connectionsIn != nullptr)
        {
            inputVal = 0.0;
            for (int i = 0; i < ctConnectionsIn; i++)
            {
                connectionsIn[i].feedThrough();
            }
            inputVal += bias;
        }
    }
    void activation(double inputVal)
    {
        outputVal = activationFunction->operator()(inputVal);
        // outputVal = 1.0 / (1.0 + exp(-inputVal));
    }
    Neuron()
    {
        this->inputVal = 0.0;
        this->outputVal = 0.0;
        this->bias = rand() / double(RAND_MAX) - 0.5;
        this->connectionsIn = nullptr;
        this->ctConnectionsIn = 0;
        this->gradientB = 0;
        this->gradientW = 0;
    }

    void setInput(double inputVal)
    {
        this->inputVal = inputVal;
    }
    double getOutput()
    {
        return outputVal;
    }
    void feedThrough()
    {
        updateInput();
        // inputVal += bias;
        activation(inputVal);
    }
};