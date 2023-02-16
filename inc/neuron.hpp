#pragma once

#include <cmath>
#include "connection.hpp"
#include "activationFun.hpp"

class Neuron
{
public:
    double inputVal;
    double outputVal;
    double gradientW;
    connection *connectionsIn;
    u_int32_t ctConnectionsIn;
    ActivationFunctionType type;
    void updateInput()
    {
        if(connectionsIn != nullptr)
        {
            inputVal = 0.0;
            for (uint16_t i = 0; i < ctConnectionsIn; i++)
            {
                connectionsIn[i].feedThrough();
            }
        }
    }
    void activation(double inputVal)
    {
        outputVal = activationFunction(type, inputVal);
    }
    Neuron()
    {
        this->inputVal = 0.0;
        this->outputVal = 0.0;
        this->connectionsIn = nullptr;
        this->ctConnectionsIn = 0;
        this->gradientW = 0;
    }
    ~Neuron()
    {
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
        activation(inputVal);
    }
};