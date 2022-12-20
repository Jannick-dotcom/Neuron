#pragma once

#include <cmath>
#include "connection.hpp"

class Neuron
{
public:
    double inputVal;
    double outputVal;
    connection *connectionsIn;
    u_int32_t ctConnectionsIn;
    void updateInput()
    {
        inputVal = 0.0;
        for (int i = 0; i < ctConnectionsIn; i++)
        {
            connectionsIn[i].feedThrough();
        }
    }
    double sigmoid(double x)
    {
        return 1.0 / (1.0 + exp(-x));
    }
    void activation()
    {
        outputVal = sigmoid(inputVal);
    }
    double derivative()
    {
        return sigmoid(inputVal) * (1.0 - sigmoid(inputVal));
    }
    Neuron()
    {
        this->inputVal = 0.0;
        this->outputVal = 0.0;
        this->connectionsIn = nullptr;
        this->ctConnectionsIn = 0;
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
        activation();
    }
};