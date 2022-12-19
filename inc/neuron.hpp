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
    // connection *connectionsOut;
    // u_int32_t ctConnectionsOut;
    void updateInput()
    {
        inputVal = 0.0;
        for (int i = 0; i < ctConnectionsIn; i++)
        {
            connectionsIn[i].feedThrough();
        }
    }
    void activation()
    {
        outputVal = 1.0 / (1.0 + exp(-inputVal));
    }
    // void updateOutput()
    // {
    //     for (int i = 0; i < ctConnectionsOut; i++)
    //     {
    //         connectionsOut[i].setInput(outputVal);
    //     }
    // }
    Neuron()
    {
        this->inputVal = 0.0;
        this->outputVal = 0.0;
        this->connectionsIn = nullptr;
        this->ctConnectionsIn = 0;
        // this->connectionsOut = nullptr;
        // this->ctConnectionsOut = 0;
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
        // updateOutput();
    }
};