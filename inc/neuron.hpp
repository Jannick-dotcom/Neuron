#pragma once

#include <cmath>
#include <iostream>
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
    __device__ void updateInput()
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
    __device__ void activation(double inputVal)
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
        if(connectionsIn == nullptr) return;
        delete[] connectionsIn;
    }

    void setInput(double inputVal)
    {
        this->inputVal = inputVal;
    }
    double getOutput()
    {
        return outputVal;
    }
    __device__ void feedThrough()
    {
        updateInput();
        activation(inputVal);
    }
    void print()
    {
        std::cout << connectionsIn << "\t";
        for(uint16_t i = 0; i < ctConnectionsIn; i++)
        {
            std::cout << connectionsIn[i].fromNeuron << " ";
            std::cout << connectionsIn[i].toNeuron << " ";
            std::cout << connectionsIn[i].inputVal << " ";
            std::cout << connectionsIn[i].outputVal << " ";
        }
        std::cout << "\n";
    }
    void * operator new(size_t size)
    {
        void *p;
        cudaMallocManaged(&p, size);
        return p;
    }
    void * operator new[](size_t size)
    {
        void *p;
        cudaMallocManaged(&p, size);
        return p;
    }
 
    void operator delete(void * p)
    {
        cudaFree(p);
        free(p);
    }
};