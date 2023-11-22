#pragma once

#include <cmath>
#include <iostream>
#include "connection.hpp"
#include "activationFun.hpp"
#include "neuronTypes.hpp"

class Neuron
{
public:
    in_out_t inputVal;
    in_out_t outputVal;
    weight_t gradientW;
    connection *connectionsIn;
    count_t ctConnectionsIn;
    ActivationFunctionType type;
    #ifdef useGPU
    __device__ 
    #endif
    void updateInput()
    {
        if(connectionsIn != nullptr)
        {
            inputVal = 0.0;
            for (count_t i = 0; i < ctConnectionsIn; i++)
            {
                connectionsIn[i].feedThrough();
            }
        }
    }
    #ifdef useGPU
    __device__ 
    #endif
    void activation(in_out_t inputVal)
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

    void setInput(in_out_t inputVal)
    {
        this->inputVal = inputVal;
    }
    in_out_t getOutput()
    {
        return outputVal;
    }
    #ifdef useGPU
    __device__ 
    #endif
    void feedThrough()
    {
        updateInput();
        activation(inputVal);
    }
    void print()
    {
        std::cout << connectionsIn << "\t";
        for(count_t i = 0; i < ctConnectionsIn; i++)
        {
            std::cout << connectionsIn[i].fromNeuron << " ";
            std::cout << connectionsIn[i].toNeuron << " ";
            std::cout << connectionsIn[i].inputVal << " ";
            std::cout << connectionsIn[i].outputVal << " ";
        }
        std::cout << "\n";
    }
    #ifdef useGPU
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
    #endif
};