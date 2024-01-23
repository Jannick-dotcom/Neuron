#pragma once
#include "neuronTypes.hpp"
class Neuron;

class connection
{
public:
    weight_t weight;
    weight_t prevWeightChange;
    in_out_t *inputVal;
    in_out_t *outputVal;
    Neuron *fromNeuron;
    Neuron *toNeuron;
    connection()
    {
        this->inputVal = nullptr;
        this->outputVal = nullptr;
        this->fromNeuron = nullptr;
        this->toNeuron = nullptr;
        this->weight = 1.0;
        this->prevWeightChange = 0;
    }
    connection(const connection &conn)
    {
        this->inputVal = conn.inputVal;
        this->outputVal = conn.outputVal;
        this->fromNeuron = conn.fromNeuron;
        this->toNeuron = conn.toNeuron;
        this->weight = conn.weight;
        this->prevWeightChange = conn.prevWeightChange;
    }
    #ifdef useGPU
    __device__ 
    #endif
    void feedThrough()
    {
        *outputVal += *inputVal * weight;
    }
    connection &operator=(const connection second)
    {
        this->fromNeuron = second.fromNeuron;
        this->toNeuron = second.toNeuron;
        this->inputVal = second.inputVal;
        this->outputVal = second.outputVal;
        this->weight = second.weight;
        this->prevWeightChange = second.prevWeightChange;
        return *this;
    }
    #ifdef useGPU
    void* operator new(size_t size)
    {
        void *temp;
        cudaMallocManaged(&temp, size);
        return temp;
    }
    void * operator new[](size_t size)
    {
        void *p;
        cudaMallocManaged(&p, size);
        return p;
    }
    void operator delete(void* ptr)
    {
        cudaFree(ptr);
    }
    #endif
};