#pragma once

class Neuron;

class connection
{
public:
    double weight;
    double prevWeightChange;
    double *inputVal;
    double *outputVal;
    Neuron *fromNeuron;
    Neuron *toNeuron;
    connection()
    {
        this->inputVal = nullptr;
        this->outputVal = nullptr;
        this->fromNeuron = nullptr;
        this->toNeuron = nullptr;
        this->weight = 1.0;
    }
    connection(double *in, double *out, double weight)
    {
        this->inputVal = in;
        this->outputVal = out;
        this->weight = weight;
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