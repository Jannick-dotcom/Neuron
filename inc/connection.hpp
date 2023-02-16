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
    connection(double *in, double *out, double weight, double bias)
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
    void feedThrough()
    {
        if(weight == 0.0)
            return;
        if(inputVal != nullptr && outputVal != nullptr)
            *outputVal += *inputVal * weight;
    }
};