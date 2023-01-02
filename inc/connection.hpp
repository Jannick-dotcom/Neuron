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
        this->weight = 1.0;
    }
    connection(double *in, double *out, double weight, double bias)
    {
        this->inputVal = in;
        this->outputVal = out;
        this->weight = weight;
        this->prevWeightChange = 0;
    }
    void feedThrough()
    {
        *outputVal += *inputVal * weight;
    }
};