#pragma once

class Neuron;

class connection
{
public:
    double weight;
    double *inputVal;
    double *outputVal;
    Neuron *fromNeuron;
    Neuron *toNeuron;
    // double delta;
    connection()
    {
        this->inputVal = nullptr;
        this->outputVal = nullptr;
        this->weight = 1.0;
        // this->delta = 0.0;
    }
    connection(double *in, double *out, double weight, double bias)
    {
        this->inputVal = in;
        this->outputVal = out;
        this->weight = weight;
    }
    void feedThrough()
    {
        *outputVal += *inputVal * weight;
    }
    double derivativeBias()
    {
        return 1;
    }
    double derivativeWeight()
    {
        return *inputVal;
    }
};