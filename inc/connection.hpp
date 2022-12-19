#pragma once

class connection
{
public:
    double weight;
    double bias;
    double *inputVal;
    double *outputVal;

    connection()
    {
        this->inputVal = nullptr;
        this->outputVal = nullptr;
        this->weight = 1.0;
        this->bias = 0.0;
    }
    connection(double *in, double *out, double weight, double bias)
    {
        this->inputVal = in;
        this->outputVal = out;
        this->weight = weight;
        this->bias = bias;
    }
    void feedThrough()
    {
        *outputVal += *inputVal * weight + bias;
    }
};