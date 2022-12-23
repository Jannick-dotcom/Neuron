#pragma once

#include "neuron.hpp"
#include "connection.hpp"
#include "layer.hpp"
#include "dataPoint.hpp"

class Network
{
public:
    Layer *firstLayer;
    u_int16_t ctLayers;
    Network()
    {
        this->firstLayer = nullptr;
        this->ctLayers = 0;
    }
    ~Network()
    {
        Layer *currentLayer = firstLayer;
        while(currentLayer != nullptr)
        {
            Layer *nextLayer = currentLayer->nextLayer;
            delete currentLayer;
            currentLayer = nextLayer;
        }
    }

    Layer *addLayer(long ctNeurons)
    {
        if(firstLayer == nullptr || ctLayers == 0)
        {
            firstLayer = new Layer(ctNeurons, nullptr);
            ctLayers++;
            return firstLayer;
        }
        else
        {
            Layer *currentLayer = firstLayer;
            while(currentLayer->nextLayer != nullptr)
            {
                currentLayer = currentLayer->nextLayer;
            }
            currentLayer->nextLayer = new Layer(ctNeurons, currentLayer);
            ctLayers++;
            return currentLayer->nextLayer;
        }
    }

    void feedThrough(dataPoint data)
    {
        Layer *currentLayer = firstLayer;
        if(currentLayer == nullptr)
        {
            return;
        }

        for(u_int16_t i = 0; i < firstLayer->ctNeurons; i++)
        {
            firstLayer->neurons[i].inputVal = data.inputs[i];
        }
        while(currentLayer != nullptr)
        {
            currentLayer->feedThrough();
            currentLayer = currentLayer->nextLayer;
        }
    }
    void clearGradients()
    {
        Layer *currentLayer = firstLayer;
        while(currentLayer != nullptr)
        {
            for(u_int16_t i = 0; i < currentLayer->ctNeurons; i++)
            {
                currentLayer->neurons[i].gradientBias = 1;
                for(u_int16_t j = 0; j < currentLayer->neurons[i].ctConnectionsIn; j++)
                {
                    currentLayer->neurons[i].gradientWeights = 1;
                }
            }
            currentLayer = currentLayer->nextLayer;
        }
    }
    double sigmoid(double x)
    {
        return 1.0 / (1.0 + exp(-x));
    }

    double dcost_dout(double expected, double actual)
    {
        return 2.0*(actual - expected);
    }
    double dOut_dWin(double w_in)
    {
        return sigmoid(w_in) * (1 - sigmoid(w_in));
    }
    double dWin_dW(double weight)
    {
        return weight;
    }
    double dWin_dB(double bias)
    {
        return 1;
    }
    void updateWeightsAndBiases(double learnRate)
    {
        Layer *currentLayer = firstLayer;
        while(currentLayer != nullptr)
        {
            for(u_int16_t i = 0; i < currentLayer->ctNeurons; i++)
            {
                currentLayer->neurons[i].bias -= learnRate * currentLayer->neurons[i].gradientBias;
                for(u_int16_t j = 0; j < currentLayer->neurons[i].ctConnectionsIn; j++)
                {
                    currentLayer->neurons[i].connectionsIn[j].weight -= learnRate * currentLayer->neurons[i].gradientWeights;
                }
            }
            currentLayer = currentLayer->nextLayer;
        }
        clearGradients();
    }
};