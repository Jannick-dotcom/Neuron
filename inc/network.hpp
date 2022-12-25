#pragma once

#include "neuron.hpp"
#include "connection.hpp"
#include "layer.hpp"
#include "dataPoint.hpp"

typedef unsigned short uint16_t;

class Network
{
public:
    Layer *firstLayer;
    uint16_t ctLayers;
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

        for(uint16_t i = 0; i < firstLayer->ctNeurons; i++)
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
            for(uint16_t i = 0; i < currentLayer->ctNeurons; i++)
            {
                currentLayer->neurons[i].gradient = 0;
            }
            currentLayer = currentLayer->nextLayer;
        }
    }
    double sigmoid(double x)
    {
        return 1.0 / (1.0 + exp(-x));
    }
    double nodeCost(double output, double target)
    {
        return pow((output - target), 2); // return squared error
    }

    double dcost_dout(double expected, double actual)
    {
        return 2.0*(actual - expected);
    }
    double dOut_dWin(double w_in)
    {
        return sigmoid(w_in) * (1 - sigmoid(w_in));
    }
    double dWin_dW(double input)
    {
        return input;
    }
    double dWin_dB()
    {
        return 1;
    }
    double dWin_dIn(double weight)
    {
        return weight;
    }
    
    void updateWeightsAndBiases(double learnRate)
    {
        Layer *currentLayer = firstLayer;
        while(currentLayer != nullptr)
        {
            for(uint16_t i = 0; i < currentLayer->ctNeurons; i++)
            {
                currentLayer->neurons[i].bias -= learnRate * currentLayer->neurons[i].gradient * dWin_dB();
                for(uint16_t j = 0; j < currentLayer->neurons[i].ctConnectionsIn; j++)
                {
                    currentLayer->neurons[i].connectionsIn[j].weight -= learnRate * currentLayer->neurons[i].gradient * dWin_dW(*currentLayer->neurons[i].connectionsIn[j].inputVal);
                }
            }
            currentLayer = currentLayer->nextLayer;
        }
        clearGradients();
    }

    void learn(double *expected)
    {
        Layer *currentLayer = firstLayer;
        while(currentLayer->nextLayer != nullptr) currentLayer = currentLayer->nextLayer; //Get last layer

        while(currentLayer != nullptr)
        {
            for(uint16_t i = 0; i < currentLayer->ctNeurons; i++)
            {
                double output = currentLayer->neurons[i].outputVal;
                double w_in = currentLayer->neurons[i].inputVal;

                if(currentLayer->nextLayer == nullptr)
                {
                    currentLayer->neurons[i].gradient = dcost_dout(expected[i], output) * dOut_dWin(w_in);
                }
                
                for(uint16_t con = 0; con < currentLayer->neurons[i].ctConnectionsIn; con++)
                {
                    currentLayer->neurons[i].connectionsIn[con].fromNeuron->gradient += currentLayer->neurons[i].gradient * dOut_dWin(w_in) * dWin_dIn(currentLayer->neurons[i].connectionsIn[con].weight);
                }
            }
            currentLayer = currentLayer->prevLayer;
        }
    }
};