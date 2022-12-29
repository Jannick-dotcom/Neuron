#pragma once

#include "neuron.hpp"
#include "connection.hpp"
#include "layer.hpp"
#include "dataPoint.hpp"
#include "cost.hpp"

typedef unsigned short uint16_t;

class Network
{
public:
    Layer *firstLayer;
    uint16_t ctLayers;
    double cost;
    Cost *costFunction;
    Network(Cost *costFunction)
    {
        this->firstLayer = nullptr;
        this->ctLayers = 0;
        this->costFunction = costFunction;
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

    Layer *addLayer(long ctNeurons, ActivationFunction *activationFunction)
    {
        if(firstLayer == nullptr || ctLayers == 0)
        {
            firstLayer = new Layer(ctNeurons, nullptr, activationFunction);
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
            currentLayer->nextLayer = new Layer(ctNeurons, currentLayer, activationFunction);
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
                currentLayer->neurons[i].gradientW = 0;
            }
            currentLayer = currentLayer->nextLayer;
        }
    }
    
    double nodeCost(double output, double target)
    {
        return costFunction->cost(output, target);
    }

    double dcost_dout(double expected, double actual)
    {
        return costFunction->costDerivative(actual, expected);
    }
    double dOut_dWin(Neuron n, double w_in)
    {
        return n.activationFunction->derivative(w_in);
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
        Layer *currentLayer = firstLayer->nextLayer;
        while(currentLayer != nullptr)
        {
            for(uint16_t i = 0; i < currentLayer->ctNeurons; i++)
            {
                for(uint16_t j = 0; j < currentLayer->neurons[i].ctConnectionsIn; j++)
                {
                    currentLayer->neurons[i].connectionsIn[j].weight -= learnRate * currentLayer->neurons[i].gradientW * dWin_dW(*currentLayer->neurons[i].connectionsIn[j].inputVal);
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
        Layer *lastLayer = currentLayer;

        while(currentLayer->prevLayer != nullptr)
        {
            for(uint16_t i = 0; i < currentLayer->ctNeurons; i++)
            {
                double output = currentLayer->neurons[i].outputVal;
                double w_in = currentLayer->neurons[i].inputVal;
                double doutdwin = currentLayer->neurons[i].activationFunction->derivative(w_in);

                if(currentLayer->nextLayer == nullptr)
                {
                    cost += costFunction->cost(currentLayer->neurons[i].outputVal, expected[i]);
                    currentLayer->neurons[i].gradientW = dcost_dout(expected[i], output) * dOut_dWin(currentLayer->neurons[i], w_in);
                }
                
                for(uint16_t con = 0; con < currentLayer->neurons[i].ctConnectionsIn; con++)
                {
                    double weight = currentLayer->neurons[i].connectionsIn[con].weight;
                    double input = *currentLayer->neurons[i].connectionsIn[con].inputVal;
                    currentLayer->neurons[i].connectionsIn[con].fromNeuron->gradientW += currentLayer->neurons[i].gradientW * dWin_dIn(weight) * dOut_dWin(currentLayer->neurons[i], w_in);
                }
            }
            currentLayer = currentLayer->prevLayer;
        }
        cost = cost / lastLayer->ctNeurons;
    }
    void print()
    {
        Layer *currentLayer = firstLayer;
        uint16_t layerNum = 0;
        while(currentLayer != nullptr)
        {
            std::cout << "Layer " << layerNum << std::endl;
            currentLayer->print();
            currentLayer = currentLayer->nextLayer;
            layerNum++;
        }
    }
};