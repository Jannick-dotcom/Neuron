#pragma once
#include "neuron.hpp"
#include "connection.hpp"
#include <iostream>

class Layer
{
public:
    Neuron *neurons;
    long ctNeurons;
    Layer *nextLayer;
    Layer *prevLayer;

    Layer(long ctNeurons, Layer *prevLayer, ActivationFunction *ActivationFunction)
    {
        this->ctNeurons = ctNeurons;
        this->neurons = new Neuron[ctNeurons];
        this->nextLayer = nullptr;
        this->prevLayer = prevLayer;

        if (prevLayer != nullptr)
        {
            for (u_int16_t i = 0; i < ctNeurons; i++)
            {
                neurons[i].connectionsIn = new connection[prevLayer->ctNeurons]; //connections from previous layer to this layer
                neurons[i].ctConnectionsIn = prevLayer->ctNeurons;  //count of connections from previous layer to this layer
                neurons[i].activationFunction = ActivationFunction;
                for (int neurPrevLayer = 0; neurPrevLayer < prevLayer->ctNeurons; neurPrevLayer++) //for each connection to currentNeuron
                {
                    neurons[i].connectionsIn[neurPrevLayer].inputVal = &prevLayer->neurons[neurPrevLayer].outputVal; //set start of connection to previous layer's neuron
                    neurons[i].connectionsIn[neurPrevLayer].outputVal = &neurons[i].inputVal; //set end of connection to current layer's neuron
                    neurons[i].connectionsIn[neurPrevLayer].weight = rand() / double(RAND_MAX) - 0.5; //set weight of connection to random value between -0.5 and 0.5
                    neurons[i].connectionsIn[neurPrevLayer].fromNeuron = &prevLayer->neurons[neurPrevLayer];
                    neurons[i].connectionsIn[neurPrevLayer].toNeuron = &neurons[i];
                }
            }
        }
        else
        {
            for (int i = 0; i < ctNeurons; i++)
            {
                neurons[i].bias = 0;
                neurons[i].activationFunction = ActivationFunction;
                neurons[i].connectionsIn = nullptr;
                neurons[i].ctConnectionsIn = 0;
            }
        }
    }
    ~Layer()
    {
        if(prevLayer != nullptr)
        {
            for (u_int16_t i = 0; i < ctNeurons; i++)
            {
                delete[] neurons[i].connectionsIn;
            }
        }
        delete[] neurons;
    }

    void feedThrough()
    {
        for (int i = 0; i < ctNeurons; i++)
        {
            neurons[i].feedThrough();
        }
    }
    void print()
    {
        for (int i = 0; i < ctNeurons; i++)
        {
            std::cout << i << ": " << neurons[i].inputVal << " -> " << neurons[i].outputVal << " \t\tb:" << neurons[i].bias << std::endl;
        }
        std::cout << std::endl;
    }
};