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
        this->neurons = new Neuron[ctNeurons+1];
        this->nextLayer = nullptr;
        this->prevLayer = prevLayer;

        if (prevLayer != nullptr)
        {
            for (u_int16_t i = 0; i < ctNeurons; i++)
            {
                neurons[i].connectionsIn = new connection[prevLayer->ctNeurons+1]; //connections from previous layer to this layer
                neurons[i].ctConnectionsIn = prevLayer->ctNeurons;  //count of connections from previous layer to this layer
                neurons[i].activationFunction = ActivationFunction;
                if(i == ctNeurons) //bias neuron
                {
                    neurons[i].outputVal = 1; //bias neuron
                    neurons[i].activationFunction = nullptr; //TODO: if activation function is nullptr, then neuron is bias neuron
                    neurons[i].connectionsIn = nullptr;
                    neurons[i].ctConnectionsIn = 0;
                    continue; //skip to next neuron because bias neuron has no incoming connections
                }

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
            for (int i = 0; i < ctNeurons-1; i++)
            {
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
            for(int c = 0; c < neurons[i].ctConnectionsIn; c++)
            {
                std::cout << neurons[i].connectionsIn[c].weight << "\t ";
            }
            std::cout << i << ": " << neurons[i].inputVal << " -> " << neurons[i].outputVal << std::endl;
        }
        std::cout << std::endl;
    }
};