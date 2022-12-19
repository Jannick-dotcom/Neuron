#pragma once
#include "neuron.hpp"
#include "connection.hpp"

class Layer
{
public:
    Neuron *neurons;
    long ctNeurons;
    Layer *nextLayer;
    Layer *prevLayer;

    Layer(long ctNeurons, Layer *prevLayer)
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
                for (int neurPrevLayer = 0; neurPrevLayer < prevLayer->ctNeurons; neurPrevLayer++) //for each connection to currentNeuron
                {
                    neurons[i].connectionsIn[neurPrevLayer].inputVal = &prevLayer->neurons[neurPrevLayer].outputVal; //set start of connection to previous layer's neuron
                    neurons[i].connectionsIn[neurPrevLayer].outputVal = &neurons[i].inputVal; //set end of connection to current layer's neuron
                }
            }
        }
        else
        {
            for (int i = 0; i < ctNeurons; i++)
            {
                neurons[i].connectionsIn = nullptr;
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
};