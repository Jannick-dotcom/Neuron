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
            firstLayer->neurons[i].outputVal = data.inputs[i];
        }
        while(currentLayer != nullptr)
        {
            currentLayer->feedThrough();
            currentLayer = currentLayer->nextLayer;
        }
    }
};