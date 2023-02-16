#pragma once

#include "neuron.hpp"
#include "connection.hpp"

void copyNeuronExcept(Neuron *newNeurons, Neuron *old, uint16_t count, uint16_t index)
{
    for(uint16_t i = 0; i < count+1; i++) //copy all neurons except the one to be removed
    {
        if(i < index)
        {
            Neuron *newatindex = &newNeurons[i];
            Neuron *oldatindex = &old[i];
            newatindex->type = oldatindex->type;
            newatindex->inputVal = oldatindex->inputVal;
            newatindex->outputVal = oldatindex->outputVal;
            newatindex->gradientW = oldatindex->gradientW;
            newatindex->connectionsIn = oldatindex->connectionsIn;
            newatindex->ctConnectionsIn = oldatindex->ctConnectionsIn;
        }
        else if(i > index)
        {
            Neuron *newatindex = &newNeurons[i-1];
            Neuron *oldatindex = &old[i];
            newatindex->type = oldatindex->type;
            newatindex->inputVal = oldatindex->inputVal;
            newatindex->outputVal = oldatindex->outputVal;
            newatindex->gradientW = oldatindex->gradientW;
            newatindex->connectionsIn = oldatindex->connectionsIn;
            newatindex->ctConnectionsIn = oldatindex->ctConnectionsIn;
        }
    }
}

uint8_t isBiasNeuron(const Neuron *Neuron)
{
    if(Neuron->connectionsIn == nullptr) return 1;
    else return 0;
}

int checkHeapAllocated(void *ptr)
{
    if(ptr == nullptr)
    {
        return 1;
    }
    return 0;
}
