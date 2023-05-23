#pragma once

#include "neuron.hpp"
#include "connection.hpp"

void copySingleNeuron(Neuron *source, Neuron *target)
{
    target->type = source->type;
    target->inputVal = source->inputVal;
    target->outputVal = source->outputVal;
    target->gradientW = source->gradientW;
    target->connectionsIn = source->connectionsIn;
    target->ctConnectionsIn = source->ctConnectionsIn;
    
    if(source->ctConnectionsIn == 0) return;

    for(uint16_t j = 0; j < source->ctConnectionsIn; j++)
    {
        target->connectionsIn[j].toNeuron = target;
        target->connectionsIn[j].outputVal = &target->inputVal;

        target->connectionsIn[j].fromNeuron = source->connectionsIn[j].fromNeuron;
        target->connectionsIn[j].inputVal = source->connectionsIn[j].inputVal;
    }
}

void copyNeuronExcept(Neuron *newNeurons, Neuron *old, uint16_t count, uint16_t index)
{
    for(uint16_t i = 0; i < count+1; i++) //copy all neurons except the one to be removed
    {
        Neuron *newatindex;
        Neuron *oldatindex = &old[i];
        if(i < index)
        {
            newatindex = &newNeurons[i];
        }
        else if(i > index)
        {
            newatindex = &newNeurons[i-1];
        }
        else
            continue;
            
        copySingleNeuron(oldatindex, newatindex);
    }
}

bool isBiasNeuron(const Neuron *Neuron)
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
