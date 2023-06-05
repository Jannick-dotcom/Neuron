#pragma once
#include "neuron.hpp"
#include "connection.hpp"
#include "mutationHelper.hpp"
#include <iostream>
#include <fstream>

class Layer
{
public:
    Neuron *neurons;
    uint16_t ctNeurons;
    Layer *nextLayer;
    Layer *prevLayer;

    Layer(long ctNeurons, Layer *prevLayer, ActivationFunctionType ActivationFunction)
    {
        this->ctNeurons = ctNeurons;
        this->neurons = new Neuron[ctNeurons+1];
        this->nextLayer = nullptr;
        this->prevLayer = prevLayer;
        
        for (u_int16_t i = 0; i < ctNeurons+1; i++)
        {
            neurons[i].type = ActivationFunction;
            if (i == ctNeurons) // bias neuron
            {
                neurons[i].outputVal = 1;                // bias neuron
                neurons[i].type = NONE; 
                neurons[i].connectionsIn = nullptr;
                neurons[i].ctConnectionsIn = 0;
                continue; // skip to next neuron because bias neuron has no incoming connections
            }

            if (prevLayer != nullptr) // hidden or output layer
            {
                neurons[i].connectionsIn = new connection[prevLayer->ctNeurons + 1]; // connections from previous layer to this layer
                neurons[i].ctConnectionsIn = prevLayer->ctNeurons+1;                   // count of connections from previous layer to this layer
            }
            else // input layer
            {
                neurons[i].connectionsIn = nullptr;
                neurons[i].ctConnectionsIn = 0;
                continue; // skip to next neuron because input layer has no incoming connections
            }


            for (uint16_t neurPrevLayer = 0; neurPrevLayer < prevLayer->ctNeurons+1; neurPrevLayer++) // for each connection to currentNeuron
            {
                neurons[i].connectionsIn[neurPrevLayer].inputVal = &prevLayer->neurons[neurPrevLayer].outputVal; // set start of connection to previous layer's neuron
                neurons[i].connectionsIn[neurPrevLayer].outputVal = &neurons[i].inputVal;                        // set end of connection to current layer's neuron
                neurons[i].connectionsIn[neurPrevLayer].weight = rand() / double(RAND_MAX) - 0.5;                // set weight of connection to random value between -0.5 and 0.5
                neurons[i].connectionsIn[neurPrevLayer].fromNeuron = &prevLayer->neurons[neurPrevLayer];
                neurons[i].connectionsIn[neurPrevLayer].toNeuron = &neurons[i];
            }
        }
    }
    ~Layer()
    {
        delete[] neurons;
    }

    void addNeuron(ActivationFunctionType ActivationFunction, uint16_t count = 1)
    {
        if(count == 0) return;
        Neuron *newNeurons = new Neuron[ctNeurons + 1 + count];
        //////////////Newly added Neurons////////////////////////////////////////////////////////////////////////
        for(uint16_t i = 0; i < count; i++) //add new Neurons to the start of array
        {
            newNeurons[i].type = ActivationFunction;
            newNeurons[i].ctConnectionsIn = prevLayer->ctNeurons + 1;
            newNeurons[i].connectionsIn = new connection[newNeurons[i].ctConnectionsIn];
            for(uint16_t j = 0; j < newNeurons[i].ctConnectionsIn; j++) // define connections of new neurons
            {
                connection *tempCurrCon = &(newNeurons[i].connectionsIn[j]);
                tempCurrCon->fromNeuron = &(prevLayer->neurons[j]);
                tempCurrCon->toNeuron = &(newNeurons[i]);
                tempCurrCon->inputVal = &(tempCurrCon->fromNeuron->outputVal);
                tempCurrCon->outputVal = &(tempCurrCon->toNeuron->inputVal);
            }
        }
        //////////////Copy not change neurons///////////////////////////////////////////////////////////////////
        for(uint16_t i = 0; i < ctNeurons + 1; i++)
        {
            Neuron *nOld = &(neurons[i]);
            Neuron *nNew = &(newNeurons[count + i]);
            nNew->connectionsIn = nOld->connectionsIn; //Transfer connections
            nNew->ctConnectionsIn = nOld->ctConnectionsIn;
            nOld->connectionsIn = nullptr; //Disown connections of old neurons (otherwise they would be deleted)
            nNew->gradientW = nOld->gradientW;
            nNew->type = nOld->type;
            for(uint16_t j = 0; j < nNew->ctConnectionsIn; j++)
            {
                nNew->connectionsIn[j].outputVal = &(nNew->inputVal);
                nNew->connectionsIn[j].toNeuron = nNew;
            }
        }
        //////////////Delete old Neurons///////////////////////////////////////////////////////////////////////
        delete[] neurons;
        neurons = newNeurons;
        ctNeurons += count;
        //////////////Change next layer connections to new neurons/////////////////////////////////////////////
        for(uint16_t i = 0; i < nextLayer->ctNeurons + 1; i++)
        {
            if(nextLayer->neurons[i].type == NONE) continue; //If bias neuron - nothing to be done
            connection *newCon = new connection[ctNeurons + 1]; //create new connections with updated count
            for(uint16_t j = 0; j < ctNeurons + 1; j++)
            {
                if(j < count) //Newly added neuron connection
                {
                    newCon->fromNeuron = &(neurons[j]);
                    newCon->toNeuron = &(nextLayer->neurons[i]);
                    newCon->inputVal = &(neurons[j].outputVal);
                    newCon->outputVal = &(nextLayer->neurons[i].inputVal);
                    newCon->weight = 1;
                    newCon->prevWeightChange = 0;
                }
                else //Already present connections
                {
                    newCon->fromNeuron = &(neurons[j]);
                    newCon->inputVal = &(neurons[j].outputVal);
                    newCon->toNeuron = &(nextLayer->neurons[i]);
                    newCon->outputVal = &(nextLayer->neurons[i].inputVal);
                    newCon->weight = nextLayer->neurons[i].connectionsIn[j - count].weight;
                    newCon->prevWeightChange = nextLayer->neurons[i].connectionsIn[j - count].prevWeightChange;
                }
            }
            nextLayer->neurons[i].connectionsIn = newCon;
            nextLayer->neurons[i].ctConnectionsIn = ctNeurons + 1;
        }
    }
    void removeNeuron(uint16_t index, Layer *nextLayer)
    {
    }

    void mutate(double mutationRate)
    {
        //to leave the chance that the layer does not mutate at all, we do modulo n+1
        uint8_t mutationSpecifier = rand() % 5; // 0 = add neuron, 1 = remove neuron, 2 = change connection 3 = change activation function 4 = do nothing
        switch (mutationSpecifier)
        {
            case 0: // add neuron
                addNeuron((ActivationFunctionType)(rand() % ActivationFunctionType::NONE));
                break;
            case 1: // remove neuron
                removeNeuron(rand() % ctNeurons+1, nextLayer);
                break;
            case 2: // change connection
            {
                uint16_t neuronSpecifier = rand() % ctNeurons;
                uint16_t connectionSpecifier = rand() % neurons[neuronSpecifier].ctConnectionsIn;
                double weightchange = (rand() / double(RAND_MAX) - 0.5) * mutationRate;
                double *weight = &neurons[neuronSpecifier].connectionsIn[connectionSpecifier].weight;
                *weight += weightchange;
                break;
            }
            case 3: //change activation function
            {
                uint16_t neuronSpecifier = rand() % ctNeurons;
                ActivationFunctionType newActivationfunction = (ActivationFunctionType)(rand() % ActivationFunctionType::NONE);
                neurons[neuronSpecifier].type = newActivationfunction;
                break;
            }
            default:
                break;
        }
    }

    void feedThrough()
    {
        for (uint16_t i = 0; i < ctNeurons+1; i++)
        {
            neurons[i].feedThrough();
        }
    }
    void print()
    {
        for (uint16_t i = 0; i < ctNeurons+1; i++)
        {
            for(uint16_t c = 0; c < neurons[i].ctConnectionsIn; c++)
            {
                std::cout << neurons[i].connectionsIn[c].weight << "\t ";
            }
            std::cout << i << ": " << neurons[i].inputVal << " -> " << neurons[i].outputVal << std::endl;
        }
        std::cout << std::endl;
    }
    void exportToFile(std::ofstream &file)
    {
        for (uint16_t i = 0; i < ctNeurons+1; i++)
        {
            if(neurons[i].type != NONE)
            {
                file << neurons[i].type;
            }
            // else
            // {
            //     file << "biasNeuron";
            // }
            if(neurons[i].ctConnectionsIn == 0)
            {
                file << "\n";
                continue;
            }

            for (uint16_t c = 0; c < neurons[i].ctConnectionsIn; c++)
            {
                file << ", " << neurons[i].connectionsIn[c].weight;
            }
            file << std::endl;
        }
        file << std::endl;
    }
};