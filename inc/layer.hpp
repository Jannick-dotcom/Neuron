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
        Neuron *newNeurons = new Neuron[ctNeurons + count + 1];

        for(uint16_t j = 0; j < count; j++)
        {
            copySingleNeuron(&neurons[j], &newNeurons[j]);
            newNeurons[j].connectionsIn = new connection[prevLayer->ctNeurons+1];
            for(uint16_t i = 0; i<prevLayer->ctNeurons+1; i++)
            {
                newNeurons[j].connectionsIn[i].fromNeuron = &prevLayer->neurons[i];
                newNeurons[j].connectionsIn[i].toNeuron = &newNeurons[0];
                newNeurons[j].connectionsIn[i].inputVal = &prevLayer->neurons[i].outputVal;
                newNeurons[j].connectionsIn[i].outputVal = &newNeurons[0].inputVal;
                newNeurons[j].connectionsIn[i].weight = 1;
            }
        }
        for(uint16_t i = count; i<ctNeurons + count + 1; i++)
        {
            copySingleNeuron(&neurons[i-count], &newNeurons[i]);
        }

        //NextLayer
        for(uint16_t n = 0; n < nextLayer->ctNeurons+1; n++) //For each neuron in next layer
        {
            Neuron *n_temp = &nextLayer->neurons[n];
            if(n_temp->type = NONE) continue;
            connection *newConnections = new connection[ctNeurons + count + 1];
            for(uint16_t c = 0; c < count; c++)
            {
                connection *c_temp = &newConnections[c];
                c_temp->weight = 1;
                c_temp->fromNeuron = &newNeurons[c];
                c_temp->inputVal = &newNeurons[c].outputVal;
                c_temp->toNeuron = n_temp;
                c_temp->outputVal = &newNeurons[c].inputVal;
            }

            for(uint16_t c = count; c < ctNeurons + count + 1; c++)
            {
                connection *c_temp = &newConnections[c];
                c_temp->weight = n_temp->connectionsIn[c - count].weight;
                c_temp->fromNeuron = &newNeurons[c];
                c_temp->inputVal = &newNeurons[c].outputVal;
                c_temp->toNeuron = n_temp;
                c_temp->outputVal = &n_temp->inputVal;
            }
        }
    }
    void removeNeuron(uint16_t index, Layer *nextLayer)
    {
        Neuron *newNeurons = new Neuron[ctNeurons];   //create new array with one less neuron
        if(checkHeapAllocated(newNeurons)) return;

        copyNeuronExcept(newNeurons, neurons, ctNeurons, index); //copy all neurons except the one to be removed
        delete[] neurons[index].connectionsIn; //delete connections of removed neuron
        ctNeurons--; //decrease count of neurons

        for(uint16_t i = 0; i < ctNeurons; i++) //fix connections of current layer
        {
            for(uint16_t j = 0; j < newNeurons[i].ctConnectionsIn; j++)
            {
                newNeurons[i].connectionsIn[j].toNeuron = &newNeurons[i];
                newNeurons[i].connectionsIn[j].outputVal = &newNeurons[i].inputVal;
            }
        }

        for(uint16_t i = 0; i < nextLayer->ctNeurons; i++) //fix connections of next layer
        {
            if(isBiasNeuron(&nextLayer->neurons[i])) continue;

            connection *newConnections = new connection[ctNeurons+1]; //create new array with one less connection
            if(checkHeapAllocated(newConnections)) return;

            uint16_t connIndex = 0;
            for(uint16_t j = 0; j < nextLayer->neurons[i].ctConnectionsIn; j++) //copy all connections except the one to be removed
            {
                if(nextLayer->neurons[i].connectionsIn[j].fromNeuron == &neurons[index]) //if pointed to neuron to be removed
                {
                    continue;
                }
                newConnections[connIndex] = nextLayer->neurons[i].connectionsIn[j];
                newConnections[connIndex].fromNeuron = &newNeurons[connIndex];
                newConnections[connIndex].inputVal = &newNeurons[connIndex].outputVal;
                connIndex++;
            }
            delete[] nextLayer->neurons[i].connectionsIn;
            nextLayer->neurons[i].connectionsIn = newConnections;
            nextLayer->neurons[i].ctConnectionsIn--;
        }

        delete[] neurons;
        neurons = newNeurons;
    }

    void mutate(double mutationRate)
    {
        //to leave the chance that the layer does not mutate at all, we do modulo 4 instead of modulo 3
        uint8_t mutationSpecifier = rand() % 5; // 0 = add neuron, 1 = remove neuron, 2 = change connection 3 = change activation function 4 = do nothing
        switch (mutationSpecifier)
        {
            case 0: // add neuron
                //TODO: Fix activation function setting
                // addNeuron(ActivationFunctionType::LINEAR);//(rand() % ActivationFunctionType::NONE));
                addNeuron((ActivationFunctionType)(rand() % ActivationFunctionType::NONE));
                break;
            case 1: // remove neuron
                removeNeuron(rand() % ctNeurons, nextLayer);
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
        for (uint16_t i = 0; i < ctNeurons; i++)
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