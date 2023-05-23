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
        if(count == 0) //If no neurons should be added
        {
            return;
        }
        Neuron *newNeurons = new Neuron[ctNeurons + count + 1];   //create new array with more neurons ( + 1 because of bias neuron)
        for (int i = 0; i < ctNeurons+1; i++) //for each neuron in current layer
        {
            newNeurons[i+count].inputVal = neurons[i].inputVal;
            newNeurons[i+count].outputVal = neurons[i].outputVal;
            newNeurons[i+count].gradientW = neurons[i].gradientW;
            newNeurons[i+count].connectionsIn = neurons[i].connectionsIn; //create new array for connections
            newNeurons[i+count].ctConnectionsIn = neurons[i].ctConnectionsIn; //copy count of connections
            newNeurons[i+count].type = neurons[i].type; //copy activation function
            for(uint16_t conn = 0; conn < newNeurons[i+count].ctConnectionsIn; conn++) //for each connection
            {
                newNeurons[i+count].connectionsIn[conn].toNeuron = &newNeurons[i+count]; //set end of connection to new neuron
                newNeurons[i+count].connectionsIn[conn].outputVal = &newNeurons[i+count].inputVal; //set end of connection to new neuron
            }
        }
        ctNeurons += count; //increase count of neurons

        //add new neurons to the start of the neuron array
        for(uint16_t i = 0; i < count; i++) //only added neurons
        {
            newNeurons[i].type = ActivationFunction; // set activation function of neuron
            newNeurons[i].connectionsIn = new connection[prevLayer->ctNeurons + 1]; // connections from previous layer to this layer
            newNeurons[i].ctConnectionsIn = prevLayer->ctNeurons+1;                   // count of connections from previous layer to this layer
            for (uint16_t neurPrevLayer = 0; neurPrevLayer < prevLayer->ctNeurons+1; neurPrevLayer++) // for each connection to currentNeuron
            {
                newNeurons[i].connectionsIn[neurPrevLayer].inputVal = &prevLayer->neurons[neurPrevLayer].outputVal; // set start of connection to previous layer's neuron
                newNeurons[i].connectionsIn[neurPrevLayer].outputVal = &newNeurons[i].inputVal;                        // set end of connection to current layer's neuron
                newNeurons[i].connectionsIn[neurPrevLayer].weight = rand() / double(RAND_MAX) - 0.5;                // set weight of connection to random value between -0.5 and 0.5
                newNeurons[i].connectionsIn[neurPrevLayer].fromNeuron = &prevLayer->neurons[neurPrevLayer];
                newNeurons[i].connectionsIn[neurPrevLayer].toNeuron = &newNeurons[i];
            }
        }
        delete[] neurons;
        neurons = newNeurons;
        
        //add connections to next layer
        for(uint16_t i = 0; i < nextLayer->ctNeurons; i++)
        {
            connection *newConnections = new connection[nextLayer->neurons[i].ctConnectionsIn + count];
            for(uint16_t j = 0; j < nextLayer->neurons[i].ctConnectionsIn; j++)
            {
                newConnections[j+count] = nextLayer->neurons[i].connectionsIn[j];
            }
            for(uint16_t j = 0; j < count; j++)
            {
                newConnections[j].inputVal = &neurons[j].outputVal;
                newConnections[j].outputVal = &nextLayer->neurons[i].inputVal;
                newConnections[j].weight = rand() / double(RAND_MAX) - 0.5;
                newConnections[j].fromNeuron = &neurons[j];
                newConnections[j].toNeuron = &nextLayer->neurons[i];
            }
            nextLayer->neurons[i].ctConnectionsIn += count;
            delete[] nextLayer->neurons[i].connectionsIn;
            nextLayer->neurons[i].connectionsIn = newConnections;
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