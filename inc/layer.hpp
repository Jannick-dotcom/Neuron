#pragma once
#include "neuron.hpp"
#include "connection.hpp"
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
                neurons[i].connectionsIn[neurPrevLayer].weight = 0;//rand() / double(RAND_MAX) - 0.5;                // set weight of connection to random value between -0.5 and 0.5
                neurons[i].connectionsIn[neurPrevLayer].fromNeuron = &prevLayer->neurons[neurPrevLayer];
                neurons[i].connectionsIn[neurPrevLayer].toNeuron = &neurons[i];
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
                neurons[i].connectionsIn[neurPrevLayer].inputVal = &prevLayer->neurons[neurPrevLayer].outputVal; // set start of connection to previous layer's neuron
                neurons[i].connectionsIn[neurPrevLayer].outputVal = &newNeurons[i].inputVal;                        // set end of connection to current layer's neuron
                neurons[i].connectionsIn[neurPrevLayer].weight = rand() / double(RAND_MAX) - 0.5;                // set weight of connection to random value between -0.5 and 0.5
                neurons[i].connectionsIn[neurPrevLayer].fromNeuron = &prevLayer->neurons[neurPrevLayer];
                neurons[i].connectionsIn[neurPrevLayer].toNeuron = &newNeurons[i];
            }
        }
        delete[] neurons;
        neurons = newNeurons;
        
        //add connections to next layer
        for(uint16_t i = 0; i < nextLayer->ctNeurons; i++)
        {
            connection *newConnections = new connection[nextLayer->neurons[i].ctConnectionsIn + count+1];
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
        if(index >= ctNeurons)
        {
            return;
        }
        Neuron *newNeurons = new Neuron[ctNeurons]; //create new array with one less neuron
        
        //copy old neurons to new array below removed neuron
        for (uint16_t i = 0; i < index; i++)
        {
            newNeurons[i] = neurons[i];
        }
        //
        
        //delete all connections to removed neuron
        delete[] neurons[index].connectionsIn;
        neurons[index].connectionsIn = nullptr;
        //

        //copy old neurons to new array above removed neuron
        for (uint16_t i = index+1; i < ctNeurons+1; i++) //copy old neurons to new array
        {
            newNeurons[i-1] = neurons[i];
        }
        //

        ctNeurons -= 1; //decrease count of neurons

        if(nextLayer == nullptr || nextLayer->ctNeurons == 0 || nextLayer->neurons == nullptr)
        {
            return;
        }

        //Clean connections of next Layer
        for(uint16_t indexNeuron = 0; indexNeuron < nextLayer->ctNeurons; indexNeuron++) //For each Neuron in next layer
        {
            connection *newConnections = new connection[nextLayer->neurons[indexNeuron].ctConnectionsIn - 1]; //create new array with one less connection
            for(uint16_t indexPrevNeuron = 0; indexPrevNeuron < ctNeurons+1; indexPrevNeuron++) //for each connection to currentNeuron
            {
                newConnections[indexPrevNeuron].inputVal = &newNeurons[indexPrevNeuron].outputVal; //copy connection to new array
                newConnections[indexPrevNeuron].outputVal = &nextLayer->neurons[indexNeuron].inputVal; //copy connection to new array
                newConnections[indexPrevNeuron].weight = nextLayer->neurons[indexNeuron].connectionsIn[indexPrevNeuron].weight; //copy connection to new array
                newConnections[indexPrevNeuron].fromNeuron = &newNeurons[indexPrevNeuron];
                newConnections[indexPrevNeuron].toNeuron = &nextLayer->neurons[indexNeuron];
            }
            nextLayer->neurons[indexNeuron].ctConnectionsIn -= 1; //decrease count of connections
            delete[] nextLayer->neurons[indexNeuron].connectionsIn; //delete old array
            nextLayer->neurons[indexNeuron].connectionsIn = newConnections; //set new array
        }
        delete[] neurons;
        neurons = newNeurons;
    }

    void mutate(double mutationRate)
    {
        //to leave the chance that the layer does not mutate at all, we do modulo 4 instead of modulo 3
        uint8_t mutationSpecifier = 1;//rand() % 4; // 0 = add neuron, 1 = remove neuron, 2 = change connection 3 = do nothing
        switch (mutationSpecifier)
        {
            case 0: // add neuron
                addNeuron(neurons[0].type);
                break;
            case 1: // remove neuron
                removeNeuron(rand() % ctNeurons, nextLayer);
                break;
            case 2: // change connection
            {
                Neuron *tempNeuron = &neurons[rand() % ctNeurons];
                connection *tempConn = &tempNeuron->connectionsIn[rand() % tempNeuron->ctConnectionsIn];
                tempConn->weight += (rand() / double(RAND_MAX) - 0.5) * mutationRate;
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
            else
            {
                file << "biasNeuron";
            }
            if(neurons[i].ctConnectionsIn == 0)
            {
                file << "," << std::endl;
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