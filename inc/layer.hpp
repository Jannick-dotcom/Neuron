#pragma once
#include "neuron.hpp"
#include "connection.hpp"
// #include "mutationHelper.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstring>
const count_t cBiasNeuronCt = 1U;
class Layer
{
public:
    Neuron *neurons;
    count_t ctNeurons;
    Layer *nextLayer;
    Layer *prevLayer;
    bool heapAllocatedNeurons;
    Layer()
    {
        neurons = nullptr;
        ctNeurons = 0;
        nextLayer = nullptr;
        prevLayer = nullptr;
        heapAllocatedNeurons = false;
    }
    Layer(count_t ctNeurons, Layer *prevLayer, ActivationFunctionType ActivationFunction)
    {
        this->ctNeurons = ctNeurons;
        this->neurons = new Neuron[ctNeurons+cBiasNeuronCt];
        this->heapAllocatedNeurons = true;
        this->nextLayer = nullptr;
        this->prevLayer = prevLayer;
        
        for (u_int16_t i = 0; i < ctNeurons+cBiasNeuronCt; i++)
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
                neurons[i].connectionsIn = new connection[prevLayer->ctNeurons + cBiasNeuronCt]; // connections from previous layer to this layer
                neurons[i].ctConnectionsIn = (count_t)prevLayer->ctNeurons + (count_t)cBiasNeuronCt; // count of connections from previous layer to this layer
            }
            else // input layer
            {
                neurons[i].connectionsIn = nullptr;
                neurons[i].ctConnectionsIn = 0;
                continue; // skip to next neuron because input layer has no incoming connections
            }


            for (count_t neurPrevLayer = 0; neurPrevLayer < prevLayer->ctNeurons+cBiasNeuronCt; neurPrevLayer++) // for each connection to currentNeuron
            {
                neurons[i].connectionsIn[neurPrevLayer].inputVal = &prevLayer->neurons[neurPrevLayer].outputVal; // set start of connection to previous layer's neuron
                neurons[i].connectionsIn[neurPrevLayer].outputVal = &neurons[i].inputVal;                        // set end of connection to current layer's neuron
                neurons[i].connectionsIn[neurPrevLayer].weight = (weight_t)rand() / weight_t(RAND_MAX) - (weight_t)0.5;                // set weight of connection to random value between -0.5 and 0.5
                neurons[i].connectionsIn[neurPrevLayer].fromNeuron = &prevLayer->neurons[neurPrevLayer];
                neurons[i].connectionsIn[neurPrevLayer].toNeuron = &neurons[i];
            }
        }
    }
    ~Layer()
    {
        if(heapAllocatedNeurons)
        {
            for(count_t i = 0; i < ctNeurons; i++)
            {
                delete[] neurons[i].connectionsIn;
            }
            delete[] neurons;
        }
    }

    void addNeuron(ActivationFunctionType ActivationFunction, count_t count = 1)
    {
        if(count == 0) return;
        Neuron *newNeurons = new Neuron[ctNeurons + cBiasNeuronCt + count];
        //////////////Newly added Neurons////////////////////////////////////////////////////////////////////////
        for(count_t i = 0; i < count; i++) //add new Neurons to the start of array
        {
            newNeurons[i].type = ActivationFunction;
            if(newNeurons[i].type > NONE)
            {
                std::cout << "ERROR Type in newly added Neuron" << "\n";
                throw std::system_error();
                exit(1);
            }
            newNeurons[i].ctConnectionsIn = (count_t)prevLayer->ctNeurons + (count_t)cBiasNeuronCt;
            newNeurons[i].connectionsIn = new connection[newNeurons[i].ctConnectionsIn];
            for(count_t j = 0; j < newNeurons[i].ctConnectionsIn; j++) // define connections of new neurons
            {
                connection *tempCurrCon = &(newNeurons[i].connectionsIn[j]);
                tempCurrCon->fromNeuron = &(prevLayer->neurons[j]);
                tempCurrCon->toNeuron = &(newNeurons[i]);
                tempCurrCon->inputVal = &(tempCurrCon->fromNeuron->outputVal);
                tempCurrCon->outputVal = &(tempCurrCon->toNeuron->inputVal);
                tempCurrCon->weight = (weight_t)rand() / weight_t(RAND_MAX) - (weight_t)0.5;
            }
        }
        //////////////Copy not change neurons///////////////////////////////////////////////////////////////////
        for(count_t i = 0; i < ctNeurons + cBiasNeuronCt; i++)
        {
            Neuron *nOld = &(neurons[i]);
            Neuron *nNew = &(newNeurons[count + i]);
            *nNew = *nOld; //Copy entire neuron
            nOld->connectionsIn = nullptr; //Disown connections of old neurons (otherwise they would be deleted)
            for(count_t j = 0; j < nNew->ctConnectionsIn; j++)
            {
                nNew->connectionsIn[j].outputVal = &(nNew->inputVal);
                nNew->connectionsIn[j].toNeuron = nNew;
            }
            if(nOld->type > NONE || nNew->type > NONE)
            {
                std::cout << "ERROR Type in unchanged Neuron: " << nNew->type << "\n";
                throw std::system_error();
                exit(1);
            }
        }
        //////////////Delete old Neurons///////////////////////////////////////////////////////////////////////
        delete[] neurons;
        neurons = newNeurons;
        ctNeurons = count_t(ctNeurons + count);
        //////////////Change next layer connections to new neurons/////////////////////////////////////////////
        for(count_t i = 0; i < nextLayer->ctNeurons + cBiasNeuronCt; i++)
        {
            if(nextLayer->neurons[i].type == NONE) continue; //If bias neuron - nothing to be done
            connection *newCon = new connection[ctNeurons + cBiasNeuronCt]; //create new connections with updated count
            for(count_t j = 0; j < ctNeurons + cBiasNeuronCt; j++)
            {
                if(j < count) //Newly added neuron connection
                {
                    newCon[j].fromNeuron = &(neurons[j]);
                    newCon[j].toNeuron = &(nextLayer->neurons[i]);
                    newCon[j].inputVal = &(neurons[j].outputVal);
                    newCon[j].outputVal = &(nextLayer->neurons[i].inputVal);
                    newCon[j].weight = (weight_t)rand() / weight_t(RAND_MAX) - (weight_t)0.5;
                    newCon[j].prevWeightChange = 0;
                }
                else //Already present connections
                {
                    newCon[j].fromNeuron = &(neurons[j]);
                    newCon[j].inputVal = &(neurons[j].outputVal);
                    newCon[j].toNeuron = &(nextLayer->neurons[i]);
                    newCon[j].outputVal = &(nextLayer->neurons[i].inputVal);
                    newCon[j].weight = nextLayer->neurons[i].connectionsIn[j - count].weight;
                    newCon[j].prevWeightChange = nextLayer->neurons[i].connectionsIn[j - count].prevWeightChange;
                }
            }
            delete[] nextLayer->neurons[i].connectionsIn;
            nextLayer->neurons[i].connectionsIn = newCon;
            nextLayer->neurons[i].ctConnectionsIn = count_t(ctNeurons + cBiasNeuronCt);
        }
    }
    void removeNeuron(count_t index)
    {
        if(index >= ctNeurons) return;
        ////////Generate New neurons//////////////////////////////////////////////////////
        Neuron *newNeurons = new Neuron[ctNeurons]; //One less neuron generated
        count_t newNeuronIndex = 0;
        for(count_t i = 0; i < ctNeurons+cBiasNeuronCt; i++)
        {
            if(i == index) //if to be deleted
            {
                delete[] neurons[i].connectionsIn;
                neurons[i].connectionsIn = nullptr;
                continue;
            }
            
            for(count_t j = 0; j < neurons[i].ctConnectionsIn; j++) //update connection endpoints
            {
                neurons[i].connectionsIn[j].toNeuron = &(newNeurons[newNeuronIndex]);
                neurons[i].connectionsIn[j].outputVal = &(newNeurons[newNeuronIndex].inputVal);
            }
            newNeurons[newNeuronIndex] = neurons[i]; //copy entire neuron with connections
            if(newNeurons[newNeuronIndex].type > NONE)
            {
                std::cout << "ERROR Type in copied Neuron: " << newNeurons[newNeuronIndex].type << "\n";
                throw std::system_error();
                exit(1);
            }
            neurons[i].connectionsIn = nullptr;
            newNeuronIndex++;
        }
        ctNeurons--;
        delete[] neurons;
        neurons = newNeurons;
        /////////////////Fix next layer connections//////////////////////////////////////
        for(count_t i = 0; i < nextLayer->ctNeurons +cBiasNeuronCt; i++)
        {
            if(nextLayer->neurons[i].type == NONE) continue; //If bias neuron - nothing to be done
            connection *newCon = new connection[ctNeurons+cBiasNeuronCt];
            count_t newconindex = 0;
            for(count_t j = 0; j < nextLayer->neurons[i].ctConnectionsIn; j++)
            {
                if(j == index) continue;
                newCon[newconindex] = connection(nextLayer->neurons[i].connectionsIn[j]);
                newCon[newconindex].fromNeuron = &(neurons[newconindex]);
                newCon[newconindex].inputVal = &(neurons[newconindex].outputVal);
                newconindex++;
            }
            delete[] nextLayer->neurons[i].connectionsIn;
            nextLayer->neurons[i].connectionsIn = newCon;
            nextLayer->neurons[i].ctConnectionsIn = (count_t)ctNeurons + cBiasNeuronCt;
        }
    }

    void mutate(weight_t mutationRate)
    {
        //to leave the chance that the layer does not mutate at all, we do modulo n+1
        uint8_t mutationSpecifier; // 0 = add neuron, 1 = remove neuron, 2 = change connection 3 = change activation function
        if(nextLayer == nullptr) 
        {
            mutationSpecifier = 2; //if current layer is output layer, force only changing the weights
        }
        else
        {
            mutationSpecifier = uint8_t(rand() % 4); // 0 = add neuron, 1 = remove neuron, 2 = change connection 3 = change activation function
        }
        switch (mutationSpecifier)
        {
            case 0: // add neuron
                addNeuron((ActivationFunctionType)(rand() % ActivationFunctionType::NONE));
                break;
            case 1: // remove neuron
                removeNeuron(count_t(rand() % ctNeurons+cBiasNeuronCt));
                break;
            case 2: // change connection
            {
                count_t neuronSpecifier = count_t(rand() % ctNeurons);
                // if(neurons[neuronSpecifier].ctConnectionsIn == 0) 
                // {
                //     std::cout << "Somehow selected strange Neuron: " << neuronSpecifier << "\n";
                //     throw std::system_error();
                //     return;
                // }
                count_t connectionSpecifier = count_t(rand() % neurons[neuronSpecifier].ctConnectionsIn);
                weight_t weightchange = ((weight_t)rand() / (weight_t)RAND_MAX - (weight_t)0.5) * mutationRate;
                neurons[neuronSpecifier].connectionsIn[connectionSpecifier].weight += weightchange;
                break;
            }
            case 3: //change activation function
            {
                count_t neuronSpecifier = count_t(rand() % ctNeurons);
                ActivationFunctionType newActivationfunction = (ActivationFunctionType)(rand() % ActivationFunctionType::NONE);
                neurons[neuronSpecifier].type = newActivationfunction;
                break;
            }
            default:
                break;
        }
    }
    #ifdef useGPU
    __global__ friend void LayerFeedThroughGpu(Layer *layer);
    #endif

    void feedThrough()
    {
        #ifdef useGPU
        if(prevLayer == nullptr)
        {
            LayerFeedThroughGpu<<<1, ctNeurons+cBiasNeuronCt>>>(this);
        }
        else
        {
            LayerFeedThroughGpu<<<prevLayer->ctNeurons+cBiasNeuronCt, ctNeurons+cBiasNeuronCt>>>(this);
        }
        cudaDeviceSynchronize();
        #else
        for (count_t i = 0; i < ctNeurons+cBiasNeuronCt; i++)
        {
            neurons[i].feedThrough();
        }
        #endif
        if(nextLayer != nullptr)
            nextLayer->feedThrough();
    }
    void print()
    {
        for (count_t i = 0; i < ctNeurons+1; i++)
        {
            for(count_t c = 0; c < neurons[i].ctConnectionsIn; c++)
            {
                std::cout << neurons[i].connectionsIn[c].weight << "\t ";
            }
            std::cout << i << ": " << neurons[i].inputVal << " -> " << neurons[i].outputVal << "\n";
        }
        std::cout << "\n";
    }
    void exportToFile(std::ofstream &file, bool humanReadable)
    {
        for (count_t i = 0; i < ctNeurons+cBiasNeuronCt; i++)
        {
            if(neurons[i].type < NONE)
            {
                file << neurons[i].type;
            }
            else if(neurons[i].type > NONE)
            {
                std::cout << "ERROR Neuron Type: " << neurons[i].type << "\n";
                throw std::system_error();
                exit(1);
            }

            for (count_t c = 0; c < neurons[i].ctConnectionsIn; c++)
            {
                if(!humanReadable)
                {
                    uint64_t iWeight;
                    memcpy(&iWeight, &(neurons[i].connectionsIn[c].weight), sizeof(neurons[i].connectionsIn[c].weight));
                    file << ", " << iWeight;
                }
                else
                {
                    file << ", " << neurons[i].connectionsIn[c].weight;
                }
            }
            file << "\n";
        }
    }
    #ifdef useGPU
    void* operator new(size_t size)
    {
        void *temp;
        cudaMallocManaged(&temp, size);
        return temp;
    }
    void operator delete(void* ptr)
    {
        cudaFree(ptr);
    }
    void operator delete[](void* ptr)
    {
        cudaFree(ptr);
    }
    #endif
};