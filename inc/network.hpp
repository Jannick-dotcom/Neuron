#pragma once

#include "neuron.hpp"
#include "connection.hpp"
#include "layer.hpp"
#include "dataPoint.hpp"
#include "cost.hpp"
#include <fstream>

class Network
{
public:
    Layer *firstLayer;
    Layer *outputLayer;
    count_t ctLayers;
    in_out_t cost;
    CostFunctionType costType;
    Network(CostFunctionType costType = CostFunctionType::CostQUADRATIC)
    {
        this->firstLayer = nullptr;
        this->ctLayers = 0;
        this->costType = costType;
    }
    ~Network()
    {
        Layer *currentLayer = firstLayer;
        firstLayer = nullptr;
        outputLayer = nullptr;
        while(currentLayer != nullptr)
        {
            Layer *nextLayer = currentLayer->nextLayer;
            if(currentLayer->heapAllocatedNeurons) delete currentLayer;
            currentLayer = nextLayer;
        }
    }

    Layer *addLayer(count_t ctNeurons, ActivationFunctionType activationFunction)
    {
        if(firstLayer == nullptr || ctLayers == 0)
        {
            firstLayer = new Layer(ctNeurons, nullptr, activationFunction);
            ctLayers++;
            outputLayer = firstLayer;
            return firstLayer;
        }
        else
        {
            Layer *currentLayer = firstLayer;
            while(currentLayer->nextLayer != nullptr)
            {
                currentLayer = currentLayer->nextLayer;
            }
            currentLayer->nextLayer = new Layer(ctNeurons, currentLayer, activationFunction);
            ctLayers++;
            outputLayer = currentLayer->nextLayer;
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

        for(count_t i = 0; i < firstLayer->ctNeurons; i++)
        {
            firstLayer->neurons[i].inputVal = data.inputs[i];
        }
        while(currentLayer != nullptr)
        {
            currentLayer->feedThrough();
            currentLayer = currentLayer->nextLayer;
        }
    }
    void feedThrough()
    {
        Layer *currentLayer = firstLayer;
        if(currentLayer == nullptr)
        {
            return;
        }
        while(currentLayer != nullptr)
        {
            currentLayer->feedThrough();
            currentLayer = currentLayer->nextLayer;
        }
        cudaDeviceSynchronize();
    }
    void clearGradients()
    {
        Layer *currentLayer = firstLayer;
        while(currentLayer != nullptr)
        {
            for(count_t i = 0; i < currentLayer->ctNeurons; i++)
            {
                currentLayer->neurons[i].gradientW = 0;
            }
            currentLayer = currentLayer->nextLayer;
        }
    }
    #ifdef useGPU
    __device__ 
    #endif
    in_out_t nodeCost(in_out_t output, in_out_t target)
    {
        return costFunction(costType, output, target);
    }
    #ifdef useGPU
    __device__ 
    #endif
    in_out_t dcost_dout(in_out_t expected, in_out_t actual)
    {
        return costFunctionDerivative(costType ,actual, expected);
    }
    #ifdef useGPU
    __device__ 
    #endif
    in_out_t dOut_dWin(Neuron &n, weight_t w_in)
    {
        return activationFunctionDerivative(n.type, w_in);
    }
    #ifdef useGPU
    __device__ 
    #endif
    in_out_t dWin_dW(weight_t input)
    {
        return input;
    }
    #ifdef useGPU
    __device__ 
    #endif
    weight_t dWin_dB()
    {
        return 1;
    }
    #ifdef useGPU
    __device__ 
    #endif
    weight_t dWin_dIn(weight_t weight)
    {
        return weight;
    }
    
    #ifdef useGPU
    __device__ 
    #endif
    void updateWeightsAndBiases(weight_t learnRate, weight_t momentumFactor)
    {
        Layer *currentLayer = firstLayer->nextLayer;
        while(currentLayer != nullptr)
        {
            for(count_t i = 0; i < currentLayer->ctNeurons; i++)
            {
                for(count_t j = 0; j < currentLayer->neurons[i].ctConnectionsIn; j++)
                {
                    weight_t weightChange = learnRate * currentLayer->neurons[i].gradientW * dWin_dW(*currentLayer->neurons[i].connectionsIn[j].inputVal);
                    currentLayer->neurons[i].connectionsIn[j].weight -= weightChange + (currentLayer->neurons[i].connectionsIn[j].prevWeightChange * momentumFactor);
                    currentLayer->neurons[i].connectionsIn[j].prevWeightChange += weightChange;
                }
            }
            currentLayer = currentLayer->nextLayer;
        }
        clearGradients();
    }

    void mutate(double mutationRate) //Mutate the network by a certain rate
    {
        uint8_t layerSpecifier = uint8_t((rand() % (ctLayers-1)) + 1); //select a random layer
        //Also give the chance that no layer is mutated (By excluding the first and last layer)
        if(layerSpecifier == 0) return; //Don't mutate the input layer

        Layer *currentLayer = firstLayer; //Get the first layer
        for(count_t i = 0; i < layerSpecifier; i++) 
            currentLayer = currentLayer->nextLayer; //Get the specified random layer
        currentLayer->mutate((weight_t)mutationRate); //Mutate the specified layer
    }
    #ifdef useGPU
    __device__ 
    #endif
    void learn(in_out_t *expected) //Improve the network based on the defined cost function and expected outputs
    {
        Layer *currentLayer = firstLayer;
        while(currentLayer->nextLayer != nullptr) currentLayer = currentLayer->nextLayer; //Get last layer
        Layer *lastLayer = currentLayer;

        while(currentLayer->prevLayer != nullptr)
        {
            for(count_t i = 0; i < currentLayer->ctNeurons; i++)
            {
                auto output = currentLayer->neurons[i].outputVal;
                auto w_in = currentLayer->neurons[i].inputVal;

                if(currentLayer->nextLayer == nullptr)
                {
                    cost += costFunction(costType, currentLayer->neurons[i].outputVal, expected[i]);
                    currentLayer->neurons[i].gradientW = dcost_dout(expected[i], output) * dOut_dWin(currentLayer->neurons[i], w_in);
                }
                
                for(count_t con = 0; con < currentLayer->neurons[i].ctConnectionsIn; con++)
                {
                    auto weight = currentLayer->neurons[i].connectionsIn[con].weight;
                    currentLayer->neurons[i].connectionsIn[con].fromNeuron->gradientW += currentLayer->neurons[i].gradientW * dWin_dIn(weight) * dOut_dWin(currentLayer->neurons[i], w_in);
                }
            }
            currentLayer = currentLayer->prevLayer;
        }
        cost = cost / lastLayer->ctNeurons;
    }
    void print() //Show the network in the console
    {
        Layer *currentLayer = firstLayer;
        count_t layerNum = 0;
        while(currentLayer != nullptr)
        {
            std::cout << "Layer " << layerNum << "\n";
            currentLayer->print();
            currentLayer = currentLayer->nextLayer;
            layerNum++;
        }
    }
    
    void exportNetwork(std::string fileName, bool humanReadable = false) //Export the network to a file (For later use) and for analysis
    {
        std::ofstream file;
        file.open(fileName);
        Layer *currentLayer = firstLayer;
        count_t layerNum = 0;
        while(currentLayer != nullptr)
        {
            file << "Layer" << layerNum << ": " << currentLayer->ctNeurons << "\n";
            currentLayer->exportToFile(file, humanReadable);
            currentLayer = currentLayer->nextLayer;
            layerNum++;
        }
        file.close();
    }

    //Import a network from a file

    void getConnections(std::string str, Layer *currentLayer)
    {
        std::size_t globalStart = 0;
        count_t connectionIndex = 0;
        count_t neuronIndex = 0;
        Neuron *currentNeuron = currentLayer->neurons;
        if(currentLayer->prevLayer == nullptr) //If the current layer is the input layer
        {
            return;
        }

        while(str.find(",", globalStart) < str.length()) //While there are still connections to be found
        {
            std::size_t start = str.find(", ", globalStart) + 2; //Get the start of the connection
            std::size_t end = str.find(", ", start); //Get the end of the connection
            std::size_t newline = str.find("\n", start); //Get the end of the line
            if(start < globalStart) 
            {
                break;
            }
            if(newline < end)
            {
                end = newline;
            }

            if(start != (uint32_t)-1 && end != (uint32_t)-1) //If the connection is valid
            {
                std::string weight = str.substr(start, end - start); //Get the weight of the connection
                uint64_t weightValInt = std::stoul(weight);
                weight_t weightValfloat;
                std::memcpy(&weightValfloat, &weightValInt, sizeof(weight_t));
                currentNeuron->connectionsIn[connectionIndex].weight = weightValfloat;
                connectionIndex++;
                globalStart = end;
                if(connectionIndex == currentNeuron->ctConnectionsIn) //If the current neuron has no more connections
                {
                    connectionIndex = 0; //Reset the connection index
                    currentNeuron++;    //Move to the next neuron
                    neuronIndex++;     //Increment the neuron index
                    if(neuronIndex == currentLayer->ctNeurons) //If the current layer has no more neurons
                    {
                        break;
                    }
                }
            }
            else //If the connection is invalid
            {
                break;
            }
        }
    }
    Layer *parseLayer(std::string str)
    {
        unsigned long layerIndex = str.find("Layer");
        if(layerIndex == std::string::npos)
        {
            return nullptr;
        }

        std::string layerNum = str.substr(layerIndex + 5, str.find(":", layerIndex+5) - layerIndex - 5);
        std::string ctNeurons = str.substr(str.find(":",layerIndex) + 2, str.find("\n", layerIndex) - str.find(":", layerIndex) - 2);
        layerIndex = str.find("\n", layerIndex+1);
        Layer *newLayer = addLayer(count_t(std::stoi(ctNeurons)), NONE);
        for(count_t i = 0; i < newLayer->ctNeurons; i++)
        {
            std::string actString = str.substr(str.find("\n",layerIndex) + 1, str.find(",",layerIndex) - str.find("\n",layerIndex) - 1);
            ActivationFunctionType activationFunction = (ActivationFunctionType)std::stoi(actString); ////////////////Kacke !!!!!!!!!!!!!!!!!!!
            layerIndex = str.find("\n", layerIndex+1);
            if(activationFunction < NONE) 
            {
                newLayer->neurons[i].type = activationFunction;
            }
            else
            {
                std::cout << "ERROR: Activation function not found" << "\n";
                throw std::system_error();
                return nullptr;
            }
        }
        return newLayer;
    }
    void importNetwork(std::string fileName)
    {
        std::ifstream file;
        file.open(fileName);
        std::string lines;
        std::string line;
        
        while(std::getline(file, line))
        {
            if(line.find("Cost:") != std::string::npos) continue;
            if(line != "")
            {
                lines.append(line);
                lines.append("\n");
            }
            else
            {
                Layer *newLayer = parseLayer(lines);
                if(newLayer != nullptr)
                {
                    getConnections(lines, newLayer);
                }
                lines = "";
            }
        }

        file.close();
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