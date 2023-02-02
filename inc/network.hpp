#pragma once

#include "neuron.hpp"
#include "connection.hpp"
#include "layer.hpp"
#include "dataPoint.hpp"
#include "cost.hpp"
#include <fstream>

typedef unsigned short uint16_t;

class Network
{
public:
    Layer *firstLayer;
    Layer *outputLayer;
    uint16_t ctLayers;
    double cost;
    Cost *costFunction;
    Network()
    {
        this->firstLayer = nullptr;
        this->ctLayers = 0;
        this->costFunction = nullptr;
    }
    Network(Cost *costFunction)
    {
        this->firstLayer = nullptr;
        this->ctLayers = 0;
        this->costFunction = costFunction;
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

    Layer *addLayer(long ctNeurons, ActivationFunctionType activationFunction)
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

        for(uint16_t i = 0; i < firstLayer->ctNeurons; i++)
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
    }
    void clearGradients()
    {
        Layer *currentLayer = firstLayer;
        while(currentLayer != nullptr)
        {
            for(uint16_t i = 0; i < currentLayer->ctNeurons; i++)
            {
                currentLayer->neurons[i].gradientW = 0;
            }
            currentLayer = currentLayer->nextLayer;
        }
    }
    
    double nodeCost(double output, double target)
    {
        return costFunction->cost(output, target);
    }

    double dcost_dout(double expected, double actual)
    {
        return costFunction->costDerivative(actual, expected);
    }
    double dOut_dWin(Neuron n, double w_in)
    {
        return activationFunctionDerivative(n.type, w_in);
    }
    double dWin_dW(double input)
    {
        return input;
    }
    double dWin_dB()
    {
        return 1;
    }
    double dWin_dIn(double weight)
    {
        return weight;
    }
    
    void updateWeightsAndBiases(double learnRate, double momentumFactor)
    {
        Layer *currentLayer = firstLayer->nextLayer;
        while(currentLayer != nullptr)
        {
            for(uint16_t i = 0; i < currentLayer->ctNeurons; i++)
            {
                for(uint16_t j = 0; j < currentLayer->neurons[i].ctConnectionsIn; j++)
                {
                    double weightChange = learnRate * currentLayer->neurons[i].gradientW * dWin_dW(*currentLayer->neurons[i].connectionsIn[j].inputVal);
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
        uint8_t layerSpecifier = (rand() % (ctLayers-1)) + 1; //select a random layer
        //Also give the chance that no layer is mutated (By excluding the first and last layer)
        if(layerSpecifier == 0) return; //Don't mutate the input layer
        else if(layerSpecifier == ctLayers - 1) return; //Don't mutate the output layer

        Layer *currentLayer = firstLayer; //Get the first layer
        for(uint16_t i = 0; i < layerSpecifier; i++) 
            currentLayer = currentLayer->nextLayer; //Get the specified random layer
        currentLayer->mutate(mutationRate); //Mutate the specified layer
    }
    void learn(double *expected) //Improve the network based on the defined cost function and expected outputs
    {
        Layer *currentLayer = firstLayer;
        while(currentLayer->nextLayer != nullptr) currentLayer = currentLayer->nextLayer; //Get last layer
        Layer *lastLayer = currentLayer;

        while(currentLayer->prevLayer != nullptr)
        {
            for(uint16_t i = 0; i < currentLayer->ctNeurons; i++)
            {
                double output = currentLayer->neurons[i].outputVal;
                double w_in = currentLayer->neurons[i].inputVal;

                if(currentLayer->nextLayer == nullptr)
                {
                    cost += costFunction->cost(currentLayer->neurons[i].outputVal, expected[i]);
                    currentLayer->neurons[i].gradientW = dcost_dout(expected[i], output) * dOut_dWin(currentLayer->neurons[i], w_in);
                }
                
                for(uint16_t con = 0; con < currentLayer->neurons[i].ctConnectionsIn; con++)
                {
                    double weight = currentLayer->neurons[i].connectionsIn[con].weight;
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
        uint16_t layerNum = 0;
        while(currentLayer != nullptr)
        {
            std::cout << "Layer " << layerNum << std::endl;
            currentLayer->print();
            currentLayer = currentLayer->nextLayer;
            layerNum++;
        }
    }
    
    void exportNetwork(std::string fileName) //Export the network to a file (For later use) and for analysis
    {
        std::ofstream file;
        file.open(fileName);
        Layer *currentLayer = firstLayer;
        uint16_t layerNum = 0;
        file << "Cost: " << cost << std::endl;
        while(currentLayer != nullptr)
        {
            file << "Layer" << layerNum << ": " << currentLayer->ctNeurons << std::endl;
            currentLayer->exportToFile(file);
            currentLayer = currentLayer->nextLayer;
            layerNum++;
        }
        file.close();
    }

    //Import a network from a file

    void getConnections(std::string str, Layer *currentLayer)
    {
        uint32_t globalStart = 0;
        uint32_t connectionIndex = 0;
        uint32_t neuronIndex = 0;
        Neuron *currentNeuron = currentLayer->neurons;
        if(currentLayer->prevLayer == nullptr) //If the current layer is the input layer
        {
            return;
        }

        while(str.find(",", globalStart) < str.length()) //While there are still connections to be found
        {
            uint32_t start = str.find(", ", globalStart) + 2; //Get the start of the connection
            uint32_t end = str.find(", ", start); //Get the end of the connection
            uint32_t newline = str.find("\n", start); //Get the end of the line
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
                double weightVal = std::stod(weight); //Convert the weight to a double
                currentNeuron->connectionsIn[connectionIndex].weight = weightVal;
                currentNeuron->connectionsIn[connectionIndex].fromNeuron = currentLayer->prevLayer->neurons + connectionIndex;
                currentNeuron->connectionsIn[connectionIndex].toNeuron = currentNeuron;
                currentNeuron->connectionsIn[connectionIndex].inputVal = &currentLayer->prevLayer->neurons[connectionIndex].outputVal;
                currentNeuron->connectionsIn[connectionIndex].outputVal = &currentNeuron->inputVal;
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
                    // currentNeuron->connectionsIn = new connection[currentNeuron->ctConnectionsIn+1];
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
        ActivationFunctionType activationFunction = (ActivationFunctionType)std::stoi(str.substr(str.find("\n",layerIndex) + 1, str.find(",",layerIndex) - str.find("\n",layerIndex) - 1));

        if(activationFunction < NONE) 
        {
            Layer *newLayer = addLayer(std::stoi(ctNeurons), activationFunction);
            return newLayer;
        }
        else
        {
            std::cout << "ERROR: Activation function not found" << std::endl;
            return nullptr;
        }
    }
    void importNetwork(std::string fileName)
    {
        std::ifstream file;
        file.open(fileName);
        std::string lines;
        std::string line;
        
        while(std::getline(file, line))
        {
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
};