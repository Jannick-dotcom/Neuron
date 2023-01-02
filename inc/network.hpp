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
    uint16_t ctLayers;
    double cost;
    Cost *costFunction;
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

    Layer *addLayer(long ctNeurons, ActivationFunction *activationFunction)
    {
        if(firstLayer == nullptr || ctLayers == 0)
        {
            firstLayer = new Layer(ctNeurons, nullptr, activationFunction);
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
            currentLayer->nextLayer = new Layer(ctNeurons, currentLayer, activationFunction);
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
        return n.activationFunction->derivative(w_in);
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

    void learn(double *expected)
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
                double doutdwin = currentLayer->neurons[i].activationFunction->derivative(w_in);

                if(currentLayer->nextLayer == nullptr)
                {
                    cost += costFunction->cost(currentLayer->neurons[i].outputVal, expected[i]);
                    currentLayer->neurons[i].gradientW = dcost_dout(expected[i], output) * dOut_dWin(currentLayer->neurons[i], w_in);
                }
                
                for(uint16_t con = 0; con < currentLayer->neurons[i].ctConnectionsIn; con++)
                {
                    double weight = currentLayer->neurons[i].connectionsIn[con].weight;
                    double input = *currentLayer->neurons[i].connectionsIn[con].inputVal;
                    currentLayer->neurons[i].connectionsIn[con].fromNeuron->gradientW += currentLayer->neurons[i].gradientW * dWin_dIn(weight) * dOut_dWin(currentLayer->neurons[i], w_in);
                }
            }
            currentLayer = currentLayer->prevLayer;
        }
        cost = cost / lastLayer->ctNeurons;
    }
    void print()
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
    void exportNetwork(std::string fileName)
    {
        std::ofstream file;
        file.open(fileName);
        Layer *currentLayer = firstLayer;
        uint16_t layerNum = 0;
        while(currentLayer != nullptr)
        {
            file << "Layer" << layerNum << ": " << currentLayer->ctNeurons << std::endl;
            currentLayer->exportToFile(file);
            currentLayer = currentLayer->nextLayer;
            layerNum++;
        }
        file.close();
    }

    void getConnections(std::string str, Layer *currentLayer)
    {
        uint32_t globalStart = 0;
        uint32_t connectionIndex = 0;
        uint32_t neuronIndex = 0;
        Neuron *currentNeuron = currentLayer->neurons;
        if(currentLayer->prevLayer != nullptr)
        {
            currentNeuron->ctConnectionsIn = currentLayer->prevLayer->ctNeurons+1;
            currentNeuron->connectionsIn = new connection[currentNeuron->ctConnectionsIn+1];
        }
        else
        {
            return;
        }

        while(str.find(",", globalStart) < str.length())
        {
            uint32_t start = str.find(", ", globalStart) + 2;
            uint32_t end = str.find(", ", start);
            uint32_t newline = str.find("\n", start);
            if(start < globalStart)
            {
                break;
            }
            if(newline < end)
            {
                end = newline;
            }

            if(start != -1 && end != -1)
            {
                std::string weight = str.substr(start, end - start);
                double weightVal = std::stod(weight);
                currentNeuron->connectionsIn[connectionIndex].weight = weightVal;
                currentNeuron->connectionsIn[connectionIndex].fromNeuron = currentLayer->prevLayer->neurons + connectionIndex;
                currentNeuron->connectionsIn[connectionIndex].toNeuron = currentNeuron;
                currentNeuron->connectionsIn[connectionIndex].inputVal = &currentLayer->prevLayer->neurons[connectionIndex].outputVal;
                currentNeuron->connectionsIn[connectionIndex].outputVal = &currentNeuron->inputVal;
                connectionIndex++;
                globalStart = end;
                if(connectionIndex == currentNeuron->ctConnectionsIn)
                {
                    connectionIndex = 0;
                    currentNeuron++;
                    neuronIndex++;
                    if(neuronIndex == currentLayer->ctNeurons)
                    {
                        break;
                    }
                    currentNeuron->connectionsIn = new connection[currentNeuron->ctConnectionsIn+1];
                }
            }
            else
            {
                break;
            }
        }
    }

    Layer *parseLayer(std::string str)
    {
        if(str.find("Layer") == std::string::npos)
        {
            return nullptr;
        }

        std::string layerNum = str.substr(str.find("Layer") + 5, str.find(":") - str.find("Layer") - 5);
        std::string ctNeurons = str.substr(str.find(":") + 2, str.find("\n") - str.find(":") - 2);
        std::string activationFunction = str.substr(str.find("\n") + 1, str.find(",") - str.find("\n") - 1);

        if(activationFunction == "Linear") 
        {
            Linear *CLASSactivationFunction = new Linear();
            Layer *newLayer = addLayer(std::stoi(ctNeurons), CLASSactivationFunction);
            return newLayer;
        }
        else if(activationFunction == "Sigmoid")
        {
            Sigmoid *CLASSactivationFunction = new Sigmoid();
            Layer *newLayer = addLayer(std::stoi(ctNeurons), CLASSactivationFunction);
            return newLayer;
        }
        else if(activationFunction == "Tanh")
        {
            Tanh *CLASSactivationFunction = new Tanh();
            Layer *newLayer = addLayer(std::stoi(ctNeurons), CLASSactivationFunction);
            return newLayer;
        }
        else if(activationFunction == "ReLU")
        {
            ReLU *CLASSactivationFunction = new ReLU();
            Layer *newLayer = addLayer(std::stoi(ctNeurons), CLASSactivationFunction);
            return newLayer;
        }
        else if(activationFunction == "LeakyReLU")
        {
            LeakyReLU *CLASSactivationFunction = new LeakyReLU();
            Layer *newLayer = addLayer(std::stoi(ctNeurons), CLASSactivationFunction);
            return newLayer;
        }
        else if(activationFunction == "Softmax")
        {
            Softmax *CLASSactivationFunction = new Softmax();
            Layer *newLayer = addLayer(std::stoi(ctNeurons), CLASSactivationFunction);
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
                getConnections(lines, newLayer);
                lines = "";
            }
        }

        file.close();
    }
};