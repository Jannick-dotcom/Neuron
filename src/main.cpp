#include "network.hpp"
#include <iostream>
#include <random>

double possibleCombinations[7][2] = {{0,0},{0,1},{1,0},{1,1},{0,0},{0,1},{1,0}};

void printLayer(Layer *layer)
{
    for (u_int32_t i = 0; i < layer->ctNeurons; i++)
    {
        std::cout << layer->neurons[i].outputVal << std::endl;
    }
    std::cout << std::endl;
}

double nodeCost(double output, double target)
{
    return pow((output - target), 2); // return squared error
    // return std::abs(output - target); // return cross entropy
}

int main(){
    double totalCost[1];
    double lastOutput = 0;
    dataPoint points;
    Network *net = new Network();
    Layer *first = net->addLayer(2);
    Layer *middle = net->addLayer(1);
    Layer *last = net->addLayer(1); 
    points.inputs = new double[2];
    points.expectedOutputs = new double[1];
    for(uint32_t l = 0; l<100000; l++)
    {
        totalCost[0] = 0;
        for (u_int32_t i = 0; i < sizeof(possibleCombinations) / sizeof(possibleCombinations[0]); i++)
        {
            // totalCost[0] = 0;
            points.inputs[0] = possibleCombinations[i][0];
            points.inputs[1] = possibleCombinations[i][1];
            points.expectedOutputs[0] = points.inputs[0] && points.inputs[1];
            net->feedThrough(points);
            totalCost[0] += nodeCost(last->neurons[0].outputVal, points.expectedOutputs[0]); //add cost of output
            // net->learn(last, points.expectedOutputs[0], last->neurons[0].outputVal);
        }
        net->updateWeightsAndBiases(0.1);
    }
    std::cout << "End total cost: " << totalCost[0] << std::endl;
    points.inputs[0] = 1;
    points.inputs[1] = 0;
    net->feedThrough(points);
    std::cout << "0 1 -> " << last->neurons[0].outputVal << std::endl;
    points.inputs[0] = 0;
    points.inputs[1] = 1;
    net->feedThrough(points);
    std::cout << "1 0 -> " << last->neurons[0].outputVal << std::endl;
    points.inputs[0] = 0;
    points.inputs[1] = 0;
    net->feedThrough(points);
    std::cout << "0 0 -> " << last->neurons[0].outputVal << std::endl;
    points.inputs[0] = 1;
    points.inputs[1] = 1;
    net->feedThrough(points);
    std::cout << "1 1 -> " << last->neurons[0].outputVal << std::endl;
    delete[] points.inputs;
    delete[] points.expectedOutputs;
    delete net;
    return 0;
}