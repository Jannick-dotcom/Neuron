#include "network.hpp"
#include <iostream>
#include <random>
#include "cost.hpp"

dataPoint points;

Sigmoid sigmoid;
ReLU relu;
Softmax softmax;
// Tanh tanh;
LeakyReLU leakyReLU;
Linear linear;

CrossEntropyCost crossEntropy;
QuadraticCost MSE;
ExponentialCost exponential;

double possibleCombinations[100][2];

int main()
{
    for(uint16_t i = 0; i < sizeof(possibleCombinations) / sizeof(possibleCombinations[0]); i++)
    {
        possibleCombinations[i][0] = (rand() % 100) / 100.0;
        possibleCombinations[i][1] = (rand() % 100) / 100.0;
    }
    Network *net = new Network(&MSE);
    Layer *first = net->addLayer(2, &linear);
    net->addLayer(20, &sigmoid);
    Layer *hidden = net->addLayer(40, &sigmoid);
    Layer *last = net->addLayer(1, &sigmoid);
    points.inputs = new double[first->ctNeurons];
    points.expectedOutputs = new double[last->ctNeurons];
    double learningRate = 0.00005;
    double lastCost = 0;
    do
    {
        lastCost = net->cost;
        net->cost = 0;
        for (uint16_t i = 0; i < sizeof(possibleCombinations) / sizeof(possibleCombinations[0]); i++)
        {
            points.inputs[0] = possibleCombinations[i][0];
            points.inputs[1] = possibleCombinations[i][1];
            points.expectedOutputs[0] = int(round(points.inputs[0])) && int(round(points.inputs[1]));

            net->feedThrough(points);
            net->learn(points.expectedOutputs);
            net->updateWeightsAndBiases(learningRate);
        }
        std::system("clear");
        // net->print();
        std::cout << "Cost: " << net->cost << std::endl;
    } while(net->cost > 10);
    std::cout << std::endl;
    for(uint16_t i = 0; i < sizeof(possibleCombinations) / sizeof(possibleCombinations[0]); i++)
    {
        points.inputs[0] = possibleCombinations[i][0];
        points.inputs[1] = possibleCombinations[i][1];
        points.expectedOutputs[0] = int(round(points.inputs[0])) && int(round(points.inputs[1]));
        net->feedThrough(points);
        std::cout << "Input: " << points.inputs[0] << " " << points.inputs[1] << " Expected: " << points.expectedOutputs[0] << " Output: " << last->neurons[0].outputVal << std::endl;
    }
    net->exportNetwork("network.txt");
    delete[] points.inputs;
    delete[] points.expectedOutputs;
    delete net;
    return 0;
}