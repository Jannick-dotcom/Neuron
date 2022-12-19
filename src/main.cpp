#include "network.hpp"
#include <iostream>
#include <random>

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
}

int main()
{
    double totalCost = 0;
    dataPoint points;
    Network *net = new Network();
    Layer *first = net->addLayer(2);
    Layer *middle = net->addLayer(30);
    Layer *last = net->addLayer(1); 
    points.inputs = new double[2];
    points.expectedOutputs = new double[1];
    for (u_int32_t i = 0; i < 200000; i++)
    {
        points.inputs[0] = rand() / (2.0*(double)RAND_MAX);
        points.inputs[1] = rand() / (2.0*(double)RAND_MAX);
        points.expectedOutputs[0] = points.inputs[0] + points.inputs[1];
        net->feedThrough(points);
        for (u_int32_t i = 0; i < last->ctNeurons; i++)
        {
            totalCost += nodeCost(last->neurons[i].outputVal, points.expectedOutputs[i]);
        }
    }
    std::cout << "Total cost: " << totalCost << std::endl;
    delete[] points.inputs;
    delete[] points.expectedOutputs;
    delete net;
    return 0;
}