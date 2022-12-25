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

int main(){
    double cost = 0;
    dataPoint points;
    Network *net = new Network();
    Layer *first = net->addLayer(2);
    Layer *middle = net->addLayer(3);
    Layer *last = net->addLayer(1); 
    points.inputs = new double[first->ctNeurons];
    points.expectedOutputs = new double[last->ctNeurons];
    for(uint32_t l = 0; l<100000; l++)
    {
        cost = 0;
        points.inputs[0] = rand() % 2;
        points.inputs[1] = rand() % 2;
        points.expectedOutputs[0] = points.inputs[0] && points.inputs[1];
        
        net->feedThrough(points);
        net->learn(points.expectedOutputs);
        net->updateWeightsAndBiases(0.01);
    }
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