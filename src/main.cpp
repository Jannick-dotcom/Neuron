#include "network.hpp"
#include "networkV2.hpp"
#include <iostream>
#include <random>

#include "neuronTypes.hpp"

dataPoint points;

#include <fstream>

std::ofstream file;

void doLearning(Network *net)
{
    in_out_t possibleCombinations[100][2];
    for(count_t i = 0; i < sizeof(possibleCombinations) / sizeof(possibleCombinations[0]); i++)
    {
        possibleCombinations[i][0] = (rand() % 100) / 100.0;
        possibleCombinations[i][1] = (rand() % 100) / 100.0;
    }
    points.inputs = new in_out_t[net->firstLayer->ctNeurons];
    points.expectedOutputs = new in_out_t[net->outputLayer->ctNeurons];
    weight_t learningRate = 0.0001;
    weight_t lastCost = 0;
    do
    {
        lastCost = net->cost;
        net->cost = 0;
        for (count_t i = 0; i < sizeof(possibleCombinations) / sizeof(possibleCombinations[0]); i++)
        {
            points.inputs[0] = possibleCombinations[i][0];
            points.inputs[1] = possibleCombinations[i][1];
            points.expectedOutputs[0] = points.inputs[0] + points.inputs[1];

            net->feedThrough(points);
            net->learn(points.expectedOutputs);
            net->updateWeightsAndBiases(learningRate, 0.001);
        }
        // net->print();
        std::cout << "Cost: " << net->cost << "\n";
    } while(net->cost > 0.1 || lastCost == 0);
    net->exportNetwork("network.txt");
    delete[] points.inputs;
    delete[] points.expectedOutputs;
}

Network *generateMutatedNets(Network *net, count_t countAgents)
{
    Network *agents = new Network[countAgents];
    //////////////////////////////////////////////////
    //TODO: Improve copying of networks!!
    //////////////////////////////////////////////////
    net->exportNetwork("transfer.txt");
    for(count_t i = 0; i < countAgents; i++)
    {
        agents[i].importNetwork("transfer.txt");
        agents[i].costType = net->costType;
        agents[i].mutate(0.01);
    }
    remove("transfer.txt");
    return agents;
}

void doMutating(Network *net, count_t countAgents, count_t countGenerations)
{
    count_t bestNet = 0;
    Network *agents = generateMutatedNets(net, countAgents);
    for(count_t i = 0; i < countGenerations; i++)
    {
        for(count_t agent = 0; agent < countAgents; agent++)
        {
            //////////////////////////////////////////////////
            //TODO: Do Task and save Result!!
            agents[agent].firstLayer->neurons[0].inputVal = 0.5;
            agents[agent].firstLayer->neurons[1].inputVal = 0;
            agents[agent].feedThrough();
            agents[agent].cost = costFunction(agents[agent].costType, agents[agent].outputLayer->neurons[0].outputVal, 0.2);
            //////////////////////////////////////////////////
            if(agents[agent].cost < agents[bestNet].cost) //If performance of agent is better than the current best
            {
                bestNet = agent; //Set the current best to the agent
            }
        }
        Network *tempAgents = generateMutatedNets(&agents[bestNet], countAgents); //Generate new agents based on the best agent
        delete[] agents; //Delete the agents
        agents = tempAgents; //Set the agents to the new agents
    }
    agents[bestNet].exportNetwork("best.txt"); //Print the best agent
    delete[] agents;
}

void test()
{
    in_out_t inputs[2];
    for(int i = 0; i < 10000; i++)
    {
        inputs[0] = i % 20;
        inputs[1] = i;
        NetworkV2 *net = new NetworkV2();
        net->addLayer(2, LINEAR);
        net->addLayer(20, SIGMOID);
        net->addLayer(5, RELU);
        net->addLayer(1, SIGMOID);
        net->feedThrough(inputs);
        double cost = costFunction(CostFunctionType::CostQUADRATIC, net->lastLayer->activations[0], 0.2);
        printf("%lf\n", cost);
        delete net;
    }
}

int main()
{
    test();
    return 0;
    Network *net = new Network(CostQUADRATIC);
    // net->importNetwork("network.txt");
    net->addLayer(2, LINEAR);
    net->addLayer(20, RELU);
    net->addLayer(5, RELU);
    net->addLayer(1, LINEAR);
    net->cost = 100;
    // doMutating(net, 100, 50);
    doLearning(net);
    delete net;
    return 0;
}