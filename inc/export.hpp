#include "network.hpp"
#include <fstream>
std::ofstream file;
std::string tab = "\t";
extern std::string fileOutputPath;

void exportConnections(count_t layerIndex, Layer *runningLayer)
{
    file << "connection c" << layerIndex 
    << "[" << runningLayer->ctNeurons+1 << "]" 
    << "[" << runningLayer->neurons[0].ctConnectionsIn << "];\n";

    file << "const weight_t w" << layerIndex << "[][" << runningLayer->neurons[0].ctConnectionsIn << "] = {";
    for(count_t neuronIndex = 0; neuronIndex < runningLayer->ctNeurons; neuronIndex++)
    {
        file << runningLayer->neurons[neuronIndex].connectionsIn[0].weight;
        for(count_t i = 1; i < runningLayer->neurons[neuronIndex].ctConnectionsIn; i++)
        {
            file << ", " << runningLayer->neurons[neuronIndex].connectionsIn[i].weight;
        }
        file << ",";
    }
    file << "};\n";
}

void exportNeurons(count_t layerIndex, Layer *runningLayer)
{
    file << "Neuron n" << layerIndex << "[" << runningLayer->ctNeurons+1 << "];\n";

    file << "ActivationFunctionType aT" << layerIndex << "[] = {";
    file << "ActivationFunctionType(" << runningLayer->neurons[0].type << ")";
    for(count_t i = 1; i < runningLayer->ctNeurons+1; i++)
    {
        file << ", ActivationFunctionType(" << runningLayer->neurons[i].type << ")";
    }
    file << "};\n";

    if(runningLayer->neurons[0].ctConnectionsIn > 0)
    {
        exportConnections(layerIndex, runningLayer);
    }
}


std::string generateInitConnections(count_t layerIndex)
{
    std::stringstream temp;
    temp << tab << tab << "for(count_t c = 0; c < n" << layerIndex << "[n].ctConnectionsIn; c++) {\n";
    temp << tab << tab << tab << "c" << layerIndex << "[n][c].weight = w" << layerIndex << "[n][c];\n";
    temp << tab << tab << tab << "c" << layerIndex << "[n][c].inputVal = " << "&n" << layerIndex-1 << "[c].outputVal;\n";
    temp << tab << tab << tab << "c" << layerIndex << "[n][c].fromNeuron = " << "&n" << layerIndex-1 << "[c];\n";
    temp << tab << tab << tab << "c" << layerIndex << "[n][c].outputVal = &n" << layerIndex << "[n].inputVal;\n";
    temp << tab << tab << tab << "c" << layerIndex << "[n][c].toNeuron = &n" << layerIndex << "[n];\n";
    temp << tab << tab << "}\n";
    
    return temp.str();
}

std::string generateInitMethod(count_t layerIndex, Layer *runningLayer)
{
    std::stringstream temp;
    temp << "void init_" << layerIndex << "(Layer &l){\n";
    temp << "\tstatic const count_t neuronCt = sizeof(n" << layerIndex << ")/sizeof(Neuron);\n";
    if(layerIndex > 0)
    {
        temp << tab << "l.prevLayer = &l" << layerIndex-1 << ";\n";
        temp << tab << "l.prevLayer->nextLayer = &l" << layerIndex << ";\n";
    }
    temp << "\tfor(count_t n = 0; n < neuronCt-1; n++) {\n";

    temp << "\t\tn" << layerIndex << "[n].type = aT" << layerIndex << "[n];\n";
    temp << "\t\tn" << layerIndex << "[n].ctConnectionsIn = " << (layerIndex > 0 ? runningLayer->prevLayer->ctNeurons+1 : 0) << ";\n";
    if(layerIndex > 0)
    {
        temp << generateInitConnections(layerIndex);
        temp << "\t\tn" << layerIndex << "[n].connectionsIn = c" << layerIndex << "[n];\n";
    }

    temp << "\t}\n";
    temp << "\tn" << layerIndex << "[neuronCt-1].ctConnectionsIn = 0;\n";
    temp << "\tn" << layerIndex << "[neuronCt-1].connectionsIn = nullptr;\n";
    temp << "\tn" << layerIndex << "[neuronCt-1].inputVal = 1;\n";
    temp << "\tl.neurons = n" << layerIndex << ";\n";
    
    temp << "}\n";
    return temp.str();
}

void exportNetworkToCpp(Network &net)
{
    file.open(fileOutputPath + "output.cpp");
    Layer *runningLayer = net.firstLayer;
    count_t layerIndex = 0;
    
    file << "#include <network.hpp>\n";
    file << "Network net;\n";
    std::string initMethods;
    std::string mainMethod = "int main(){\n";
    for(layerIndex = 0; runningLayer != nullptr; layerIndex++)
    {
        file << "\nLayer l" << layerIndex << ";\n";
        exportNeurons(layerIndex, runningLayer);
        initMethods += generateInitMethod(layerIndex, runningLayer);
        mainMethod += "\tinit_" + std::to_string(layerIndex) + "(l" + std::to_string(layerIndex) + ");\n";

        runningLayer = runningLayer->nextLayer;
    }
    mainMethod += "\tnet.firstLayer = &l0;\n";
    mainMethod += "\tnet.outputLayer = &l" + std::to_string(layerIndex-1) + ";\n";
    mainMethod += "}";
    file << "\n\n";
    file << initMethods;
    file << mainMethod;
    file.close();
}