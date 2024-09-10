#include "networkV2.hpp"
#include <cstring>

NetworkV2::NetworkV2()
{
    firstLayer = nullptr;
    lastLayer = nullptr;
    ctLayers = 0;
    #ifdef useGPU
    cudaStreamCreate(&currentInstance);
    #endif
}
NetworkV2::~NetworkV2()
{
    LayerV2 *currentLayer = firstLayer;
    while(currentLayer != nullptr)
    {
        LayerV2 *nextLayer = currentLayer->next;
        delete currentLayer;
        currentLayer = nextLayer;
    }
    #ifdef useGPU
    cudaStreamDestroy(currentInstance);
    #endif
}
LayerV2 *NetworkV2::addLayer(count_t size, ActivationFunctionType activationFunction)
{
    ctLayers++;
    if(firstLayer == nullptr)
    {
        firstLayer = new LayerV2(size, 0, activationFunction);
        lastLayer = firstLayer;
        return firstLayer;
    }
    else
    {
        LayerV2 *tempLayer = new LayerV2(size, lastLayer->size, activationFunction);
        lastLayer->next = tempLayer;
        lastLayer = tempLayer;
        return tempLayer;
    }
}
void NetworkV2::feedThrough(in_out_t *inputs)
{
    LayerV2 *currentLayer = firstLayer;
    in_out_t *outputsOfLastLayer = inputs;
    while(currentLayer != nullptr)
    {
        #ifdef useGPU
        currentLayer->feedThrough(outputsOfLastLayer, currentInstance);
        #else
        currentLayer->feedThrough(outputsOfLastLayer);
        #endif
        outputsOfLastLayer = currentLayer->activations;
        currentLayer = currentLayer->next;
    }
}
void NetworkV2::exportNetwork(std::string fileName, bool humanReadable)
{
    std::ofstream file;
    file.open(fileName);
    if (!file.good())
    {
        printf("File could not be opened\n");
        exit(1);
    }
    LayerV2 *currentLayer = firstLayer;
    count_t layerNum = 0;
    while(currentLayer != nullptr)
    {
        file << "Layer" << layerNum << ": " << currentLayer->size << "\n";
        currentLayer->exportToFile(file, humanReadable);
        currentLayer = currentLayer->next;
        layerNum++;
        file << "\n";
    }
    file.close();
}

//Import a network from a file
void NetworkV2::getConnections(std::string str, LayerV2 *currentLayer)
{
    std::size_t globalStart = 0;
    count_t connectionIndex = 0;
    count_t currentNeuron = 0;
    if(currentLayer == firstLayer) //If the current layer is the input layer
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
            memcpy(&weightValfloat, &weightValInt, sizeof(weight_t));
            
            if(connectionIndex == currentLayer->prevLayerSize) //If bias
            {
                currentLayer->biases[currentNeuron] = weightValfloat;
            }
            else //if normal connection
            {
                currentLayer->weights[currentNeuron][connectionIndex] = weightValfloat;
            }
            connectionIndex++;
            globalStart = end;
            if(connectionIndex == currentLayer->prevLayerSize+1)
            {
                connectionIndex = 0; //Reset the connection index
                currentNeuron++;    //Move to the next neuron
                if(currentNeuron == currentLayer->size) //If the current layer has no more neurons
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
LayerV2 *NetworkV2::parseLayer(std::string str)
{
    count_t layerIndex = static_cast<count_t>(str.find("Layer"));
    if(layerIndex == std::string::npos)
    {
        return nullptr;
    }
    std::string layerNum = str.substr(layerIndex + 5, str.find(":", layerIndex+5) - layerIndex - 5);
    std::string ctNeurons = str.substr(str.find(":",layerIndex) + 2, str.find("\n", layerIndex) - str.find(":", layerIndex) - 2);
    layerIndex = static_cast<count_t>(str.find("\n", layerIndex+1));
    LayerV2 *newLayer = addLayer(count_t(std::stoi(ctNeurons)), NONE);
    for(count_t i = 0; i < newLayer->size; i++)
    {
        std::string actString = str.substr(str.find("\n",layerIndex) + 1, str.find(",",layerIndex) - str.find("\n",layerIndex) - 1);
        ActivationFunctionType activationFunction = (ActivationFunctionType)std::stoi(actString);
        layerIndex = static_cast<count_t>(str.find("\n", layerIndex+1));
        if(activationFunction < NONE) 
        {
            newLayer->actiFun[i] = activationFunction;
        }
        else
        {
            printf("Activation function not found\n");
            throw std::system_error();
            return nullptr;
        }
    }
    return newLayer;
}
void NetworkV2::importNetwork(std::string fileName)
{
    std::ifstream file;
    file.open(fileName);
    if(file.good())
    {
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
                LayerV2 *newLayer = parseLayer(lines);
                if(newLayer != nullptr)
                {
                    getConnections(lines, newLayer);
                }
                lines = "";
            }
        }
    }
    file.close();
}

void NetworkV2::mutate(double mutationRate)
{
    uint8_t layerSpecifier = uint8_t((rand() % (ctLayers-1)) + 1); //select a random layer
    //Also give the chance that no layer is mutated (By excluding the first and last layer)
    if(layerSpecifier == 0) return; //Don't mutate the input layer

    LayerV2 *currentLayer = firstLayer; //Get the first layer
    for(count_t i = 0; i < layerSpecifier; i++) 
        currentLayer = currentLayer->next; //Get the specified random layer
    currentLayer->mutate((weight_t)mutationRate); //Mutate the specified layer
}