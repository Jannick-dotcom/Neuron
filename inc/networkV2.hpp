#include "activationFun.hpp"
class LayerV2
{
public:
    LayerV2 *next;
    weight_t **weights; //incoming weights
    weight_t *biases; //incoming biases
    ActivationFunctionType *actiFun; //activation functions
    in_out_t *activations; //current activations
    count_t size = 0;
    count_t prevLayerSize = 0;
    LayerV2(count_t size, count_t prevLayerSize, ActivationFunctionType activationFunction)
    {
        biases = nullptr;
        actiFun = new ActivationFunctionType[size];
        for(count_t i = 0; i < size; i++)
        {
            actiFun[i] = activationFunction;
        }
        if(prevLayerSize > 0)
        {
            weights = new weight_t*[size];
            for(count_t i = 0; i < size; i++)
            {
                weights[i] = new weight_t[prevLayerSize];
                for(count_t j = 0; j < prevLayerSize; j++)
                {
                    weights[i][j] = weight_t(rand()) / weight_t(RAND_MAX) - weight_t(0.5);
                }
            }
            biases = new weight_t[size];
            for(count_t i = 0; i < size; i++)
            {
                biases[i] = weight_t(rand()) / weight_t(RAND_MAX) - weight_t(0.5);
            }
        }
        activations = new in_out_t[size];
        this->size = size;
        this->prevLayerSize = prevLayerSize;
        this->next = nullptr;
    }
    ~LayerV2()
    {
        if(prevLayerSize > 0)
        {
            for(count_t i = 0; i < size; i++)
            {
                delete[] weights[i];
            }
            delete[] weights;
            delete[] biases;
        }
        delete[] actiFun;
        delete[] activations;
    }
    void exportToFile(std::ofstream &file, bool humanReadable)
    {
        for (count_t i = 0; i < size; i++)
        {
            if(actiFun[i] < NONE)
            {
                file << actiFun[i];
            }
            else if(actiFun[i] > NONE)
            {
                std::cout << "ERROR Neuron Type: " << actiFun[i] << "\n";
                throw std::system_error();
                exit(1);
            }
            for (count_t c = 0; c < prevLayerSize; c++)
            {
                if(!humanReadable)
                {
                    uint64_t iWeight;
                    memcpy(&iWeight, &(weights[i][c]), sizeof(weights[i][c]));
                    file << ", " << iWeight;
                }
                else
                {
                    file << ", " << weights[i][c];
                }
            }
            if(biases != nullptr)
            {
                uint64_t iWeight;
                memcpy(&iWeight, &(biases[i]), sizeof(biases[i]));
                file << ", " << iWeight; //last weight is always a bias
            }
            file << "\n";
        }
    }
    
    void addNeuron(ActivationFunctionType type)
    {
        weight_t **newWeights = new weight_t*[size+1];
        weight_t *newBiases = new weight_t[size+1];
        ActivationFunctionType *newActiFuns = new ActivationFunctionType[size+1];
        for(count_t i = 0; i < size; i++) //iterate over every old neuron
        {
            newWeights[i] = weights[i];
            newBiases[i] = biases[i];
            newActiFuns[i] = actiFun[i];
        }
        newWeights[size] = new weight_t[prevLayerSize];

        for(count_t conn = 0; conn < prevLayerSize; conn++)
        {
            newWeights[size][conn] = weight_t(rand()) / weight_t(RAND_MAX) - weight_t(0.5);
        }

        newBiases[size] = weight_t(rand()) / weight_t(RAND_MAX) - weight_t(0.5);
        newActiFuns[size] = type;

        //fix connections of next layer
        for(count_t neuron = 0; neuron < next->size; neuron++)
        {
            weight_t *nextLayerNewWeights = new weight_t[size+1];
            for(count_t conn = 0; conn < this->size; conn++)
            {
                nextLayerNewWeights[conn] = next->weights[neuron][conn];
            }
            nextLayerNewWeights[this->size] = weight_t(rand()) / weight_t(RAND_MAX) - weight_t(0.5);
            delete[] next->weights[neuron];
            next->weights[neuron] = nextLayerNewWeights;
        }
        next->prevLayerSize = size+1;
        ////////////////////////////////


        in_out_t *newActivations = new in_out_t[size+1]; //No need to copy these
        delete[] weights;
        delete[] biases;
        delete[] actiFun;
        delete[] activations;
        weights = newWeights;
        biases = newBiases;
        actiFun = newActiFuns;
        activations = newActivations;
        size++;
    }
    void removeNeuron(count_t neuronIndex)
    {
        weight_t **newWeights = new weight_t*[size-1];
        in_out_t *newActivations = new in_out_t[size-1]; //No need to copy these
        weight_t *newBiases = new weight_t[size-1];
        ActivationFunctionType *newActiFuns = new ActivationFunctionType[size-1];
        for(count_t i = 0; i < size; i++) //iterate over every old neuron
        {
            if(i == neuronIndex) continue;
            newWeights[i] = weights[i];
            newBiases[i] = biases[i];
            newActiFuns[i] = actiFun[i];
        }

        //fix connections of next layer
        for(count_t neuron = 0; neuron < next->size; neuron++)
        {
            weight_t *nextLayerNewWeights = new weight_t[size-1];
            for(count_t conn = 0; conn < this->size; conn++)
            {
                if(conn == neuronIndex) continue;
                nextLayerNewWeights[conn] = next->weights[neuron][conn];
            }
            delete[] next->weights[neuron];
            next->weights[neuron] = nextLayerNewWeights;
        }
        next->prevLayerSize = static_cast<count_t>(size-1);
        ////////////////////////////////


        delete[] weights;
        delete[] biases;
        delete[] actiFun;
        delete[] activations;
        weights = newWeights;
        biases = newBiases;
        actiFun = newActiFuns;
        activations = newActivations;
        size--;
    }
    void mutate(weight_t mutationRate)
    {
        //to leave the chance that the layer does not mutate at all, we do modulo n+1
        uint8_t mutationSpecifier; // 0 = add neuron, 1 = remove neuron, 2 = change connection 3 = change activation function
        if(next == nullptr) 
        {
            mutationSpecifier = 2; //if current layer is output layer, force only changing the weights
        }
        else
        {
            mutationSpecifier = uint8_t(rand() % 4); // 0 = add neuron, 1 = remove neuron, 2 = change connection 3 = change activation function
        }
        switch (mutationSpecifier)
        {
            case 0: // add neuron
                addNeuron((ActivationFunctionType)(rand() % ActivationFunctionType::NONE));
                break;
            case 1: // remove neuron
                removeNeuron(count_t(rand() % size));
                break;
            case 2: // change connection
            {
                count_t neuronSpecifier = count_t(rand() % size);
                count_t connectionSpecifier = count_t(rand() % prevLayerSize);
                weight_t weightchange = ((weight_t)rand() / (weight_t)RAND_MAX - (weight_t)0.5) * mutationRate;
                weights[neuronSpecifier][connectionSpecifier] += weightchange;
                break;
            }
            case 3: //change activation function
            {
                count_t neuronSpecifier = count_t(rand() % size);
                ActivationFunctionType newActivationfunction = (ActivationFunctionType)(rand() % ActivationFunctionType::NONE);
                actiFun[neuronSpecifier] = newActivationfunction;
                break;
            }
            default:
                break;
        }
    }
};

class NetworkV2
{
public:
    LayerV2 *firstLayer;
    LayerV2 *lastLayer;
    count_t ctLayers;
    NetworkV2()
    {
        firstLayer = nullptr;
        lastLayer = nullptr;
        ctLayers = 0;
    }
    ~NetworkV2()
    {
        LayerV2 *currentLayer = firstLayer;
        while(currentLayer != nullptr)
        {
            LayerV2 *nextLayer = currentLayer->next;
            delete currentLayer;
            currentLayer = nextLayer;
        }
    }
    LayerV2 *addLayer(count_t size, ActivationFunctionType activationFunction)
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
    void feedThrough(in_out_t *inputs)
    {
        LayerV2 *currentLayer = firstLayer;
        in_out_t *outputsOfLastLayer = inputs;
        count_t sizeOfLastLayer = 0;
        while(currentLayer != nullptr)
        {
            for(count_t i = 0; i < currentLayer->size; i++)
            {
                in_out_t neuronInput = 0;
                if(sizeOfLastLayer > 0) 
                {
                    neuronInput = currentLayer->biases[i];
                    for(count_t iWeights = 0; iWeights < sizeOfLastLayer; iWeights++)
                    {
                        neuronInput += outputsOfLastLayer[iWeights] * currentLayer->weights[i][iWeights];
                    }
                }
                else
                {
                    neuronInput = outputsOfLastLayer[i];
                }
                currentLayer->activations[i] = activationFunction(currentLayer->actiFun[i], neuronInput); //make ReLu
            }
            sizeOfLastLayer = currentLayer->size;
            outputsOfLastLayer = currentLayer->activations;
            currentLayer = currentLayer->next;
        }
    }
    void exportNetwork(std::string fileName, bool humanReadable = false)
    {
        std::ofstream file;
        file.open(fileName);
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
    void getConnections(std::string str, LayerV2 *currentLayer)
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
                std::memcpy(&weightValfloat, &weightValInt, sizeof(weight_t));
                
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
    LayerV2 *parseLayer(std::string str)
    {
        count_t layerIndex = str.find("Layer");
        if(layerIndex == std::string::npos)
        {
            return nullptr;
        }

        std::string layerNum = str.substr(layerIndex + 5, str.find(":", layerIndex+5) - layerIndex - 5);
        std::string ctNeurons = str.substr(str.find(":",layerIndex) + 2, str.find("\n", layerIndex) - str.find(":", layerIndex) - 2);
        layerIndex = str.find("\n", layerIndex+1);
        LayerV2 *newLayer = addLayer(count_t(std::stoi(ctNeurons)), NONE);
        for(count_t i = 0; i < newLayer->size; i++)
        {
            std::string actString = str.substr(str.find("\n",layerIndex) + 1, str.find(",",layerIndex) - str.find("\n",layerIndex) - 1);
            ActivationFunctionType activationFunction = (ActivationFunctionType)std::stoi(actString);
            layerIndex = str.find("\n", layerIndex+1);
            if(activationFunction < NONE) 
            {
                newLayer->actiFun[i] = activationFunction;
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
        if(file.good())
        {
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
                    LayerV2 *newLayer = parseLayer(lines);
                    if(newLayer != nullptr)
                    {
                        getConnections(lines, newLayer);
                    }
                    lines = "";
                }
            }
            file.close();
        }
    }

    void mutate(double mutationRate)
    {
        uint8_t layerSpecifier = uint8_t((rand() % (ctLayers-1)) + 1); //select a random layer
        //Also give the chance that no layer is mutated (By excluding the first and last layer)
        if(layerSpecifier == 0) return; //Don't mutate the input layer

        LayerV2 *currentLayer = firstLayer; //Get the first layer
        for(count_t i = 0; i < layerSpecifier; i++) 
            currentLayer = currentLayer->next; //Get the specified random layer
        currentLayer->mutate((weight_t)mutationRate); //Mutate the specified layer
    }
};