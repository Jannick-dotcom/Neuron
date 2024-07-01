#include "activationFun.hpp"
class LayerV2
{
public:
    LayerV2 *next;
    weight_t **weights; //incoming weights
    weight_t *biases; //incoming biases
    ActivationFunctionType *actiFun; //activation functions
    in_out_t *activations; //current activations
    int size = 0;
    int prevLayerSize = 0;
    LayerV2(int size, int prevLayerSize, ActivationFunctionType activationFunction)
    {
        actiFun = new ActivationFunctionType[size];
        for(int i = 0; i < size; i++)
        {
            actiFun[i] = activationFunction;
        }
        if(prevLayerSize > 0)
        {
            weights = new weight_t*[size];
            for(int i = 0; i < size; i++)
            {
                weights[i] = new weight_t[prevLayerSize];
                for(int j = 0; j < prevLayerSize; j++)
                {
                    weights[i][j] = weight_t(rand()) / weight_t(RAND_MAX) - weight_t(0.5);
                }
            }
            biases = new weight_t[size];
            for(int i = 0; i < size; i++)
            {
                biases[i] = weight_t(rand()) / weight_t(RAND_MAX) - weight_t(0.5);
            }
        }
        activations = new in_out_t[size];
        this->size = size;
        this->prevLayerSize = prevLayerSize;
    }
    ~LayerV2()
    {
        for(int i = 0; i < size; i++)
        {
            delete[] weights[i];
        }
        delete[] weights;
        delete[] biases;
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
                file << "\tBias:" << biases[i];
            }
            file << "\n";
        }
    }
};

class NetworkV2
{
public:
    LayerV2 *firstLayer;
    LayerV2 *lastLayer;
    int ctLayers;
    NetworkV2()
    {
        firstLayer = nullptr;
        lastLayer = nullptr;
    }
    void addLayer(int size, ActivationFunctionType activationFunction)
    {
        if(firstLayer == nullptr)
        {
            firstLayer = new LayerV2(size, 0, activationFunction);
            lastLayer = firstLayer;
        }
        else
        {
            lastLayer->next = new LayerV2(size, lastLayer->size, activationFunction);
            lastLayer = lastLayer->next;
        }
    }
    void feedThrough(in_out_t *inputs)
    {
        LayerV2 *currentLayer = firstLayer;
        in_out_t *outputsOfLastLayer = inputs;
        int sizeOfLastLayer = 0;
        while(currentLayer != nullptr)
        {
            for(int i = 0; i < currentLayer->size; i++)
            {
                in_out_t neuronInput = 0;
                if(sizeOfLastLayer > 0) 
                {
                    neuronInput = currentLayer->biases[i];
                    for(int iWeights = 0; iWeights < sizeOfLastLayer; iWeights++)
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
        }
        file.close();
    }
};