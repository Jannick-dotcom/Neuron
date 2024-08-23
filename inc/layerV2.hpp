#ifndef layerv2_h
#define layerv2_h

#include "neuronTypes.hpp"
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
                if (weights[i] != NULL) delete[] weights[i];
            }
            if (weights != NULL) delete[] weights;
            if (biases != NULL) delete[] biases;
        }
        if (actiFun != NULL) delete[] actiFun;
        if (activations != NULL) delete[] activations;
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
                printf("ERROR Neuron Type: %d", actiFun[i]);
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
        next->prevLayerSize = static_cast<count_t>(size+1);
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
        count_t newIndex = 0;
        for(count_t i = 0; i < size; i++) //iterate over every old neuron
        {
            if(i == neuronIndex) continue;
            newWeights[newIndex] = weights[i];
            newBiases[newIndex] = biases[i];
            newActiFuns[newIndex] = actiFun[i];
            newIndex++;
        }

        //fix connections of next layer
        for(count_t neuron = 0; neuron < next->size; neuron++)
        {
            weight_t *nextLayerNewWeights = new weight_t[size-1];
            count_t newWeightIndex = 0;
            for(count_t conn = 0; conn < this->size; conn++)
            {
                if(conn == neuronIndex) continue;
                nextLayerNewWeights[newWeightIndex] = next->weights[neuron][conn];
                newWeightIndex++;
            }
            delete[] next->weights[neuron];
            next->weights[neuron] = nextLayerNewWeights;
        }
        next->prevLayerSize = static_cast<count_t>(size-1);
        ////////////////////////////////

        if(weights == nullptr || biases == nullptr || actiFun == nullptr || activations == nullptr)
        {
            printf("Bitch\n");
        }
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

    #ifdef useGPU
    __global__ friend void feedThroughGPU(LayerV2 *currentLayer, in_out_t *inputs, count_t sizeOfLastLayer);
    #endif

    void feedThrough(in_out_t *inputs, count_t sizeOfLastLayer)
    {
        #ifdef useGPU
        feedThroughGPU<<<1, size>>>(this, inputs, sizeOfLastLayer);
        cudaDeviceSynchronize();
        #else
        for(count_t i = 0; i < size; i++)
        {
            in_out_t weightedSum = 0;
            if(sizeOfLastLayer > 0) 
            {
                weightedSum = biases[i];
                for(count_t iWeights = 0; iWeights < sizeOfLastLayer; iWeights++)
                {
                    weightedSum += inputs[iWeights] * weights[i][iWeights];
                }
            }
            else
            {
                weightedSum = inputs[i];
            }
            activations[i] = activationFunction(actiFun[i], weightedSum); //make ReLu
        }
        #endif
    }
    #ifdef useGPU
    __host__ void* operator new(size_t size)
    {
        void *temp;
        cudaError_t ret = cudaMallocManaged(&temp, size);
        if(ret != cudaError::cudaSuccess)
        {
            printf("Layer: Malloc failed with code %d\n", ret);
            exit(1);
        }
        return temp;
    }
    __host__ void operator delete(void* ptr)
    {
        cudaError_t ret = cudaFree(ptr);
        if(ret != cudaError::cudaSuccess)
        {
            printf("Layer: delete failed with code %d\n", ret);
            exit(1);
        }
    }
    __host__ void operator delete[](void* ptr)
    {
        cudaError_t ret = cudaFree(ptr);
        if(ret != cudaError::cudaSuccess)
        {
            printf("Layer: delete[] failed with code %d\n", ret);
            exit(1);
        }
    }
    #endif
};

#endif