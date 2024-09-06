#include "layerV2.hpp"
#include "neuronTypes.hpp"
#include "activationFun.hpp"


LayerV2::LayerV2(count_t size, count_t prevLayerSize, ActivationFunctionType activationFunction)
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
LayerV2::~LayerV2()
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
void LayerV2::exportToFile(std::ofstream &file, bool humanReadable)
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

void LayerV2::addNeuron(ActivationFunctionType type)
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
void LayerV2::removeNeuron(count_t neuronIndex)
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
void LayerV2::mutate(weight_t mutationRate)
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
        case 4: //change bias
        {
            count_t neuronSpecifier = count_t(rand() % size);
            weight_t weightchange = ((weight_t)rand() / (weight_t)RAND_MAX - (weight_t)0.5) * mutationRate;
            biases[neuronSpecifier] += weightchange;
            break;
        }
        default:
            break;
    }
}

#ifdef useGPU
void LayerV2::feedThrough(in_out_t *inputs, cudaStream_t stream)
#else
void LayerV2::feedThrough(in_out_t *inputs)
#endif
{
    #ifdef useGPU
    if(prevLayerSize > 0)
    {
        // feedThroughGPU<<<1, size, 0, stream>>>(this, inputs, sizeOfLastLayer);
        feedThroughGPU<<<1, size>>>(this, inputs);
        cudaError_t ret = cudaDeviceSynchronize();
        if(ret != cudaError::cudaSuccess)
        {
            printf("Layer: cudaDeviceSynchronize failed with code %d\n", ret);
            exit(1);
        }
    }
    else 
    {
    #endif
    for(count_t i = 0; i < size; i++)
    {
        in_out_t weightedSum = 0;
        if(prevLayerSize > 0) 
        {
            weightedSum = biases[i];
            for(count_t iWeights = 0; iWeights < prevLayerSize; iWeights++)
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
    #ifdef useGPU
    }
    #endif
}