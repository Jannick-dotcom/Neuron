#include "layerV2.hpp"
#include "neuronTypes.hpp"
#include "activationFun.hpp"
#include <cstring>

#ifndef useGPU
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
        weights = new weight_t[size * prevLayerSize];
        for(count_t i = 0; i < size; i++)
        {
            for(count_t j = 0; j < prevLayerSize; j++)
            {
                weights[i * prevLayerSize + j] = weight_t(rand()) / weight_t(RAND_MAX) - weight_t(0.5);
            }
        }
        biases = new weight_t[size];
        for(count_t i = 0; i < size; i++)
        {
            biases[i] = weight_t(rand()) / weight_t(RAND_MAX) - weight_t(0.5);
        }
    }
    this->activations = new in_out_t[size];
    this->size = size;
    this->prevLayerSize = prevLayerSize;
    this->next = nullptr;
}
LayerV2::~LayerV2()
{
    if(prevLayerSize > 0)
    {
        if (weights != NULL) delete[] weights;
        if (biases != NULL) delete[] biases;
    }
    if (actiFun != NULL) delete[] actiFun;
    if (activations != NULL) delete[] activations;
}

void LayerV2::addNeuron(ActivationFunctionType type)
{
    weight_t *newWeights = nullptr;
    weight_t *newBiases = new weight_t[size+1];
    ActivationFunctionType *newActiFuns = new ActivationFunctionType[size+1];
    for(count_t i = 0; i < size; i++) //iterate over every old neuron
    {
        newBiases[i] = biases[i];
        newActiFuns[i] = actiFun[i];
    }
    if(prevLayerSize > 0)
    {
        newWeights = new weight_t[(size+1) * prevLayerSize];
        for(count_t i = 0; i < size; i++)
        {
            for(count_t conn = 0; conn < prevLayerSize; conn++)
            {
                newWeights[i * prevLayerSize + conn] = weights[i * prevLayerSize + conn];
            }
        }
        for(count_t conn = 0; conn < prevLayerSize; conn++)
        {
            newWeights[size * prevLayerSize + conn] = weight_t(rand()) / weight_t(RAND_MAX) - weight_t(0.5);
        }
    }
    newBiases[size] = weight_t(rand()) / weight_t(RAND_MAX) - weight_t(0.5);
    newActiFuns[size] = type;

    //fix connections of next layer
    if(next != nullptr)
    {
        weight_t *nextLayerNewWeights = new weight_t[next->size * (size+1)];
        for(count_t neuron = 0; neuron < next->size; neuron++)
        {
            for(count_t conn = 0; conn < this->size; conn++)
            {
                nextLayerNewWeights[neuron * (size+1) + conn] = next->weights[neuron * this->size + conn];
            }
            nextLayerNewWeights[neuron * (size+1) + this->size] = weight_t(rand()) / weight_t(RAND_MAX) - weight_t(0.5);
        }
        delete[] next->weights;
        next->weights = nextLayerNewWeights;
        next->prevLayerSize = static_cast<count_t>(size+1);
    }

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
    if(size <= 1)
    {
        printf("Cannot remove neuron, layer size would be zero\n");
        return;
    }
    if(neuronIndex >= size)
    {
        printf("Cannot remove neuron, index out of bounds\n");
        return;
    }
    weight_t *newWeights = nullptr;
    in_out_t *newActivations = new in_out_t[size-1]; //No need to copy these
    weight_t *newBiases = new weight_t[size-1];
    ActivationFunctionType *newActiFuns = new ActivationFunctionType[size-1];
    count_t newIndex = 0;
    for(count_t i = 0; i < size; i++) //iterate over every old neuron
    {
        if(i == neuronIndex) continue;
        newBiases[newIndex] = biases[i];
        newActiFuns[newIndex] = actiFun[i];
        newIndex++;
    }
    if(prevLayerSize > 0)
    {
        newWeights = new weight_t[(size-1) * prevLayerSize];
        count_t newRow = 0;
        for(count_t i = 0; i < size; i++)
        {
            if(i == neuronIndex) continue;
            for(count_t conn = 0; conn < prevLayerSize; conn++)
            {
                newWeights[newRow * prevLayerSize + conn] = weights[i * prevLayerSize + conn];
            }
            newRow++;
        }
    }

    //fix connections of next layer
    if(next != nullptr)
    {
        weight_t *nextLayerNewWeights = new weight_t[next->size * (size-1)];
        for(count_t neuron = 0; neuron < next->size; neuron++)
        {
            count_t newWeightIndex = 0;
            for(count_t conn = 0; conn < this->size; conn++)
            {
                if(conn == neuronIndex) continue;
                nextLayerNewWeights[neuron * (size-1) + newWeightIndex] = next->weights[neuron * this->size + conn];
                newWeightIndex++;
            }
        }
        delete[] next->weights;
        next->weights = nextLayerNewWeights;
        next->prevLayerSize = static_cast<count_t>(size-1);
    }
    ////////////////////////////////

    if(weights == nullptr || biases == nullptr || actiFun == nullptr || activations == nullptr)
    {
        printf("Error\n");
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
void LayerV2::feedThrough(in_out_t *inputs)
{
    for(count_t i = 0; i < size; i++)
    {
        in_out_t weightedSum = 0;
        if(prevLayerSize > 0) 
        {
            weightedSum = biases[i];
            for(count_t iWeights = 0; iWeights < prevLayerSize; iWeights++)
            {
                weightedSum += inputs[iWeights] * weights[i * prevLayerSize + iWeights];
            }
        }
        else
        {
            weightedSum = inputs[i];
        }
        activations[i] = activationFunction(actiFun[i], weightedSum); //make ReLu
    }
}
#endif

void LayerV2::mutate(weight_t mutationRate)
{
    //to leave the chance that the layer does not mutate at all, we do modulo n+1
    uint8_t mutationSpecifier;
    mutationSpecifier = uint8_t(rand() % 4);
    switch (mutationSpecifier)
    {
        case 0: // add neuron
            if(next != nullptr && this->weights != nullptr)
            {
                addNeuron((ActivationFunctionType)(rand() % ActivationFunctionType::NONE));
                break;
            }
        case 1: // remove neuron
            if(next != nullptr && this->weights != nullptr)
            {
                removeNeuron(count_t(rand() % size));
                break;
            }
        case 2: //change bias
        {
            count_t neuronSpecifier = count_t(rand() % size);
            weight_t weightchange = ((weight_t)rand() / (weight_t)RAND_MAX - (weight_t)0.5) * mutationRate;
            biases[neuronSpecifier] += weightchange;
            break;
        }
        case 3: // change weight
        {
            count_t neuronSpecifier = count_t(rand() % size);
            count_t connectionSpecifier = count_t(rand() % prevLayerSize);
            weight_t weightchange = ((weight_t)rand() / (weight_t)RAND_MAX - (weight_t)0.5) * mutationRate;
            weights[neuronSpecifier * prevLayerSize + connectionSpecifier] += weightchange;
            break;
        }
        case 4: //change activation function
        {
            if(next != nullptr)
            {
                count_t neuronSpecifier = count_t(rand() % size);
                ActivationFunctionType newActivationfunction = (ActivationFunctionType)(rand() % ActivationFunctionType::NONE);
                actiFun[neuronSpecifier] = newActivationfunction;
                break;
            }
        }
        
        default:
            break;
    }
}

LayerV2* LayerV2::deepCopy() {
    LayerV2* copy = new LayerV2(size, prevLayerSize, actiFun[0]); // assume all neurons have same activation type
    for (count_t i = 0; i < size; i++) {
        if(biases != nullptr) copy->biases[i] = biases[i];
        if(actiFun != nullptr) copy->actiFun[i] = actiFun[i];
        for (count_t j = 0; j < prevLayerSize; j++) {
            if(weights != nullptr) copy->weights[i * prevLayerSize + j] = weights[i * prevLayerSize + j];
        }
    }
    copy->next = nullptr; // deep copy doesn't automatically link next layer
    return copy;
}

void LayerV2::exportToFile(std::ofstream &file, bool humanReadable)
{
    for (count_t i = 0; i < size; i++)
    {
        if(actiFun[i] < NONE)
        {
            humanReadable ? file << activationFunctionName[actiFun[i]] : file << actiFun[i];
        }
        else if(actiFun[i] >= NONE)
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
                memcpy(&iWeight, &(weights[i * prevLayerSize + c]), sizeof(weights[i * prevLayerSize + c]));
                file << ", " << iWeight;
            }
            else
            {
                file << ", " << weights[i * prevLayerSize + c];
            }
        }
        if(biases != nullptr)
        {
            if(!humanReadable)
            {
               uint64_t iWeight;
               memcpy(&iWeight, &(biases[i]), sizeof(biases[i]));
               file << ", " << iWeight; //last weight is always a bias
            }
            else
            {
                file << ", Bias: " << biases[i];
            } 
        }
        file << "\n";
    }
}