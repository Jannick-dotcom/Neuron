class LayerV2
{
public:
    LayerV2 *next;
    double **weights; //incoming weights
    double *biases; //incoming biases
    double *activations; //current activations
    int size = 0;
    int prevLayerSize = 0;
    LayerV2(int size, int prevLayerSize)
    {
        if(prevLayerSize > 0)
        {
            weights = new double*[size];
            for(int i = 0; i < size; i++)
            {
                weights[i] = new double[prevLayerSize];
                for(int j = 0; j < prevLayerSize; j++)
                {
                    weights[i][j] = rand() / double(RAND_MAX) - 0.5;
                }
            }
            biases = new double[size];
            for(int i = 0; i < size; i++)
            {
                biases[i] = rand() / double(RAND_MAX) - 0.5;
            }
        }
        activations = new double[size];
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
};

class NetworkV2
{
public:
    LayerV2 *firstLayer;
    LayerV2 *lastLayer;
    int ctLayers;
    double *transferFuntion;
    void addLayer(int size)
    {
        if(firstLayer == nullptr)
        {
            firstLayer = new LayerV2(size, 0);
            lastLayer = firstLayer;
        }
        else
        {
            lastLayer->next = new LayerV2(size, lastLayer->size);
            lastLayer = lastLayer->next;
        }
    }
    void feedThrough(double *inputs)
    {
        LayerV2 *currentLayer = firstLayer;
        double *outputsOfLastLayer = inputs;
        int sizeOfLastLayer = 0;
        while(currentLayer != nullptr)
        {
            for(int i = 0; i < currentLayer->size; i++)
            {
                double neuronInput = 0;
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
                currentLayer->activations[i] = (neuronInput > 0 ? neuronInput : 0); //make ReLu + bias
            }
            sizeOfLastLayer = currentLayer->size;
            outputsOfLastLayer = currentLayer->activations;
            currentLayer = currentLayer->next;
        }
    }
};