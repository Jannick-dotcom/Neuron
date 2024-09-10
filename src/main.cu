#include <sys/time.h>
#include "networkV2.hpp"
#define ctPasses 100000

int main()
{
    NetworkV2 *net = new NetworkV2();
    net->addLayer(1, LINEAR);
    net->addLayer(10000, SIGMOID);
    net->addLayer(10000, SIGMOID);
    net->addLayer(10000, SIGMOID);
    net->addLayer(10000, SIGMOID);
    net->addLayer(10000, SIGMOID);
    net->addLayer(10000, SIGMOID);
    net->addLayer(1, LINEAR);
    in_out_t *inputs = new in_out_t[net->firstLayer->size];
    in_out_t lastoutput = 0;
    struct timeval t1, t2;
    printf("Allocation done, start feeding through\n");
    gettimeofday(&t1, 0);
    for(uint32_t i = 0; i < ctPasses; i++)
    {
        net->feedThrough(inputs);
        if(net->lastLayer->activations[0] != lastoutput)
        {
            printf("Last output: %d, this: %d\n", lastoutput, net->lastLayer->activations[0]);
            lastoutput = net->lastLayer->activations[0];
        }
    }
    gettimeofday(&t2, 0);
    double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    printf("Time to generate:  %3.5f ms \n", time/ctPasses);//0.16361ms
    delete net;
    delete inputs;
}