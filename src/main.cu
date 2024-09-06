#include <sys/time.h>
#include "networkV2.hpp"
int main()
{
    NetworkV2 *net = new NetworkV2();
    net->addLayer(1, LINEAR);
    net->addLayer(10000, SIGMOID);
    net->addLayer(10000, SIGMOID);
    net->addLayer(1, LINEAR);
    in_out_t *inputs = new in_out_t[net->firstLayer->size];
    struct timeval t1, t2;
    uint32_t i = 0;
    gettimeofday(&t1, 0);
    for(; i < 1000; i++)
    {
        net->feedThrough(inputs);
    }
    gettimeofday(&t2, 0);
    double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
    printf("Time to generate:  %3.1f ms \n", time);//335ms
    delete net;
}