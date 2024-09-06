#include "neuronTypes.hpp"

#ifdef useGPU
void* operator new(size_t size)
{
    void *temp;
    cudaError_t ret = cudaMallocManaged(&temp, size);
    if(ret != cudaError::cudaSuccess)
    {
        printf("Layer: Malloc failed with code %d\n", ret);
        // exit(1);
    }
    return temp;
}
void operator delete(void* ptr) noexcept
{
    cudaError_t ret = cudaFree(ptr);
    if(ret != cudaError::cudaSuccess)
    {
        printf("Layer: delete %p failed with code %d\n", ptr, ret);
        // exit(1);
    }
}
void operator delete[](void* ptr) noexcept
{
    cudaError_t ret = cudaFree(ptr);
    if(ret != cudaError::cudaSuccess)
    {
        printf("Layer: delete[] failed with code %d\n", ret);
        // exit(1);
    }
}
#endif