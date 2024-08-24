#pragma once
#include <stdint.h>
#include <stdio.h>

#ifdef useGPU
__host__ void *operator new(size_t size)
{
    void *temp;
    cudaError_t ret = cudaMallocManaged(&temp, size);
    if (ret != cudaError::cudaSuccess)
    {
        printf("Network: Malloc failed with code %d\n", ret);
        // exit(1);
    }
    return temp;
}
__host__ void operator delete(void *ptr) noexcept
{
    cudaError_t ret = cudaFree(ptr);
    if (ret != cudaError::cudaSuccess)
    {
        printf("Network: delete failed with code %d\n", ret);
        // exit(1);
    }
}
__host__ void operator delete[](void *ptr) noexcept
{
    cudaError_t ret = cudaFree(ptr);
    if (ret != cudaError::cudaSuccess)
    {
        printf("Network: delete[] failed with code %d\n", ret);
        // exit(1);
    }
}
#endif

#ifdef useGPU
void* operator new(size_t size)
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
void operator delete(void* ptr)
{
    cudaError_t ret = cudaFree(ptr);
    if(ret != cudaError::cudaSuccess)
    {
        printf("Layer: delete %p failed with code %d\n", ptr, ret);
        exit(1);
    }
}
void operator delete[](void* ptr)
{
    cudaError_t ret = cudaFree(ptr);
    if(ret != cudaError::cudaSuccess)
    {
        printf("Layer: delete[] failed with code %d\n", ret);
        exit(1);
    }
}
#endif
typedef float weight_t;
typedef float in_out_t;
typedef uint16_t count_t;