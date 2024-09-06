#ifndef neuronTypes_h
#define neuronTypes_h
#include <stdint.h>
#include <stdio.h>

#ifdef useGPU
void* operator new(size_t size);
void operator delete(void* ptr) noexcept;
void operator delete[](void* ptr) noexcept;
#endif
typedef uint8_t weight_t;
typedef uint8_t in_out_t;
typedef uint16_t count_t;

#endif