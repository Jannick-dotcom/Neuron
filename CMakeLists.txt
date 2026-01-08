cmake_minimum_required(VERSION 3.13)  # CMake version check
file(GLOB NEURON_SRC
     "inc/*.cpp"
)

add_library(Neuron STATIC ${NEURON_SRC})

target_include_directories(Neuron PUBLIC inc)
target_compile_options(Neuron PUBLIC -g)  # Enable debugging symbols

##############################################################################
# CUDA part for Neuron library
enable_language(CUDA)
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()
file(GLOB NEURON_CUDA_SRC
     "inc/*.cu"
)
add_library(Neuron_Cuda STATIC ${NEURON_CUDA_SRC} ${NEURON_SRC})
set_target_properties(Neuron_Cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(Neuron_Cuda PROPERTIES CUDA_ARCHITECTURES "35;50;72")
target_include_directories(Neuron_Cuda PUBLIC inc)
target_compile_definitions(Neuron_Cuda PUBLIC -DuseGPU)
target_compile_options(Neuron_Cuda PUBLIC -g)  # Enable debugging symbols

set_target_properties(Neuron_Cuda PROPERTIES LINKER_LANGUAGE CUDA)