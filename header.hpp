#pragma once

#if defined(__NVCC__)

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// Datatypes
#define hipError_t                                         cudaError_t
#define hipEvent_t                                         cudaEvent_t
#define hipStream_t                                        cudaStream_t

// Enumerations
#define hipSuccess                                         cudaSuccess
#define hipMemcpyHostToDevice                              cudaMemcpyHostToDevice
#define hipMemcpyDeviceToDevice                            cudaMemcpyDeviceToDevice

// Device
#define hipDeviceSynchronize                               cudaDeviceSynchronize
#define hipSetDevice                                       cudaSetDevice
#define hipGetDevice                                       cudaGetDevice

// Error
#define hipGetErrorString                                  cudaGetErrorString

// Stream
#define hipStreamCreate                                    cudaStreamCreate
#define hipStreamDestroy                                   cudaStreamDestroy
#define hipStreamSynchronize                               cudaStreamSynchronize

// Event
#define hipEventCreate                                     cudaEventCreate
#define hipEventRecord                                     cudaEventRecord
#define hipEventDestroy                                    cudaEventDestroy
#define hipEventElapsedTime                                cudaEventElapsedTime

// Mem
#define hipMalloc                                          cudaMalloc
#define hipFree                                            cudaFree
#define hipMemcpy                                          cudaMemcpy

// P2P DMA
#define hipDeviceCanAccessPeer                             cudaDeviceCanAccessPeer
#define hipDeviceEnablePeerAccess                          cudaDeviceEnablePeerAccess
#define hipGetDeviceCount                                  cudaGetDeviceCount

#else

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#endif

#define HIP_CHECK(condition) {                                            \
        hipError_t error = condition;                                     \
        if(error != hipSuccess){                                          \
            fprintf(stderr,"HIP error: %d line: %d at file %s\n", error,  \
			    __LINE__, __FILE__); \
            exit (error);                                                 \
        }                                                                 \
    }

#if defined(__NVCC__)
    #define WARP_SIZE       32
    #define BLOCKSIZE       1024
#else
    #define WARP_SIZE       64
    #define BLOCKSIZE       256
#endif

#define WARP_COUNT BLOCKSIZE/WARP_SIZE

//https://github.com/ROCmSoftwarePlatform/rccl/blob/6ecf771832ae887ce272889b9aaab008b5e0ddb6/tools/rccl-prim-test/rccl_prim_test.cpp#L237
//is_xgmi

#define MAX_SRCS 16
#define MAX_DSTS 16

template <typename T>
struct Param
{
    size_t      size;
    int const   devices;
    int const   numSrc;
    int const   numDst;
    T*          Src[MAX_SRCS];
    T*          Dst[MAX_DSTS];
};

template <int UNROLL, int STRIDE, typename T> 
//template <typename T>
inline __device__ void Copy(T* __restrict__ dst, T const* __restrict__ src)
{
    #pragma unroll
    for (int i = 0; i < UNROLL; i++)
    {
        *(dst + i * STRIDE) = *(src + i * STRIDE);
    }
}
