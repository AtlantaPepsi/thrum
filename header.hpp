#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#define HIP_CHECK(condition) {                                            \
        hipError_t error = condition;                                     \
        if(error != hipSuccess){                                          \
            fprintf(stderr,"HIP error: %d line: %d\n", error,  __LINE__); \
            exit (error);                                                 \
        }                                                                 \
    }


#ifdef NVCC
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
