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

//warp from each block cooperate, one source to all dst
//p.size is total memcpy btw src-dst pair
template <int UNROLL, typename T>
__global__ void __launch_bounds__(BLOCKSIZE)
RemoteCopy_Warp(Param<T> const& p)
{
    int const warpId     = threadIdx.x / WARP_SIZE; // Warp index 
    int const threadId   = threadIdx.x % WARP_SIZE; // Thread index within warp
    int const blockId    = blockIdx.x;

    size_t const size    = p.size / gridDim.x;     // warps from diff block complete 1 src
    size_t const loopInc = WARP_SIZE * UNROLL;

    for (int i = 0; i < p.numDst; i += WARP_COUNT)  // each warp takes multi dst
    {
        auto src = p.Src[warpId + i] + blockId * size;
        auto dst = p.Dst[warpId + i] + blockId * size;

        for (size_t loopOffset = threadId; loopOffset < size; loopOffset += loopInc)
        {
            if ((src + loopOffset) >  (p.Src[warpId + i] + p.size))
            {
                printf("Don't do that!%zu, %zu, %zu \n",p.size,loopOffset,blockId*size);
                assert(0);
            }
            Copy<UNROLL, WARP_SIZE, T>(dst + loopOffset, src + loopOffset);
        }
    }
}

//scatter style, 1 block to 1 dst, all block on same device/buffer
template <int UNROLL, typename T>
__global__ void __launch_bounds__(BLOCKSIZE) // is this needed?
RemoteCopy_Block(Param<T> const& p)
{
    int const blockId    = blockIdx.x;
    int const threadId   = threadIdx.x; // Thread index within block
   
    size_t const loopInc = BLOCKSIZE * UNROLL;
  
    for (int i = 0; i < p.numDst; i += gridDim.x)
    {
        auto src = p.Src[blockId + i];
        auto dst = p.Dst[blockId + i];

        for (size_t offset = threadId; offset < p.size; offset += loopInc)
        {
            Copy<UNROLL, BLOCKSIZE, T>(dst + offset, src + offset); 
        }
    }
}
/*
template <int UNROLL>
__global__ void __launch_bounds__(BLOCKSIZE)
RemoteCopy_Warp_1Block(Param const& p)
{
    int const warpId     = threadIdx.x / WARP_SIZE; // Warp index 
    int const threadId   = threadIdx.x % WARP_SIZE; // Thread index within warp

    size_t const loopInc = WARP_SIZE * UNROLL;

    for (int i = 0; i < p.devices; i += WARP_COUNT)
    {
        auto src = p.Src[warpId + i];
        auto dst = p.Dst[warpId + i];

        for (size_t loopOffset = threadId; loopOffset < p.size; loopOffset += loopInc)
        {
            Copy<UNROLL,WARP_SIZE>(dst + loopOffset, src + loopOffset);
        }
    }
}

//1 process 1 warp to 1 target
template <int UNROLL>
__global__ void __launch_bounds__(BLOCKSIZE)
RemoteCopy_PtoP(size_t const size,
                int* __restrict__ Dst,
                int const* __restrict__ Src)
{
    int const warpId   = threadIdx.x / WARP_SIZE;
    int const threadId = threadIdx.x % WARP_SIZE;
   
    size_t const loopInc = WARP_SIZE * UNROLL;

    for (size_t offset = threadId; offset < size; offset += loopInc)
    {
        Copy<UNROLL,WARP_SIZE>(Dst + offset, Src + offset);
    }
}*/
