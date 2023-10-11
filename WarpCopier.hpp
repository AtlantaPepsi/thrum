#include "Copier.hpp"

//warp from each block cooperate, one source to all dst
//p.size is total memcpy btw src-dst pair
template <typename T, int UNROLL>
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
            Copy<UNROLL, WARP_SIZE, T>(dst + loopOffset, src + loopOffset);
        }
    }
}

template <typename T, int UNROLL>
class WarpCopier : public Copier<T>
{
public:
    void Copy(Param<T> const& p) override
    {
        RemoteCopy_Warp<T,UNROLL><<<4,BLOCKSIZE>>>(p);
    }
};
