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

// more warp, more or less CU (nv only)
// adjacent warps on same dst
template <typename T, int UNROLL>
__global__ void __launch_bounds__(BLOCKSIZE)
RemoteCopy_Warp2(Param<T> const& p)
{
    int const warpId     = threadIdx.x / WARP_SIZE; // Warp index
    int const threadId   = threadIdx.x % WARP_SIZE; // Thread index within warp
    int const blockId    = blockIdx.x;

    size_t const tileSze = WARP_COUNT / p.numDst;   // a tile is a group of warp on
                                                    // same SM, cooperating on same dst
    size_t const tileId  = warpId / tileSze;        // the target dst, numDst tile in total

    size_t const tilePsn = warpId % tileSze;        // this warp's position within this tile
    size_t const size    = p.size / gridDim.x /
                           tileSze;                 // total size/#SM/(#warps within tile)

    size_t const loopInc = WARP_SIZE * UNROLL;


    auto src = p.Src[tileId] + ((blockId*tileSze) + tilePsn) * size;
    auto dst = p.Dst[tileId] + ((blockId*tileSze) + tilePsn) * size;    //maybe optimize this idx?

    for (size_t loopOffset = threadId; loopOffset < size; loopOffset += loopInc)
    {
        Copy<UNROLL, WARP_SIZE, T>(dst + loopOffset, src + loopOffset);
    }
}


template <typename T, int UNROLL>
class WarpCopier : public Copier<T>
{
public:
    WarpCopier(Param<T> p, size_t grid) : Copier<T>(p, grid) {};
    void Copy() override
    {
        RemoteCopy_Warp2<T,UNROLL><<<4,BLOCKSIZE>>>(this->_p);
    }
};
