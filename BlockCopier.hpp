#include "Copier.hpp"

//scatter style, 1 block to 1 dst, all block on same device/buffer
template <typename T, int UNROLL>
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

//more block than dst
template <typename T, int UNROLL>
__global__ void __launch_bounds__(BLOCKSIZE) // is this needed?
RemoteCopy_Block2(Param<T> const& p)
{
    int const blockId    = blockIdx.x;
    int const threadId   = threadIdx.x; // Thread index within block

    size_t const tileSze = gridDim.x / p.numDst;    // a tile is a group of warp on
                                                    // same SM, cooperating on same dst
    size_t const tileId  = blockId / tileSze;      // the target dst, numDst tile in total

    size_t const tilePsn = blockId % tileSze;      // this warp's position within this tile
    size_t const size    = p.size / tileSze;        // total size/(#CU within tile)

    size_t const loopInc = BLOCKSIZE * UNROLL;

    auto src = p.Src[tileId] + tilePsn * size;
    auto dst = p.Dst[tileId] + tilePsn * size;

    for (size_t offset = threadId; offset < size; offset += loopInc)
    {
        Copy<UNROLL, BLOCKSIZE, T>(dst + offset, src + offset);
    }
}


template <typename T, int UNROLL>
class BlockCopier : public Copier<T>
{
public:
    void Copy() override
    {
        if (this->_p.numDst > this->_grid)
            RemoteCopy_Block<T,UNROLL><<<this->_grid,BLOCKSIZE>>>(*(this->p_d));
        else
            printf("????\n");
    }
};
