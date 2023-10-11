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

template <typename T, int UNROLL>
class BlockCopier : public Copier<T>
{
public:
    void Copy(Param<T> const& p) override
    {
        RemoteCopy_Block<T,UNROLL><<<4,BLOCKSIZE>>>(p);
    }
};
