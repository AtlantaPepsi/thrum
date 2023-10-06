#include "header.hpp"


template <typename T>
class Copier
{
public:
    //int unroll = 8; perhaps as another macro?
    size_t grid = 4; //blocksize, shmem
    Copier() { };

    //void run(T cpyf, Param p)
    void run(Param<T> &p)
    {
        hipLaunchKernelGGL(RemoteCopy_Warp<1>, 4, BLOCKSIZE, 0, 0, p); 
    }
};
