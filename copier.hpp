#include "header.hpp"


template <typename T>
class Copier
{
public:
    //int unroll = 8; perhaps as another macro?
    size_t grid = 4; //blocksize, shmem
    Copier();
    ~Copier();

    //void run(T cpyf, Param p)
    float run(Param<T> &p)
    {
/*
        hipExtLaunchKernelGGL(RemoteCopy_Warp<1>, 
                              4, BLOCKSIZE, 
                              0, 0, 
                              startEvent, stopEvent,
                              p); //TODO: fix for nv
*/
        HIP_CHECK(hipEventRecord(startEvent, 0));
        hipLaunchKernelGGL(RemoteCopy_Warp<1>, 4, BLOCKSIZE, 0, 0, p);
        HIP_CHECK(hipEventRecord(stopEvent, 0));
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipEventElapsedTime(&gpuTimeMs, startEvent, stopEvent));
        return gpuTimeMs;
    }

private:
    int deviceID = -1;
    hipEvent_t startEvent, stopEvent;
    float gpuTimeMs;
};

template <typename T>
Copier<T>::Copier()
{
    HIP_CHECK(hipGetDevice(&deviceID));
    HIP_CHECK(hipEventCreate(&startEvent));
    HIP_CHECK(hipEventCreate(&stopEvent));
}

template <typename T>
Copier<T>::~Copier()
{
    HIP_CHECK(hipSetDevice(deviceID));
    HIP_CHECK(hipEventDestroy(startEvent));
    HIP_CHECK(hipEventDestroy(stopEvent));
}
