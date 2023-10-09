#include "header.hpp"
#include <functional>

template <typename T> //how do i pass any function as template arg? auto F? transform?
class Copier
{
public:
    //int unroll = 8; perhaps as another macro?
    size_t grid = 4; //blocksize, shmem
    Copier();
    ~Copier();

    template <typename F>
    float Run(Param<T> &p, F&& kernel)
    {
        HIP_CHECK(hipEventRecord(startEvent, 0));
        hipLaunchKernelGGL(kernel, 4, BLOCKSIZE, 0, 0, p);
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
