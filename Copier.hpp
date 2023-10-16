#pragma once

#include "header.hpp"

template <typename T>
class Copier
{
public:
    Copier(size_t grid);
    Copier() : Copier(4) {}
    ~Copier();

    double Record(Param<T> const& p, size_t iter = 5)
    {
        float gpuTimeMs = 0;
        int device;
        HIP_CHECK(hipGetDevice(&device));
        HIP_CHECK(hipSetDevice(deviceID));

        HIP_CHECK(hipMemcpy(p_d, &p, sizeof(Param<T>), hipMemcpyHostToDevice));
        
        HIP_CHECK(hipEventRecord(startEvent, 0));
        for (int i = 0; i < iter; i++)
        {    
            Copy(*p_d);
        }
        HIP_CHECK(hipEventRecord(stopEvent, 0));
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipEventElapsedTime(&gpuTimeMs, startEvent, stopEvent));

        HIP_CHECK(hipSetDevice(device));
        
        size_t bytes = p.size * sizeof(T) * p.numDst;
        double exeDurationMs = gpuTimeMs / (1.0 * iter);
        double bandwidth = (bytes / 1.0E9) / exeDurationMs * 1000;

        return bandwidth;
    }

    void virtual Copy(Param<T> const& p){ 
        for (int i = 0; i < p.devices; i++)
        {
            HIP_CHECK(hipMemcpy(p.Dst[i], p.Src[i], p.size * sizeof(T), hipMemcpyDeviceToDevice));
        }
        printf("hipMemcpy hipMemcpy\n");
    }

protected:
    size_t _grid; //blocksize, shmem
    int deviceID = -1;
    hipEvent_t startEvent, stopEvent;
    Param<T> *p_d;
};

template <typename T>
Copier<T>::Copier(size_t grid) : _grid(grid)
{
    HIP_CHECK(hipGetDevice(&deviceID));
    HIP_CHECK(hipEventCreate(&startEvent));
    HIP_CHECK(hipEventCreate(&stopEvent));
    HIP_CHECK(hipMalloc(&p_d, sizeof(Param<T>)));
}

template <typename T>
Copier<T>::~Copier()
{
    HIP_CHECK(hipSetDevice(deviceID));
    HIP_CHECK(hipEventDestroy(startEvent));
    HIP_CHECK(hipEventDestroy(stopEvent));
    HIP_CHECK(hipFree(p_d));
}
