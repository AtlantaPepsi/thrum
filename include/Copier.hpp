#pragma once

#include "header.hpp"

template <typename T>
class Copier
{
public:
    Copier(Param<T> p, size_t grid);
    Copier(Param<T> p) : Copier(p, 4) {}
    ~Copier();

    void setGridDim(size_t grid) { _grid = grid; }

    double Record(size_t iter = 5)
    {
        float gpuTimeMs = 0;
        int device;
        HIP_CHECK(hipGetDevice(&device));
        HIP_CHECK(hipSetDevice(deviceID));

        HIP_CHECK(hipMemcpy(p_d, &_p, sizeof(Param<T>), hipMemcpyHostToDevice));
        
        HIP_CHECK(hipEventRecord(startEvent, 0));
        for (int i = 0; i < iter; i++)
        {    
            Copy();
        }
        HIP_CHECK(hipEventRecord(stopEvent, 0));
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipEventElapsedTime(&gpuTimeMs, startEvent, stopEvent));

        HIP_CHECK(hipSetDevice(device));
        
        size_t bytes = _p.size * sizeof(T) * _p.numDst;
        double exeDurationMs = gpuTimeMs / (1.0 * iter);
        double bandwidth = (bytes / 1.0E9) / exeDurationMs * 1000;

        return bandwidth;
    }

    void virtual Copy(){ 
        for (int i = 0; i < _p.devices; i++)
        {
            HIP_CHECK(hipMemcpy(_p.Dst[i], _p.Src[i], _p.size * sizeof(T), hipMemcpyDeviceToDevice));
        }
        printf("hipMemcpy hipMemcpy\n");
    }

protected:
    size_t _grid; //blocksize, shmem
    int deviceID = -1;
    hipEvent_t startEvent, stopEvent;
    Param<T> _p;
    Param<T> *p_d;
};

template <typename T>
Copier<T>::Copier(Param<T> p, size_t grid) : _p(p),  _grid(grid)
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
