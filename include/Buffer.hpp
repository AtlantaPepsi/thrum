#include "header.hpp"

template <typename T>
class Buffer
{
public:
    size_t r_size;  // = 1 << 25;  //default 33 MB //recv buffer size
    size_t s_size; // send buffer size
    int _devices;
    T *_src[MAX_SRCS], *_dst[MAX_DSTS], *hostBuffer;        

    void check(int device, size_t size);
    Param<T> parameter();     //populate parameter with buffer/sys info
    void reset();
    
    Buffer(size_t size, int devices);
    //Buffer();
    ~Buffer();
};

template <typename T>
void Buffer<T>::check(int device, size_t size)
{
    T* dst = (T*) malloc(sizeof(T) * size);
    T* src = (T*) malloc(sizeof(T) * size);
    for (int i = 0; i < _devices; i++)
    {
        //compareArrays(b._src[0] + i*size, b._dst[i], size);
        hipMemcpy(src, _src[device] + i*size, size * sizeof(T), hipMemcpyDefault);
        hipMemcpy(dst, _dst[i],               size * sizeof(T), hipMemcpyDefault);

        for (int j = 0; j < size; j++)
        {
            if (src[j] != dst[j])
            {
                printf("assertion failed at index %d: %d vs %d\n",
                        j, src[j], dst[j]);
                return;
            }
        }
    }
    free(dst);
    free(src);
}

template <typename T>
Param<T> Buffer<T>::parameter()
{
    Param<T> p{.size = r_size, .devices = _devices, .numSrc = _devices, .numDst = _devices};
    int device = -1;
    HIP_CHECK(hipGetDevice(&device));
    for (int i = 0; i < _devices; i++)
    {
        p.Src[i] = _src[device] + i * r_size;
        p.Dst[i] = _dst[i];
    }
    return p;
}

template <typename T>
void Buffer<T>::reset()
{
    int deviceID; 
    HIP_CHECK(hipGetDevice(&deviceID));
    int count = 0; //TODO: change to rand/others later

    for (int i = 0; i < _devices; i++)
    {
        HIP_CHECK(hipSetDevice(i));

        for (auto j = 0; j < s_size; j++) {
            hostBuffer[j] = count++;
        }
        HIP_CHECK(hipMemcpy(_src[i], hostBuffer, sizeof(T) * s_size, hipMemcpyHostToDevice));
    }
    HIP_CHECK(hipSetDevice(deviceID));
}

template <typename T>
Buffer<T>::Buffer(size_t size, int devices) : r_size(size), _devices(devices)
{
    int deviceID; 
    HIP_CHECK(hipGetDevice(&deviceID));

    s_size = devices * r_size;
    hostBuffer = (T*) malloc(sizeof(T) * s_size);

    for (int i = 0; i < _devices; i++)
    {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipMalloc(_src + i, sizeof(T) * s_size));
        HIP_CHECK(hipMalloc(_dst + i, sizeof(T) * r_size));   //!!
    }

    HIP_CHECK(hipSetDevice(deviceID));
    reset();
}

template <typename T>
Buffer<T>::~Buffer()
{
    HIP_CHECK(hipDeviceSynchronize()); //very important guard at end of main

    free(hostBuffer);
    for (int i = 0; i < _devices; i++)
    {
        HIP_CHECK(hipFree(_src[i]));
        HIP_CHECK(hipFree(_dst[i])); 
    }
}
