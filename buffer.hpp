#include "header.hpp"

template <typename T>
class Buffer
{
public:
    size_t r_size;  // = 1 << 25;  //default 33 MB //recv buffer size
    size_t s_size; // send buffer size
    int _devices;
    T *_src[MAX_SRCS], *_dst[MAX_DSTS], *hostBuffer;        

    //Param p;             //dont keep one due to const field, return 1 in loop/scope
    
    Param<T> parameter();     //populate parameter with buffer/sys info
    void reset();
    
    Buffer(size_t size, int devices);
    //Buffer();
    ~Buffer();
};

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
    int count = 0; //TODO: change to rand/others later
    for (int i = 0; i < _devices; i++)
    {
        HIP_CHECK(hipSetDevice(i));

        for (auto j = 0; j < s_size; j++) {
            hostBuffer[j] = count++;
        }
        HIP_CHECK(hipMemcpy(_src[i], hostBuffer, sizeof(T) * s_size, hipMemcpyHostToDevice));
    }
}

template <typename T>
Buffer<T>::Buffer(size_t size, int devices) : r_size(size), _devices(devices)
{
    s_size = devices * r_size;
    hostBuffer = (T*) malloc(sizeof(T) * s_size);

    for (int i = 0; i < _devices; i++)
    {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipMalloc(_src + i, sizeof(T) * s_size));
        HIP_CHECK(hipMalloc(_dst + i, sizeof(T) * r_size));   //!!
    }

    reset();
printf("buffer allocated\n");
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
printf("buffer deallocated\n");
}
