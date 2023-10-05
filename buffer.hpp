#include "header.hpp"

template <typename T>
class Buffer
{
public:
    size_t _size;  // = 1 << 25;  //default 33 MB
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
    Param<T> p{.size = _size, .devices = _devices, .numSrc = _devices, .numDst = _devices};
    for (int i = 0; i < _devices; i++)
    {
        p.Src[i] = _src[i];
        p.Dst[i] = _dst[i];
    }
    return p;
}

template <typename T>
void Buffer<T>::reset()
{ 
    int count = 0; //change to rand/others later
    for (int i = 0; i < _devices; i++)
    {
        HIP_CHECK(hipSetDevice(i));

        for (auto j = 0; j < _size; j++) {
            hostBuffer[j] = count++;
        }
        HIP_CHECK(hipMemcpy(_src[i], hostBuffer, sizeof(T) * _size, hipMemcpyHostToDevice));
    }
printf("count: %d\n",count);
}

template <typename T>
Buffer<T>::Buffer(size_t size, int devices) : _size(size), _devices(devices)
{
    hostBuffer = (T*) malloc(sizeof(T) * _size);

    for (int i = 0; i < _devices; i++)
    {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipMalloc(_src + i, sizeof(T) * _size));
        HIP_CHECK(hipMalloc(_dst + i, sizeof(T) * _size));   //!!
    }

    reset();
printf("it worked!\n");
}

template <typename T>
Buffer<T>::~Buffer()
{
    free(hostBuffer);
    for (int i = 0; i < _devices; i++)
    {
        HIP_CHECK(hipFree(_src[i]));
        HIP_CHECK(hipFree(_dst[i])); 
    }
printf("it worked again!\n");
}
