#include "header.hpp"
#include "util.hpp"
#include "buffer.hpp"
#include "copier.hpp"
#include <iostream>
#include <chrono>

int main(int argc, char **argv) {

    int devices, size = argc > 1 ? atoi(argv[1]) : 1 << 25;
    int iter = argc > 2 ? atoi(argv[2]) : 5;
    hipEvent_t startEvent, stopEvent;
    float gpuTimeMs;

    HIP_CHECK(hipEventCreate(&startEvent));
    HIP_CHECK(hipEventCreate(&stopEvent));

    HIP_CHECK(hipGetDeviceCount(&devices));
    SetUpPeer(devices);


    Buffer<int> b(size, devices);
    Param<int> *p_d;      //device parameter holder
    HIP_CHECK(hipMalloc(&p_d, sizeof(Param<int>)));  

    //for (int i = 0; i < iter; i++)

    hipSetDevice(0); //!!: for event, and for p
    Param<int> p = b.parameter();
    HIP_CHECK(hipMemcpy(p_d, &p, sizeof(Param<int>), hipMemcpyHostToDevice));

 
    HIP_CHECK(hipEventRecord(startEvent, 0));
    hipLaunchKernelGGL(RemoteCopy_Warp<1>, 4, BLOCKSIZE, 0, 0, *p_d); 
    HIP_CHECK(hipEventRecord(stopEvent, 0));

    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventElapsedTime(&gpuTimeMs, startEvent, stopEvent));
//    Copier<int> c;    
//    c.run(*p_d);
    printf("ms: %f\n", gpuTimeMs);
/*
    for (int i = 0; i < devices; i++)
    {
        compareArrays(b._src[0] + i*size, b._dst[i], size);
        //printf("device %d: %d , %d \n", i,b._src[i][0],b._dst[i][0]);
    }
*/
    HIP_CHECK(hipEventDestroy(startEvent));
    HIP_CHECK(hipEventDestroy(stopEvent));

    return 0;
}
