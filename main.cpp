#include "header.hpp"
#include "util.hpp"
#include "buffer.hpp"
#include "copier.hpp"
#include <iostream>
#include <chrono>

int main(int argc, char **argv) {

    int devices, size = argc > 1 ? atoi(argv[1]) : 1 << 25;
    int iter = argc > 2 ? atoi(argv[2]) : 5;
    HIP_CHECK(hipGetDeviceCount(&devices));

    SetUpPeer(devices);

    Buffer<int> b(size, devices);
    Param<int> *p_d;      //device parameter holder
    HIP_CHECK(hipMalloc(&p_d, sizeof(Param<int>)));  

    //for (int i = 0; i < iter; i++)

    Param<int> p = b.parameter();
    HIP_CHECK(hipMemcpy(p_d, &p, sizeof(Param<int>), hipMemcpyHostToDevice));

    for (int i = 0; i < 8; i++)
    {
        printf("A %p %p %zu %d\n", p_d->Src[i], p_d->Dst[i],p_d->size,p_d->Src[i][0]);
    }
    //hipLaunchKernelGGL(RemoteCopy_Warp<1>, 4, BLOCKSIZE, 0, 0, *p_d); 
    //HIP_CHECK(hipDeviceSynchronize());
    Copier<int> c;    
    c.run(*p_d);

/*
    for (int i = 0; i < devices; i++)
    {
        compareArrays(b._src[i], b._dst[i], size);
        //printf("device %d: %d , %d \n", i,b._src[i][0],b._dst[i][0]);
    }
*/
    return 0;
}
