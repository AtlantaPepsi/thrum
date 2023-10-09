#include "header.hpp"
#include "util.hpp"
#include "buffer.hpp"
#include "copier.hpp"
#include <iostream>
#include <chrono>

using _type = int;

int main(int argc, char **argv) {

    int devices, size = argc > 1 ? atoi(argv[1]) : 1 << 25;
    int iter = argc > 2 ? atoi(argv[2]) : 5;


    HIP_CHECK(hipGetDeviceCount(&devices));
    SetUpPeer(devices);


    Buffer<_type> b(size, devices);
    Param<_type> *p_d;      //device parameter holder
    HIP_CHECK(hipMalloc(&p_d, sizeof(Param<_type>)));  


    hipSetDevice(0); // ... ?
    Param<_type> p = b.parameter();
    HIP_CHECK(hipMemcpy(p_d, &p, sizeof(Param<_type>), hipMemcpyHostToDevice));

    float t1 = 0, t2 = 0;
    Copier<_type> c;
    for (int i = 0; i < iter; i++)
    {
        t1 += c.Run(*p_d,RemoteCopy_Warp<1,_type>);
        //b.reset();
        t2 += c.Run(*p_d,RemoteCopy_Block<1,_type>);
    }
    printf("warp copy time (ms): %f\n", t1 / iter);
    printf("block copy time (ms): %f\n", t2 / iter);
/*
    for (int i = 0; i < devices; i++)
    {
        compareArrays(b._src[0] + i*size, b._dst[i], size);
    }
*/

    HIP_CHECK(hipDeviceSynchronize());
    return 0;
}
