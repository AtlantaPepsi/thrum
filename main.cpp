#include "header.hpp"
#include "util.hpp"
#include "Buffer.hpp"
#include "WarpCopier.hpp"
#include "BlockCopier.hpp"
#include <iostream>
#include <chrono>

using _type = int;

int main(int argc, char **argv) {

    HIP_CHECK(hipSetDevice(0));
    int devices, size = argc > 1 ? atoi(argv[1]) : 1 << 25;
    int iter = argc > 2 ? atoi(argv[2]) : 5;


    HIP_CHECK(hipGetDeviceCount(&devices));
    SetUpPeer(devices);


    Buffer<_type> b(size, devices);
    Param<_type> *p_d;      //device parameter holder
    HIP_CHECK(hipMalloc(&p_d, sizeof(Param<_type>)));  


    Param<_type> p = b.parameter();
    HIP_CHECK(hipMemcpy(p_d, &p, sizeof(Param<_type>), hipMemcpyHostToDevice));

    double t1 = 0, t2 = 0;
    WarpCopier<_type, 1> w_c;
    BlockCopier<_type, 1> b_c;

    t1 += w_c.Record(*p_d);
    printf("warp copy bandwidth (GB/s): %f\n", t1);

    b.reset();

    t2 += b_c.Record(*p_d);
    printf("block copy bandwidth (GB/s): %f\n", t2);
/*
    for (int i = 0; i < devices; i++)
    {
        compareArrays(b._src[0] + i*size, b._dst[i], size);
    }
*/

    HIP_CHECK(hipDeviceSynchronize());
    return 0;
}
