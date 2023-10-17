#define FACTOR 1 //unroll factor

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
    int devices,    
        size = argc > 1 ? atoi(argv[1]) : 1 << 25,
        iter = argc > 2 ? atoi(argv[2]) : 5,
        grid = argc > 3 ? atoi(argv[3]) : 4;


    HIP_CHECK(hipGetDeviceCount(&devices));
    SetUpPeer(devices);

    Buffer<_type> b(size, devices);
    Param<_type> p = b.parameter(); // how abt managed mem ptr, from Buffer?

    WarpCopier<_type, FACTOR> w_c(p, 4);
    BlockCopier<_type, FACTOR> b_c(p, 4);

    for (int i = 4; i < 32; i *= 2)
    {
        printf("CU/SM : %d\n", i);
        w_c.setGridDim(i);
        b_c.setGridDim(i);
        double t1 = 0, t2 = 0;
        t1 += w_c.Record();
        printf("warp copy bandwidth (GB/s): %f\n", t1);

        t2 += b_c.Record();
        printf("block copy bandwidth (GB/s): %f\n", t2);
    }
    b.check(0, size);
    HIP_CHECK(hipDeviceSynchronize());
    return 0;
}
