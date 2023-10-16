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

    double t1 = 0, t2 = 0;
    WarpCopier<_type, 1> w_c;
    BlockCopier<_type, 1> b_c;

    t1 += w_c.Record(p);
    printf("warp copy bandwidth (GB/s): %f\n", t1);

    t2 += b_c.Record(p);
    printf("block copy bandwidth (GB/s): %f\n", t2);

    b.check(0, size);
    HIP_CHECK(hipDeviceSynchronize());
    return 0;
}
