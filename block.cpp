#include "header.hpp"
#include "util.hpp"
#include <chrono>
#include <iostream>

int main() {

    int devices, numElems = 1 << 25; //33.55MB
    int *hostArray = (int*)malloc(sizeof(int) * numElems);
    HIP_CHECK(hipGetDeviceCount(&devices));
    
    int **localMem = (int**)malloc(devices * sizeof(int*));

    int count = 0;
    for (int i = 0; i < devices; i++)
    {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipMalloc(localMem + i, sizeof(int)*numElems));

        for (auto j = 0; j < numElems; j++) {
            hostArray[j] = count++;
        }
        HIP_CHECK(hipMemcpy(localMem[i], hostArray, sizeof(int) * numElems, hipMemcpyHostToDevice));
    }

    SetUpPeer(devices);


    //main test: work to each device split across 4 block
    //BlockCopy test (7 -> 0-7), 4 block
    //Block 1: -> device 0,4; Block 2: -> device 1,5 ...

    int *src7; 
    HIP_CHECK(hipSetDevice(7)); 
    HIP_CHECK(hipMalloc(&src7, sizeof(int)*numElems));
    HIP_CHECK(hipMemcpy(src7, hostArray, sizeof(int) * numElems, hipMemcpyHostToDevice));

    Param *p_d;
    HIP_CHECK(hipMalloc(&p_d, sizeof(Param)));  

    Param p = { .size = (size_t)numElems / 8, .devices = devices, .numSrc = devices, .numDst = devices };
    for (int i = 0; i < devices; i++)
    {
        p.Src[i] = src7 + i * p.size;
        p.Dst[i] = localMem[i];
    }

    //assert unroll x warpsize < size
    //assert numSrc % warp_count == 0
    //assert size % blockidx == 0
    //assert numSrc = numDst ..?
    HIP_CHECK(hipMemcpy(p_d, &p, sizeof(Param), hipMemcpyHostToDevice));
    
    const auto start = std::chrono::high_resolution_clock::now();

    hipLaunchKernelGGL(RemoteCopy_Block<1>, 4, BLOCKSIZE, 0, 0, *p_d);
    HIP_CHECK(hipDeviceSynchronize());

    const auto end = std::chrono::high_resolution_clock::now(); 
    const std::chrono::duration<double> diff = end - start;
    auto ms = std::chrono::duration_cast<std::chrono::microseconds>(diff); 
    std::cout << ms.count() << " microsec" << std::endl;   
 

    hipFree(src7);
    for (int i = 0; i < devices; i++)
    {
       compareArrays(hostArray + i * p.size, localMem[i], numElems / 8);
    }
    hipFree(p_d);

    for (int i = 0; i < devices; i++)
        hipFree(localMem[i]);
    free(localMem);
    return 0;
}
