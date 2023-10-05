#include "../header.hpp"
#include "../util.hpp"

int main() {

    int devices, numElems = 4096;
    int hostArray[numElems];
    HIP_CHECK(hipGetDeviceCount(&devices));
    
    int **localMem = (int**)malloc(devices * sizeof(int*));

    for (int i = 0; i < devices; i++)
    {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipMalloc(localMem + i, sizeof(int)*numElems));

        for (auto j = 0; j < numElems; j++) {
            hostArray[j] = j + i * 10000;
        }
        HIP_CHECK(hipMemcpy(localMem[i], hostArray, sizeof(int) * numElems, hipMemcpyHostToDevice));
    }

    SetUpPeer(devices);

    //WarpCopy test (7 -> 0-6)
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
    hipLaunchKernelGGL(RemoteCopy_Warp_1Block<2>, 1, BLOCKSIZE, 0, 0, *p_d);
    HIP_CHECK(hipDeviceSynchronize());

    hipFree(src7);
    for (int i = 0; i < devices; i++)
    {
       compareArrays(hostArray + i * p.size, localMem[i], numElems / 8);
       printf("device %d, %d, %d\n", i, localMem[i][0], localMem[i][p.size-1]);
    }
    hipFree(p_d);

    for (int i = 0; i < devices; i++)
        hipFree(localMem[i]);
    free(localMem);
    return 0;
}
