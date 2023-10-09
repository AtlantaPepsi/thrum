#include "header.hpp"
#include "util.hpp"

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
    //PtoP test (0->7)

    HIP_CHECK(hipSetDevice(1));
    hipLaunchKernelGGL(RemoteCopy_PtoP<1>, 1, BLOCKSIZE, 0, 0,
                       numElems, localMem[0], localMem[devices-1]);
    HIP_CHECK(hipDeviceSynchronize());

    //correctness
    compareArrays(hostArray, localMem[0], numElems);


    for (int i = 0; i < devices; i++)
        hipFree(localMem[i]);
    free(localMem);
    return 0;
}
