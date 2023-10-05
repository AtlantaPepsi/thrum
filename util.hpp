#include "header.hpp"

void compareArrays(int* hostArray, int* deviceArray, int numElems) {
    for (auto i=0; i<numElems; i++) 
    {
        if (hostArray[i] != deviceArray[i])
        {
            printf("assertion failed at index %d: %d vs %d\n", 
                    i, hostArray[i], deviceArray[i]);
            return;
        }
    }
}

void SetUpPeer(int devices)
{
    for (int i = 0; i < devices; i++)
    {
        HIP_CHECK(hipSetDevice(i));
        
        //establish peer access
        for (int j = 0; j < devices; j++)
        {
            if (i == j) continue;
            int canAccess = -1;
            HIP_CHECK(hipDeviceCanAccessPeer(&canAccess, i, j));
            if (canAccess)
            {
                HIP_CHECK(hipDeviceEnablePeerAccess(j, 0)); ///what does the flag even do here?
            }
            else
                printf("?\n");            
        } 
        
    }
} 
