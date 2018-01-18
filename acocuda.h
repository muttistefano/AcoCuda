#include <stdio.h>
#include <string>
#include <sstream>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <unistd.h>
#include <ctime>
#include <cstdlib>
#include <stdint.h>
#include <cstdio>
#include <sys/mman.h>
#include <cooperative_groups.h>

struct joints{
    float jointsval[6];
    bool ch = false;
    float ph = 0.0;
};


//////////DEVICE FUNCTIONS 

__device__ __forceinline__ float atomicMul(float* address, float val)
{
  int32_t* address_as_int = reinterpret_cast<int32_t*>(address);
  int32_t old = *address_as_int, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_int, assumed, __float_as_int(val * __int_as_float(assumed)));
  } while (assumed != old);
  return __int_as_float(old);
}

///////////CLASS

class AcoCuda
{
    int n_points;
    int n_conf;
    int n_threads;
    int n_cycles;
    int n_blocks;
    float phmin;
    float phmax;
    float phdec;
    
    thrust::host_vector<joints>   host_graph;
    thrust::device_vector<joints> device_graph;
    joints*                       device_graph_ptr;
    joints*                       host_graph_ptr;

    size_t shrbytes;
    
  public:

    AcoCuda(int n_pointsex, int n_confex,int ncyc,float phminex,float phmaxex,float phdecex);
    ~AcoCuda(){};
    void LoadGraph();
    void PhInit();
    void Phrenew();
    void Phevaporate();
    
    void RunCycle();
    void print_file(bool jnts);
    void copytohost();


};
