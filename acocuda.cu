#include <stdio.h>
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
#define Infinity 65536
#define randomize() srand((unsigned)time(NULL))
#define index(length,line,column) (column + line * length)


struct joints{
    float joint1;
    float joint2;
    float joint3;
    float joint4;
    float joint5;
    float joint6;
    bool ch = false;
    float ph = 0.0;
};


//////////DEVICE FUNCTIONS

/*__device__ double atomicMul(double* address, double val) 
{ 
 unsigned long long int* address_as_ull = (unsigned long long int*)address; 
 unsigned long long int old = *address_as_ull, assumed; 
 do { 
 assumed = old; 
 old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val * __longlong_as_double(assumed))); 
 } while (assumed != old);
 return __longlong_as_double(old);
}     */ 

__device__ float atomicMul(float* address, float val) 
{
  int* address_as_int = (int*)address; 
  int old = *address_as_int, assumed; 
  do { 
    assumed = old; 
    old = atomicCAS(address_as_int, assumed, __float_as_int(val * 
__float_as_int(assumed))); 
 } while (assumed != old); return __int_as_float(old);
}

__global__ void Cycle(int n_pnt,int n_conf,int n_ants,joints* dev_graph_ptr,unsigned int seed)
{
  curandState_t state;

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int index_ch = index * n_pnt;
  float rnd_sel,prev_ph,tot_ph;

  __shared__ int sol[10000]; //controlla lunghezza o dynamic 
//   int* sol = new int[n_pnt];
  
  curand_init(clock(),index,0, &state);
  
  for (int pnt=0 ; pnt<n_pnt ; pnt++) //PROBABILISTIC SELECTION IMPLEMENTATION
  {
    prev_ph=0;
    tot_ph =0;
    rnd_sel=curand_uniform(&state);
    
    for(int cht=0;cht<n_conf;cht++)
    {
      tot_ph=tot_ph+(*(dev_graph_ptr+pnt+cht*n_pnt)).ph; //SHARED MEMORY <----
    }
    
    rnd_sel = rnd_sel * tot_ph;
//     printf("randomnum:%f  \n",rnd_sel);
    
    for(int conf=0;conf<n_conf;conf++)
    {
      prev_ph=prev_ph+(*(dev_graph_ptr+pnt+conf*n_pnt)).ph;
      if(rnd_sel<prev_ph)
      {
	sol[threadIdx.x*n_pnt + pnt]=conf;
	break;
      }
    }

  }
  /*
  for(int gg=0;gg<n_pnt;gg++){
    printf(" %d ",sol[threadIdx.x*n_pnt + gg]);
  }*/
  printf(" %d ",sol[threadIdx.x*n_pnt]);
//   printf("\n ");
  
  __syncthreads();
  
  for(int q=0;q<n_ants;q++) //PH VALUE ADDING ---- n_threads α n_points ---- OPTIMIZE 
  {
    atomicAdd(&(*(dev_graph_ptr+threadIdx.x+n_pnt*sol[q*n_pnt+threadIdx.x])).ph,0.1); //BOH
//     atomicMul(&(*(dev_graph_ptr+threadIdx.x)).ph,1.02);
  }
  
  for(int mm=0;mm<(int)((n_conf*n_pnt)/n_ants);mm++)  //FIX THIS FOR EVERY CASE 
  {
    if((*(dev_graph_ptr+threadIdx.x+n_pnt*mm)).ph > 0.2 & (*(dev_graph_ptr+threadIdx.x+n_pnt*mm)).ch)
    {
      atomicAdd(&(*(dev_graph_ptr+threadIdx.x+n_pnt*mm)).ph,-0.1);
    }
  }
}

__global__ void print_matrix(joints* ptr,int n_points,int n_conf){
    for(int i=0; i < n_points*n_conf; ++i){
        printf("%f ",ptr->ph);
	if (i%n_points==(n_points-1)) printf("\n");
        ptr++;
    }
}


///////////CLASS


class AcoCuda
{
    int n_points;
    int n_conf;
    int n_ants;

    thrust::host_vector<joints>   host_graph;
    thrust::host_vector<int>      host_path;
    thrust::device_vector<joints> device_graph;
    joints*                       device_graph_ptr;

    
  public:

    AcoCuda(int n_points,int n_conf,int n_ants);
    
    void LoadGraph();
    void PhInit();
    void Phrenew();
    void Phevaporate();
    
    void RunCycle();
    void RunPrint();

};

///////////CLASS METHODS

AcoCuda::AcoCuda(int n_pointsex, int n_confex, int n_antsex)
{
  n_ants=n_antsex;
  n_conf=n_confex;
  n_points=n_pointsex;
  thrust::host_vector<joints> tmp(n_pointsex*n_confex);
  host_graph=tmp;
}

void AcoCuda::LoadGraph()
{
  srand(time(NULL));
  printf("points: %d\n",n_points);
  printf("config: %d\n",n_conf);

  for(thrust::host_vector<joints>::iterator j = host_graph.begin(); j != host_graph.end(); j++){
    (*j).joint1=rand()/(RAND_MAX/3);
    (*j).joint2=rand()/(RAND_MAX/3);
    (*j).joint3=rand()/(RAND_MAX/3);
    (*j).joint4=rand()/(RAND_MAX/3);
    (*j).joint5=rand()/(RAND_MAX/3);
    (*j).joint6=rand()/(RAND_MAX/3);

 if(rand()<(RAND_MAX*0.6)){
      (*j).ch=true;
    }      
  
  }

}

void AcoCuda::PhInit()
{
  float n_act;
  int ind;
  ind=0;
  std::vector<float> ph_ind;
  ph_ind.clear();
  for(thrust::host_vector<joints>::iterator j = host_graph.begin(); j != host_graph.begin()+n_points; j++){
    n_act=0;
    for (int u=0;u<n_conf;u++)
    {
      n_act=n_act+(*(j+u*n_points)).ch;
    }
//     printf("%d\n",n_act);
    n_act = 1/n_act;
    ph_ind.push_back(n_act);
  }
  for(thrust::host_vector<joints>::iterator z = host_graph.begin(); z != host_graph.begin()+n_points; z++){
    
    for (int uu=0;uu<n_conf;uu++)
    {
      if ((*(z+uu*n_points)).ch){
	 (*(z+uu*n_points)).ph=ph_ind[ind];
      }
      else{
	(*(z+uu*n_points)).ph=0;
      }
    }
    ind++;
  }

   device_graph=host_graph;
   device_graph_ptr = thrust::raw_pointer_cast((device_graph.data()));
//    device_graph_ptr = thrust::raw_pointer_cast(&device_graph[0]);
}


/////////////METHODS FOR CALLING DEVICES FUNCTIONS

void AcoCuda::RunCycle()
{
  Cycle<<<1,n_ants >>>(n_points,n_conf,n_ants,device_graph_ptr,time(NULL));//<<<blocks,thread>>>
  
}

void AcoCuda::RunPrint()
{
  print_matrix<<< 1,1 >>>(this->device_graph_ptr,n_points,n_conf);
}

////////////MAIN

int main(){
 
  AcoCuda test(10,8,10);//points,conf,ants ---- MAX 128 pnt con 8 configurazioni per ora

  test.LoadGraph();
  test.PhInit();
  
  test.RunPrint();
  cudaDeviceSynchronize();
  
  for (int y=0;y<1;y++)
  {
    test.RunCycle();
    cudaDeviceSynchronize();
  }
  printf("\n");
  test.RunPrint();
  cudaDeviceSynchronize();
  
  printf("\nEnd\n");
  //test.print_matrix();
  
  return 0;
}


