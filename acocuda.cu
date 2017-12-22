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
#define Infinity 65536
#define randdouble() ((double)rand()/(double)RAND_MAX)
#define randomize() srand((unsigned)time(NULL))
#define index(length,line,column) (column + line * length)


struct joints{
    double joint1;
    double joint2;
    double joint3;
    double joint4;
    double joint5;
    double joint6;
    bool ch;
};

void print_matrix(thrust::device_vector<double > host_ph, int rows, int cols){
    for(int i=0; i < rows; ++i){
        for (int j=0; j < cols; ++j){
             std::cout << host_ph[i*8+j] << " ";
        }
        std::cout << std::endl;
    }
}


//////////DEVICE FUNCTIONS


__global__ void Cycle(int n_pnt,joints* dev_graph_ptr)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int index_ch = index * n_pnt;
  
  printf("index:%d\n",index);
  
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
    thrust::host_vector<double>   host_ph;
    thrust::device_vector<double> device_ph;
    double*                       device_ph_ptr;
    
  public:
  
    
    AcoCuda(int n_points,int n_conf,int n_ants);
    void LoadGraph();
    void PhInit();
    void Phrenew();
    void Phevaporate();
    void RunCycle();

};

///////////CLASS METHODS

AcoCuda::AcoCuda(int n_pointsex, int n_confex, int n_antsex)
{
  this->n_ants=n_antsex;
  this->n_conf=n_confex;
  this->n_points=n_pointsex;
  thrust::host_vector<joints> tmp(n_pointsex*n_confex);
  thrust::host_vector<double> tmpph(n_pointsex*n_confex);
  this->host_graph=tmp;
  this->host_ph=tmpph;
}

void AcoCuda::LoadGraph()
{
  
  for(int j=0;j<n_points*n_conf;j++){
    this->host_graph[j].joint1=rand();
    this->host_graph[j].joint2=rand();
    this->host_graph[j].joint3=rand();
    this->host_graph[j].joint4=rand();
    this->host_graph[j].joint5=rand();
    this->host_graph[j].joint6=rand();
  }
   this->device_graph=this->host_graph;
   this->device_graph_ptr = thrust::raw_pointer_cast((this->device_graph.data()));
}

void AcoCuda::PhInit()
{
  int n_act;
  for (int t=0;t<this->n_points;t++)
  {
    n_act=0;
    for (int u=0;u<this->n_conf;u++)
    {
      n_act=n_act+this->host_graph[t+this->n_conf*u].ch;
    }
    for (int a=0;a<this->n_conf;a++)
    {
      this->host_ph[t+this->n_conf*a]=n_act;
    }
  }
  this->device_ph=this->host_ph;
  this->device_ph_ptr = thrust::raw_pointer_cast((this->device_ph.data()));
}


/////////////METHODS FOR CALLING DEVICES FUNCTIONS

void AcoCuda::RunCycle()
{
  Cycle<<< 1,10 >>>(this->n_points,this->device_graph_ptr);
  cudaDeviceSynchronize();
}

/////////////////////////////////////

int main(){


  AcoCuda test(10,8,1000);//points,conf,ants

  test.LoadGraph();
  test.PhInit();
  test.RunCycle();

  
  return 0;
}


