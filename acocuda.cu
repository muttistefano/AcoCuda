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

// void print_matrix(thrust::device_vector<double > host_ph, int rows, int cols){
//     for(int i=0; i < rows; ++i){
//         for (int j=0; j < cols; ++j){
//              std::cout << host_ph[i*8+j] << " ";
//         }
//         std::cout << std::endl;
//     }
// }


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
    void print_matrix();

};

///////////CLASS METHODS

AcoCuda::AcoCuda(int n_pointsex, int n_confex, int n_antsex)
{
  n_ants=n_antsex;
  n_conf=n_confex;
  n_points=n_pointsex;
  thrust::host_vector<joints> tmp(n_pointsex*n_confex);
  thrust::host_vector<double> tmpph(n_pointsex*n_confex);
  host_graph=tmp;
  host_ph=tmpph;
}

void AcoCuda::LoadGraph()
{
  printf("points: %d\n",n_points);
  printf("config: %d\n",n_conf);

  for(thrust::host_vector<joints>::iterator j = host_graph.begin(); j != host_graph.end(); j++){
    (*j).joint1=(double)rand()/(RAND_MAX/3);
    printf("j1:%f\n",(*j).joint1);
    (*j).joint2=(double)rand()/(RAND_MAX/3);
    printf("j2:%f\n",(*j).joint2);
    (*j).joint3=(double)rand()/(RAND_MAX/3);
    printf("j3:%f\n",(*j).joint3);
    (*j).joint4=(double)rand()/(RAND_MAX/3);
    printf("j4:%f\n",(*j).joint4);
    (*j).joint5=(double)rand()/(RAND_MAX/3);
    printf("j5:%f\n",(*j).joint5);
    (*j).joint6=(double)rand()/(RAND_MAX/3);
    printf("j6:%f\n",(*j).joint6);
 
    if(rand()<(RAND_MAX*0.8)){
      printf("false");
      (*j).ch=true;
    }      
    printf("ch:%d\n\n",(*j).ch);
  }
   device_graph=host_graph;
   device_graph_ptr = thrust::raw_pointer_cast((device_graph.data()));
}

void AcoCuda::PhInit()
{
  int n_act;
  for (int t=0;t<n_points;t++)
  {
    n_act=0;
    for (int u=0;u<n_conf;u++)
    {
      n_act=n_act+(int)host_graph[t+n_conf*u].ch;
    }
    for (int a=0;a<n_conf;a++)
    {
      host_ph[t+n_conf*a]=(1/n_act);
    }
  }
  device_ph=host_ph;
  device_ph_ptr = thrust::raw_pointer_cast((device_ph.data()));
}

void AcoCuda::print_matrix(){
    for(int i=0; i < n_conf; ++i){
        for (int j=0; j < n_points; ++j){
             std::cout << host_ph[i*8+j] << " ";
        }
        std::cout << std::endl;
    }
}

/////////////METHODS FOR CALLING DEVICES FUNCTIONS

void AcoCuda::RunCycle()
{
  Cycle<<< 1,10 >>>(n_points,device_graph_ptr);
  cudaDeviceSynchronize();
}

/////////////////////////////////////

int main(){
  srand(time(0));

  AcoCuda test(10,8,1000);//points,conf,ants

  test.LoadGraph();
  //test.PhInit();
  //test.RunCycle();
  //test.print_matrix();
  
  return 0;
}


