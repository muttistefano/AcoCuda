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
#include <stdint.h>
#include <cstdio>


struct joints{
    float jointsval[6];
    bool ch = false;
    float ph = 0.0;
};


//////////DEVICE FUNCTIONS 


__device__ __forceinline__ float atomicMul(float* address, float val) {
  int32_t* address_as_int = reinterpret_cast<int32_t*>(address);
  int32_t old = *address_as_int, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_int, assumed, __float_as_int(val * __int_as_float(assumed)));
  } while (assumed != old);
  return __int_as_float(old);
}

__device__ void eval(int* solptr,float* phobjpnt, joints* grp, int npts){
  float phinc=0;
  float nrm1[6];
  float nrm2=0;
  for(int e=0;e<npts-1;e++)
  {
    nrm1[0] = (*(grp+e+npts*solptr[e+threadIdx.x*npts])).jointsval[0]-(*(grp+e+1+npts*solptr[e+threadIdx.x*npts+1])).jointsval[0];
    nrm1[1] = (*(grp+e+npts*solptr[e+threadIdx.x*npts])).jointsval[1]-(*(grp+e+1+npts*solptr[e+threadIdx.x*npts+1])).jointsval[1];
    nrm1[2] = (*(grp+e+npts*solptr[e+threadIdx.x*npts])).jointsval[2]-(*(grp+e+1+npts*solptr[e+threadIdx.x*npts+1])).jointsval[2];
    nrm1[3] = (*(grp+e+npts*solptr[e+threadIdx.x*npts])).jointsval[3]-(*(grp+e+1+npts*solptr[e+threadIdx.x*npts+1])).jointsval[3];
    nrm1[4] = (*(grp+e+npts*solptr[e+threadIdx.x*npts])).jointsval[4]-(*(grp+e+1+npts*solptr[e+threadIdx.x*npts+1])).jointsval[4];
    nrm1[5] = (*(grp+e+npts*solptr[e+threadIdx.x*npts])).jointsval[5]-(*(grp+e+1+npts*solptr[e+threadIdx.x*npts+1])).jointsval[5];
    
    nrm2 = 0.05*sqrt(nrm1[0]*nrm1[0]+nrm1[1]*nrm1[1]+nrm1[2]*nrm1[2]+nrm1[3]*nrm1[3]+nrm1[4]*nrm1[4]+nrm1[5]*nrm1[5]) ;
    phinc = phinc + 1/nrm2;
  }
  *phobjpnt=phinc;
}

__global__ void Cycle(int n_pnt,int n_conf,int n_threads,joints* dev_graph_ptr,unsigned int seed,int n_cycles)
{
  curandState_t state;

  int index = threadIdx.x + blockIdx.x * blockDim.x;

  float rnd_sel,prev_ph,tot_ph;

  __shared__ int sol[10000]; //controlla lunghezza o dynamic
  __shared__ float phobj[1000]; //uno per ogni formica se tutto il path viene agiornato con lo stesso ferormone
  
  curand_init(clock(),index,0, &state);
  for(int cyc=0;cyc<n_cycles;cyc++) //CYCLE NUMBER
  {
    
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
    
//     for(int gg=0;gg<n_pnt;gg++){
//       printf(" %d ",sol[threadIdx.x*n_pnt + gg]);
//     }

  //   printf("\n ");
    
    __syncthreads();
    
    for(int q=0;q<n_threads;q++) //PH VALUE ADDING ---- n_threads α n_points ---- Ottimizza per fare lavorare tutti i thread assime
    {
      if(threadIdx.x<n_pnt)
      {
	if((*(dev_graph_ptr+threadIdx.x+n_pnt*sol[q*n_pnt+threadIdx.x])).ph < 1000 )
	{
	  eval(sol,&phobj[threadIdx.x],dev_graph_ptr,n_pnt);
//   	  printf("value : %f \n", phobj[threadIdx.x]);
	  atomicMul(&(*(dev_graph_ptr+threadIdx.x+n_pnt*sol[q*n_pnt+threadIdx.x])).ph,phobj[threadIdx.x]); 
	}
      }
    }
    
    
    /*for(int mm=0;mm<(int)((n_conf*n_pnt)/n_threads);mm++)*/  //FIX THIS FOR EVERY CASE 
    for(int mm=0;mm<n_conf;mm++)
    {
      if(threadIdx.x<n_pnt) //più efficente 
      {
	if((*(dev_graph_ptr+threadIdx.x+n_pnt*mm)).ph > 0.1 & (*(dev_graph_ptr+threadIdx.x+n_pnt*mm)).ch)
	{
	  atomicMul(&(*(dev_graph_ptr+threadIdx.x+n_pnt*mm)).ph,0.4); //FIX
	}
      }
    }

  }
  
}


__global__ void print_matrix(joints* ptr,int n_points,int n_conf){
  for(int i=0; i < n_points*n_conf; ++i){
      if (ptr->ph < 100 & ptr->ph > 10) printf("  ");
      if (ptr->ph < 10) printf("   ");
      printf("  %.2f",ptr->ph);
      if (i%n_points==(n_points-1)) printf("\n");
      ptr++;
  }
}


///////////CLASS


class AcoCuda
{
    int n_points;
    int n_conf;
    int n_threads;
    int n_cycles;
    int n_blocks;
    
    thrust::host_vector<joints>   host_graph;
    thrust::device_vector<joints> device_graph;
    joints*                       device_graph_ptr;
    joints*                       host_graph_ptr;

    
  public:

    AcoCuda(int n_points,int n_conf,int n_nths,int nblks,int ncyc);
    
    void LoadGraph();
    void PhInit();
    void Phrenew();
    void Phevaporate();
    
    void RunCycle();
    void RunPrint();
    void print_file();
    void copytohost();


};

///////////CLASS METHODS

AcoCuda::AcoCuda(int n_pointsex, int n_confex,int nths, int nblks,int ncyc)
{
  n_conf=n_confex;
  n_points=n_pointsex;
  n_cycles=ncyc;
  n_threads=nths;
  n_blocks=nblks;
  
  thrust::host_vector<joints> tmp(n_pointsex*n_confex);
  host_graph=tmp;
}

void AcoCuda::LoadGraph()
{
  srand(time(NULL));
  printf("points: %d\n",n_points);
  printf("config: %d\n",n_conf);
  printf("threads: %d\n",n_threads);
  printf("blocks: %d\n",n_blocks);
  printf("cycles: %d\n",n_cycles);

  for(thrust::host_vector<joints>::iterator j = host_graph.begin(); j != host_graph.end(); j++)
  {
    if(rand()<(RAND_MAX*0.6))
    {
    (*j).ch=true;
    (*j).jointsval[0]=-20+static_cast <float> (rand()) / static_cast <float> (RAND_MAX/40);
//     printf("1: %f\n",(*j).jointsval[0]);
    (*j).jointsval[1]=-30+static_cast <float> (rand()) / static_cast <float> (RAND_MAX/60);
//     printf("2: %f\n",(*j).jointsval[1]);
    (*j).jointsval[2]=-30+static_cast <float> (rand()) / static_cast <float> (RAND_MAX/60);
//     printf("3: %f\n",(*j).jointsval[2]);
    (*j).jointsval[3]=-180+static_cast <float> (rand()) / static_cast <float> (RAND_MAX/360);
//     printf("4: %f\n",(*j).jointsval[3]);
    (*j).jointsval[4]=-180+static_cast <float> (rand()) / static_cast <float> (RAND_MAX/360);
//     printf("5: %f\n",(*j).jointsval[4]);
    (*j).jointsval[5]=-180+static_cast <float> (rand()) / static_cast <float> (RAND_MAX/360);
//     printf("6: %f\n",(*j).jointsval[5]);
    }
    else
    {
      (*j).jointsval[0]=0;
    (*j).jointsval[1]=0;
    (*j).jointsval[2]=0;
    (*j).jointsval[3]=0;
    (*j).jointsval[4]=0;
    (*j).jointsval[5]=0;
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
   host_graph_ptr   = thrust::raw_pointer_cast((host_graph.data()));

}


/////////////METHODS 

void AcoCuda::RunCycle()
{
  Cycle<<<n_blocks,n_threads >>>(n_points,n_conf,n_threads,device_graph_ptr,time(NULL),n_cycles);//<<<blocks,thread>>>
}

void AcoCuda::RunPrint()
{
  print_matrix<<< 1,1 >>>(this->device_graph_ptr,n_points,n_conf);
}


 void AcoCuda::print_file(){
    FILE *fp;
    fp = fopen("log.txt","w");
//     fprintf(fp,"");
    joints* ptr = this->host_graph_ptr;
    for(int i=0; i < n_conf; i++)
    {
      for(int k=0;k<6;k++)
      {
        for(int j=0; j<n_points; j++)
	{
	  if ((*(ptr+j+n_points*i)).jointsval[k]>0){
	  if ((*(ptr+j+n_points*i)).jointsval[k] < 1000 & (*(ptr+j+n_points*i)).jointsval[k] > 100) fprintf(fp,"  ");
          if ((*(ptr+j+n_points*i)).jointsval[k] < 100 & (*(ptr+j+n_points*i)).jointsval[k] > 10) fprintf(fp,"   ");
          if ((*(ptr+j+n_points*i)).jointsval[k] < 10) fprintf(fp,"    ");
	  }
	  if ((*(ptr+j+n_points*i)).jointsval[k]<0){
	  if ((*(ptr+j+n_points*i)).jointsval[k] > -1000 & (*(ptr+j+n_points*i)).jointsval[k] < -100) fprintf(fp," ");
          if ((*(ptr+j+n_points*i)).jointsval[k] > -100 & (*(ptr+j+n_points*i)).jointsval[k] < -10) fprintf(fp,"  ");
          if ((*(ptr+j+n_points*i)).jointsval[k] > -10) fprintf(fp,"   ");
	  
	  }
	  if ((*(ptr+j+n_points*i)).jointsval[k]==0) fprintf(fp,"    ");
	  fprintf(fp,"  %.2f",(*(ptr+j+n_points*i)).jointsval[k]);
	}
	fprintf(fp,"\n");
      }
      fprintf(fp,"\n\n\n");
    }
    
    for(int z=0; z < n_points*n_conf; z++){
    if (ptr->ph < 100 & ptr->ph > 10) fprintf(fp,"  ");
    if (ptr->ph < 10) fprintf(fp,"   ");
    fprintf(fp,"  %.2f",ptr->ph);
    if (z%n_points==(n_points-1)) fprintf(fp,"\n");
    ptr++;
}
}

void AcoCuda::copytohost()
{
  thrust::copy(this->device_graph.begin(),this->device_graph.end(),this->host_graph.begin());
}

////////////MAIN

int main(int argc, char *argv[]){
 
  int nth=15,nbl=1,ncyc=20;
  int pointsnumber=15;
  int configurations=8;
  if(argc==4)
  {
    nth = strtol(argv[1], NULL, 10);
    nbl = strtol(argv[2], NULL, 10);
    ncyc= strtol(argv[3], NULL, 10);
  }
  
  printf("size of joints : %d\n\n",sizeof(joints));
  
  if(pointsnumber>nth)
  {
    printf("Threads cannot be less than points(%d)",pointsnumber);
    return(0);
  }
  
  AcoCuda test(pointsnumber,configurations,nth,nbl,ncyc);//points,conf,threads,blocks 
  test.LoadGraph();
  test.PhInit();
  
//   test.RunPrint();
//   cudaDeviceSynchronize();
//   

  test.RunCycle();
  cudaDeviceSynchronize();

  
//   printf("Sol: \n");
//   test.RunPrint();
//   cudaDeviceSynchronize(); 
//   printf("\nEnd\n");
//   
  test.copytohost();
//   test.print_file();

  
  return 0;
}


