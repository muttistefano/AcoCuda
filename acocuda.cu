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
typedef unsigned short int sint;

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

__device__ void eval(int* solptr,float* phobjpnt, joints* grp, int npts, int ncfg)// funzione obiettivo da personalizzare
{
  float phinc=0;
  float nrm1[6]={0,0,0,0,0,0};
  float nrm2=0;
  for(int e=0;e<npts-1;e++)
  {
    nrm1[0] = (*(grp+e*ncfg+solptr[e+threadIdx.x*npts])).jointsval[0]-(*(grp+(e+1)*ncfg+solptr[e+(threadIdx.x*npts)+1])).jointsval[0];
    nrm1[1] = (*(grp+e*ncfg+solptr[e+threadIdx.x*npts])).jointsval[1]-(*(grp+(e+1)*ncfg+solptr[e+(threadIdx.x*npts)+1])).jointsval[1];
    nrm1[2] = (*(grp+e*ncfg+solptr[e+threadIdx.x*npts])).jointsval[2]-(*(grp+(e+1)*ncfg+solptr[e+(threadIdx.x*npts)+1])).jointsval[2];
    nrm1[3] = (*(grp+e*ncfg+solptr[e+threadIdx.x*npts])).jointsval[3]-(*(grp+(e+1)*ncfg+solptr[e+(threadIdx.x*npts)+1])).jointsval[3];
    nrm1[4] = (*(grp+e*ncfg+solptr[e+threadIdx.x*npts])).jointsval[4]-(*(grp+(e+1)*ncfg+solptr[e+(threadIdx.x*npts)+1])).jointsval[4];
    nrm1[5] = (*(grp+e*ncfg+solptr[e+threadIdx.x*npts])).jointsval[5]-(*(grp+(e+1)*ncfg+solptr[e+(threadIdx.x*npts)+1])).jointsval[5];
    
    nrm2 = sqrt(nrm1[0]*nrm1[0]+nrm1[1]*nrm1[1]+nrm1[2]*nrm1[2]+nrm1[3]*nrm1[3]+nrm1[4]*nrm1[4]+nrm1[5]*nrm1[5]) ;
    phinc = phinc + __fdividef(1, nrm2); 
  }
*phobjpnt=1000*__fdividef(phinc,npts-1);
}

__global__ void Cycle(int n_pnt,int n_conf,int n_threads,joints* dev_graph_ptr,unsigned int seed,int n_cycles,float phmin,float phmax,float phdec)
{
  
  float rnd_sel,prev_ph,tot_ph;
  curandState_t state;
  extern __shared__ int shmem[];
  int *sol             = (int *)&shmem;
  float *phobj         = (float *)&shmem[n_pnt*n_threads];
  
  curand_init(clock64() ,threadIdx.x, 0, &state);
  __syncthreads();
  
  for(int cyc=0;cyc<n_cycles;cyc++) //CYCLE NUMBER
  {
    
    for (int pnt=0 ; pnt<n_pnt ; pnt++) //PROBABILISTIC SELECTION IMPLEMENTATION
    {
      prev_ph=0;
      tot_ph =0;
      rnd_sel=curand_uniform(&state);
      for(int cht=0;cht<n_conf;cht++)  //sum of all pherormone in configurations
      {
	tot_ph=tot_ph+(*(dev_graph_ptr+pnt*n_conf+cht)).ph;
      }
      rnd_sel = rnd_sel * tot_ph;
      
      for(int conf=0;conf<n_conf;conf++)
      {
	prev_ph=prev_ph+(*(dev_graph_ptr+pnt*n_conf+conf)).ph;
	if(rnd_sel<=prev_ph)
	{
	  sol[threadIdx.x*n_pnt + pnt]=conf;
	  break;
	}
	if(rnd_sel>prev_ph && conf==(n_conf-1)){
	  printf("BIG ERROR\n");
	  return;
	}
      }
    }
    __syncthreads();
    
    eval(sol,&phobj[threadIdx.x],dev_graph_ptr,n_pnt,n_conf); //calcolo ph per ogni soluzione totale
//     printf("ph obj : %f\n",phobj[threadIdx.x]);
    
    if(threadIdx.x<n_pnt)
    {
      for(int mm=0;mm<n_conf;mm++)
      {
	if((*(dev_graph_ptr+threadIdx.x*n_conf+mm)).ph > phmin && (*(dev_graph_ptr+threadIdx.x*n_conf+mm)).ch)
	{
	  atomicMul(&(*(dev_graph_ptr+threadIdx.x*n_conf+mm)).ph,phdec); //FIX
          //atomicAdd(&(*(dev_graph_ptr+threadIdx.x*n_conf+mm)).ph,0.01); //FIX
	}
      }
    }
    __syncthreads();
    
    if(threadIdx.x<n_pnt)
    {
      for(int q=0;q<n_threads;q++) 
      {
	if((*(dev_graph_ptr+threadIdx.x*n_conf+sol[q*n_pnt+threadIdx.x])).ph < phmax )
	{
	  atomicAdd(&(*(dev_graph_ptr+threadIdx.x*n_conf+sol[q*n_pnt+threadIdx.x])).ph,phobj[q]); 
	}
      }
    }
    
  __syncthreads();
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
    float phmin;
    float phmax;
    float phdec;
    
    thrust::host_vector<joints>   host_graph;
    thrust::device_vector<joints> device_graph;
    joints*                       device_graph_ptr;
    joints*                       host_graph_ptr;

    
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

///////////CLASS METHODS

AcoCuda::AcoCuda(int n_pointsex, int n_confex,int ncyc,float phminex,float phmaxex,float phdecex)
{
  n_conf=n_confex;
  n_points=n_pointsex;
  n_cycles=ncyc;
  n_threads=static_cast<int>(ceil(static_cast<float>(n_points)/64)*64); //32 or 16??
  n_blocks=1;
  phmin=phminex;
  phmax=phmaxex;
  phdec=phdecex;
  
  thrust::host_vector<joints> tmp(n_pointsex*n_confex);
  host_graph=tmp;
}

void AcoCuda::LoadGraph()
{
  srand(time(NULL));

  for(thrust::host_vector<joints>::iterator j = host_graph.begin(); j != host_graph.end(); j++)
  {
    if(rand()<(RAND_MAX*0.6))
    {
    (*j).ch=true;
    (*j).jointsval[0]=-20+static_cast <float> (rand()) / static_cast <float> (RAND_MAX/40);
    (*j).jointsval[1]=-30+static_cast <float> (rand()) / static_cast <float> (RAND_MAX/60);
    (*j).jointsval[2]=-30+static_cast <float> (rand()) / static_cast <float> (RAND_MAX/60);
    (*j).jointsval[3]=-180+static_cast <float> (rand()) / static_cast <float> (RAND_MAX/360);
    (*j).jointsval[4]=-180+static_cast <float> (rand()) / static_cast <float> (RAND_MAX/360);
    (*j).jointsval[5]=-180+static_cast <float> (rand()) / static_cast <float> (RAND_MAX/360);
    }
    else
    {
    (*j).ch=false;
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
  for(thrust::host_vector<joints>::iterator j = host_graph.begin(); j != host_graph.end();){
    n_act=0;
    for (int u=0;u<n_conf;u++)
    {
      n_act=n_act+(*j).ch;
      j++;
    }
    n_act = 1/n_act;
    ph_ind.push_back(n_act);
  }
  for(thrust::host_vector<joints>::iterator z = host_graph.begin(); z != host_graph.end();){
    
    for (int uu=0;uu<n_conf;uu++)
    {
      if ((*z).ch){
	 (*z).ph=ph_ind[ind];
      }
      else{
	(*z).ph=0;
      }
      z++;
    }
    ind++;
  }

   device_graph=host_graph;
   device_graph_ptr = thrust::raw_pointer_cast((device_graph.data()));
   host_graph_ptr   = thrust::raw_pointer_cast((host_graph.data()));
}


/////////////METHODS 

void AcoCuda::RunCycle() //launch cuda kernel 
{
  printf("points: %d\n",n_points);
  printf("config: %d\n",n_conf);
  printf("threads: %d\n",n_threads);
  printf("blocks:  %d\n",n_blocks);
  printf("cycles:  %d\n",n_cycles);
  printf("ph min: %f\n",phmin);
  printf("ph max:  %f\n",phmax);
  printf("ph evaporation:  %f\n",phdec);
  size_t shrbytes =(n_points*n_threads)*sizeof(int)+n_threads*sizeof(float);
  printf("shared bytes: %lu\n",shrbytes);
  Cycle<<<n_blocks,n_threads,shrbytes>>>(n_points,n_conf,n_threads,device_graph_ptr,time(NULL),1,phmin,phmax,phdec);//<<<blocks,thread>>>
  if (cudaSuccess != cudaDeviceSynchronize()) {
    printf("ERROR in Cycle\n");
    exit(-2);
  }
}

void AcoCuda::print_file(bool jnts) //log data to external file
{
    FILE *fp;
    std::ostringstream name;
    name << "log/" << "pnt" << n_points << "cnf" << n_conf << "cyc" << n_cycles << "phmin" << phmin << "phmax" << phmax << "phdec" << phdec;
    fp = fopen(name.str().c_str(),"a");
    joints* ptr = this->host_graph_ptr;
    if(0){ //fix this
      for(int i=0; i < n_points; i++)
      {
	for(int k=0;k<6;k++)
	{
	  for(int j=0; j<n_conf; j++)
	  {
	    if ((*(ptr+j+n_conf*i)).jointsval[k]>0){
	    if ((*(ptr+j+n_conf*i)).jointsval[k] < 1000 && (*(ptr+j+n_conf*i)).jointsval[k] > 100) fprintf(fp,"  ");
	    if ((*(ptr+j+n_conf*i)).jointsval[k] < 100 && (*(ptr+j+n_conf*i)).jointsval[k] > 10) fprintf(fp,"   ");
	    if ((*(ptr+j+n_conf*i)).jointsval[k] < 10) fprintf(fp,"    ");
	    }
	    if ((*(ptr+j+n_conf*i)).jointsval[k]<0){
	    if ((*(ptr+j+n_conf*i)).jointsval[k] > -1000 && (*(ptr+j+n_conf*i)).jointsval[k] < -100) fprintf(fp," ");
	    if ((*(ptr+j+n_conf*i)).jointsval[k] > -100 && (*(ptr+j+n_conf*i)).jointsval[k] < -10) fprintf(fp,"  ");
	    if ((*(ptr+j+n_conf*i)).jointsval[k] > -10) fprintf(fp,"   ");
	    
	    }
	    if ((*(ptr+j+n_conf*i)).jointsval[k]==0) fprintf(fp,"    ");
	    fprintf(fp,"  %.2f",(*(ptr+j+n_conf*i)).jointsval[k]);
	  }
	  fprintf(fp,"\n");
	}
	fprintf(fp,"\n\n\n");
      }
    }
    for(int z=0; z < n_points*n_conf; z++){
      if (ptr->ph < 1000 && ptr->ph > 100) fprintf(fp," ");
      if (ptr->ph < 100 && ptr->ph > 10) fprintf(fp,"  ");
      if (ptr->ph < 10) fprintf(fp,"   ");
      fprintf(fp,"  %.2f",ptr->ph);
      if (z%n_conf==(n_conf-1)) fprintf(fp,"\n");
      ptr++;
    }
    fprintf(fp,"\n");
    fclose(fp);
}

void AcoCuda::copytohost() //copy results from gpu to host
{
  thrust::copy(this->device_graph.begin(),this->device_graph.end(),this->host_graph.begin());
  cudaDeviceSynchronize();
}

////////////MAIN

int main(int argc, char *argv[]){
  float pointsnumber;
  int configurations=10;
  
  pointsnumber=atof(argv[1]);
//   int ncyc=atoi(argv[2]);;
  int ncyc = 500;
  
  for(float gg=0.1;gg<=0.95;gg=gg+0.05){
    AcoCuda test(pointsnumber,configurations,ncyc,0.15,2000,gg);//points,conf,cycles,phmin,phmax,phdec
    
    test.LoadGraph();
    test.PhInit();
    
    for (int t=0;t<ncyc;t++){
      test.RunCycle();
      test.copytohost();
      test.print_file(t==0);
    }
    test.~AcoCuda();
  }
  
  return 0;
}


