#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/complex.h>
#include <sys/time.h>
#include <omp.h>
#include <cuda.h>

extern "C" {
  #include "write.h"
}

const int MAX_ITER = 100;

/*CPU*/
void runCPU(int c0_real, int c0_imag, int c1_real, int c1_imag, int width, int height, int threads, char *output);
void mandelbrotCPU(int index, int c0_real, int c0_imag, int c1_real, int c1_imag, int width, int height, int *buffer);

/*GPU*/
void runGPU(int c0_real, int c0_imag, int c1_real, int c1_imag, int width, int height, int threads, char *output);
__global__ void mandelbrotGPU(int *c0_real, int *c0_imag, int *c1_real, int *c1_imag, int *width, int *height, int *buffer);


int main(int argc, char *argv[]){
    //Input params
    if(argc < 10){
      printf("USAGE: mbrot <C0_REAL> <C0_IMAG> <C1_REAL> <C1_IMAG> <W> <H> <CPU/GPU> <THREADS> <OUTPUT>");
    }

    struct timeval start, end;

    int c0_real = atof(argv[1]);
    int c0_imag = atof(argv[2]);
    int c1_real = atof(argv[3]);
    int c1_imag = atof(argv[4]);
    int width = atof(argv[5]);
    int height = atof(argv[6]);
    char *execution = argv[7];
    int threads = atof(argv[8]);
    char *output = argv[9];

    gettimeofday(&start, NULL);

    if(strncmp(execution, "CPU", 3) == 0){
      runCPU(c0_real, c0_imag, c1_real, c1_imag, width, height, threads, output);
    } else {
      runGPU(c0_real, c0_imag, c1_real, c1_imag, width, height, threads, output);
    }

    gettimeofday(&end, NULL);
    double elapsed_time = (end.tv_sec - start.tv_sec) +
                              (end.tv_usec - start.tv_usec) / 1000000.0;
    printf("%.4fs\n", elapsed_time);

    return 0;
}

void runCPU(int c0_real, int c0_imag, int c1_real, int c1_imag, int width, int height, int threads, char *output){
  int *buffer = (int *) malloc(width * height * sizeof(int));

  omp_set_num_threads(threads);

  #pragma omp parallel for
  for (int i = 0; i < width * height; i++){
    mandelbrotCPU(i, c0_real, c0_imag, c1_real, c1_imag, width, height, buffer);
  }

  writeImage(output, width, height, MAX_ITER, buffer);

  free(buffer);
}

void mandelbrotCPU(int index, int c0_real, int c0_imag, int c1_real, int c1_imag, int width, int height, int *buffer){
    int i;
    int y = index / width;
    int x = index % width;
    float w = width;
    float h = height;

    thrust::complex<float> c = thrust::complex<float>((c0_real + (x / w) * (c1_real - c0_real)),
                                 (c0_imag + (y / h) * (c1_imag - c0_imag)));
    
    thrust::complex<float> z = thrust::complex<float>(0,0);

    for(i = 0; i < MAX_ITER; i++) {
        z = z*z + c;
        if(z.real() > 2 || z.imag() > 2) break;
    }
    
    buffer[y*width + x] = (i == MAX_ITER ? 0 : i);
}

void runGPU(int c0_real, int c0_imag, int c1_real, int c1_imag, int width, int height, int threads, char *output){
  int *buffer = (int *) malloc(width * height * sizeof(int));

  int *d_c0_real, *d_c0_imag, *d_c1_real, *d_c1_imag;
  int *d_width, *d_height;
  int *d_buffer;

  //cuda alloc
  cudaMalloc(&d_c0_real, sizeof(int));
  cudaMalloc(&d_c0_imag, sizeof(int));
  cudaMalloc(&d_c1_real, sizeof(int));
  cudaMalloc(&d_c1_imag, sizeof(int));
  cudaMalloc(&d_width, sizeof(int));
  cudaMalloc(&d_height, sizeof(int));
  cudaMalloc((void **) &d_buffer, width * height * sizeof(int));
  //cuda alloc

  //memcpy
  cudaMemcpy(d_c0_real, &c0_real, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c0_imag, &c0_imag, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c1_real, &c1_real, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c1_imag, &c1_imag, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_width, &width, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_height, &height, sizeof(int), cudaMemcpyHostToDevice);
  //memcpy

  unsigned blocks_per_grid = ceil((width * height) / threads);
  mandelbrotGPU<<< blocks_per_grid , threads >>>(
    d_c0_real, d_c0_imag, d_c1_real, d_c1_imag, d_width, d_height, d_buffer);

  //sync
  cudaDeviceSynchronize();
  cudaMemcpy(buffer, d_buffer, width * height * sizeof(int), cudaMemcpyDeviceToHost);

  //free cuda
  cudaFree(d_c0_real);
  cudaFree(d_c0_imag);
  cudaFree(d_c1_real);
  cudaFree(d_c1_imag);
  cudaFree(d_width);
  cudaFree(d_height);
  cudaFree(d_buffer);
  //free cuda

  writeImage(output, width, height, MAX_ITER, buffer);

  free(buffer);
}

__global__ void mandelbrotGPU(int *c0_real, int *c0_imag, int *c1_real, int *c1_imag, int *width, int *height, int *buffer){
    

    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index > ((*width) * (*height)) - 1){
      return ;
    }
    int y = index / (*width);
    int x = index % (*width);
    int i;
    float w = *width;
    float h = *height;

    thrust::complex<float> c = thrust::complex<float>((*c0_real + (x / w) * (*c1_real - *c0_real)),
                                 (*c0_imag + (y / h) * (*c1_imag - *c0_imag)));
    
    thrust::complex<float> z = thrust::complex<float>(0,0);

    for(i = 0; i < MAX_ITER; i++) {
        z = z*z + c;
        if(z.real() > 2 || z.imag() > 2) break;
    }
    
    buffer[y * (*width) + x] = (i == MAX_ITER ? 0 : i);
}