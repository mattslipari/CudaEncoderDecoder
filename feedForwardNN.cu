#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cublas_v2.h"
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

__global__ void relu(float* input, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    input[idx]=fmaxf(0.0,input[idx]);
}

__global__ void softmax(float *A, float* p, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float sum = 0;
        for(int i=0;i<n;i++){
            p[i] = expf(A[i]);
            sum = sum + p[i];
        }
        p[idx] = logf(p[idx] / sum);
    }
}

void fully_connected(cublasHandle_t* handle,float* x, float* y, float* w, int n, int m){
    float alpha = 1;
    float beta = 0;
    cublasSgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, m,
                &alpha,
                w, m,
                x, m,
                &beta,
                y, n);
    dim3 blockDim(16);

    relu<<<blockDim,1>>> (y, n);
}

void forward(float* x, float* w1, float* w2,float* y, float* loss, int n, int m) {
    // cudaError_t cudaStat;
    // cublasStatus_t stat;
    cublasHandle_t handle;
    // stat = cublasCreate(&handle);
		dim3 blockDim(16);
    float* output1;
  	float* output2;
  	float* predict;
    cudaMalloc((void**)&output1 ,n*sizeof(*x));
    cudaMalloc((void**)&output2, n*sizeof(*x));
    cudaMalloc((void**)&predict, n*sizeof(*x));
  
    fully_connected(&handle, x, output1, w1, n, m);
    fully_connected(&handle, output1, output2, w2, n, n);
  
    //float index;
    //cublasIsamin(handle,n,output,1,&index);
    softmax<<<blockDim,1>>> (output2,predict,n);

    // stat = cublasSdot(handle,n,predict,1,y,1,loss);
    cublasSdot(handle,n,predict,1,y,1,loss);
}

int main() {
    float* x;
    float* y;
    float* w1;
    float* w2;

    int n = 5;
    int m = 3;
    float loss;

    x = (float *)malloc (m * sizeof(*x));
    y = (float *)malloc (n * sizeof(*y));
    w1 = (float *)malloc (n * m * sizeof(*w1));
    w2 = (float *)malloc (n * m * sizeof(*w2));

    for(int j = 0; j < m; j++) {
        x[j] = (float)j;
        for (int i = 0; i < n; i++) {
            w1[i*m+j] = j;
            w2[i*m+j] = i;
        }
    }

    float* c_x = NULL;
    float* c_y = NULL;
    float* c_w1 = NULL;
    float* c_w2 = NULL;

    cudaMalloc((void**)c_x, sizeof(x));
    cudaMalloc((void**)c_y, sizeof(y));
    cudaMalloc((void**)c_w1, sizeof(w1));
    cudaMalloc((void**)c_w2, sizeof(w2));

    cublasSetMatrix(n, m, sizeof(float), (void *)w1, n, (void *)c_w1, n);
    cublasSetMatrix(n, m, sizeof(float), (void *)w2, n, (void *)c_w2, n);

    cublasSetMatrix(m, 1, sizeof(float), (void *)x, m, (void *)c_x, m);
    cublasSetMatrix(n, 1, sizeof(float), (void *)y, n, (void *)c_y, n);

    forward(x, w1, w2, y, &loss, n, m);
  	return 0;
}

