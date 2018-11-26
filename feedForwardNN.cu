#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cublas_v2.h"

#define IDX2C(i, j, ld) (((j)*(ld))+(i))

__global__ void relu(float *input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    input[idx] = fmaxf(0.0, input[idx]);
}

__global__ void softmax(float *A, float *p, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = 0;
        for (int i = 0; i < n; i++) {
            p[i] = expf(A[i]);
            sum = sum + p[i];
        }
        p[idx] = logf(p[idx] / sum);
    }
}

static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

void fully_connected(cublasHandle_t *handle, float *x, float *y, float *w, int n, int m) {
    float alpha = 1;
    float beta = 0;
    cublasStatus_t stat;
    stat=cublasSgemv(*handle, CUBLAS_OP_N,
                n, m,
                &alpha,
                w, n,
                x, 1,
                &beta,
                y, 1);
    dim3 blockDim(16);
    printf("%s\n", _cudaGetErrorEnum(stat));

    float *tmp = (float *) malloc(n * sizeof(*y));
    cublasGetVector(n, sizeof(*y), y, 1, tmp, 1); //cp d_c->c printf("c after Sgemm :\n");
    for (int i = 0; i < n; i++) {
        printf("%7.0f", tmp[i]); //print c after Sgemm
    }

    relu <<< blockDim, 1 >>> (y, n);

}

void forward(float *x, float *w1, float *w2, float *y, float *loss, int n, int m) {
    // cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    stat = cublasCreate(&handle);
    dim3 blockDim(16);
    float *output1;
    float *output2;
    float *predict;

    cudaMalloc((void **) &output1, n * sizeof(*x));
    cudaMalloc((void **) &output2, n * sizeof(*x));
    cudaMalloc((void **) &predict, n * sizeof(*x));

    fully_connected(&handle, x, output1, w1, n, m);
    fully_connected(&handle, output1, output2, w2, n, n);

    //float index;
    //cublasIsamin(handle,n,output,1,&index);
    softmax << < blockDim, 1 >> > (output2, predict, n);

    // stat = cublasSdot(handle,n,predict,1,y,1,loss);
    cublasSdot(handle, n, predict, 1, y, 1, loss);
}

int main() {
    float *x;
    float *y;
    float *w1;
    float *w2;

    int n = 5;
    int m = 3;
    float loss;

    x = (float *) malloc(m * sizeof(*x));
    y = (float *) malloc(n * sizeof(*y));
    w1 = (float *) malloc(n * m * sizeof(*w1));
    w2 = (float *) malloc(n * n * sizeof(*w2));

    for (int j = 0; j < m; j++) {
        x[j] = 1;
    }
    for (int j = 0; j < n * m; j++) {
        w1[j] = 1;
    }

    for (int j = 0; j < n * n; j++) {
        w2[j] = 1;
    }

    float *c_x = NULL;
    float *c_y = NULL;
    float *c_w1 = NULL;
    float *c_w2 = NULL;

    cudaMalloc((void **) &c_x, m * sizeof(*x));
    cudaMalloc((void **) &c_y, n * sizeof(*y));
    cudaMalloc((void **) &c_w1, n * m * sizeof(*w1));
    cudaMalloc((void **) &c_w2, n * n * sizeof(*w2));

    cublasSetMatrix(n, m, sizeof(float), (void *) w1, n, (void *) c_w1, n);
    cublasSetMatrix(n, n, sizeof(float), (void *) w2, n, (void *) c_w2, n);

    cublasSetVector(m, sizeof(float), (void *) x, 1, (void *) c_x, 1);
    cublasSetVector(n, sizeof(float), (void *) y, 1, (void *) c_y, 1);

    forward(c_x, c_w1, c_w2, c_y, &loss, n, m);
    return 0;
}
