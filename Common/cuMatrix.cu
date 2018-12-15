/*
Modified from
https://github.com/zhxfl/CUDA-CNN
*/

#include "cuMatrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "CycleTimer.h"


cublasHandle_t &getHandle() {
    static cublasHandle_t handle = NULL;
    if (handle == NULL) {
        cublasStatus_t stat;
        stat = cublasCreate(&handle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf("init: CUBLAS initialization failed\n");
            exit(0);
        }
    }
    return handle;
}

__global__ void elementwiseMul(float *x, float *y, float *z, int rows, int cols) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= cols || i >= rows) return;
    z[i * cols + j] = x[i * cols + j] * y[i * cols + j];
}

void matrixElementWiseMul(cuMatrix<float> *x, cuMatrix<float> *y, cuMatrix<float> *z) {
    if (x->cols != y->cols || z->cols != x->cols || x->rows != y->rows || x->rows != z->rows) {
        printf("matrix elementwise multiply invalid dim\n");
        exit(0);
    }
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((x->cols + blockDim.x - 1) / blockDim.x,
                 (x->rows + blockDim.y - 1) / blockDim.y);
    elementwiseMul << < blockDim, gridDim >> > (x->getDev(), y->getDev(), z->getDev(), x->rows, x->cols);

}

/*Matrix Concatenation*/
/*z = [x;y]*/
void matrixConcat(cuMatrix<float> *x, cuMatrix<float> *y, cuMatrix<float> *z) {
    if (x->cols != y->cols || z->cols != x->cols || z->rows != x->rows + y->rows) {
        printf("matrix concat invalid dim\n");
        exit(0);
    }

    float *res = z->getDev();
    cudaMemcpy(res, x->getDev(), x->rows * x->cols * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(&res[x->rows * x->cols], y->getDev(), y->rows * y->cols * sizeof(float), cudaMemcpyDeviceToDevice);
}

/*Matrix Split*/
/*y = x[1:row][:] z = x[row:end][:]*/
void matrixSplit(cuMatrix<float> *x, cuMatrix<float> *y, cuMatrix<float> *z) {
    if (x->cols != y->cols || x->cols != z->cols || x->rows != y->rows + z->rows) {
        printf("matrix split invalid dim\n");
        exit(0);
    }

    cudaMemcpy(y->getDev(), x->getDev(), y->rows * y->cols * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(z->getDev(), &(x->getDev())[y->rows * y->cols], z->rows * z->cols * sizeof(float),
               cudaMemcpyDeviceToDevice);
}

/*matrix transpose*/
/*x = T(x)*/
double matrixTranspose(cuMatrix<float> *x) {
    float alpha = 1.0;
    float beta = 0.0;
    float *y;
    cublasHandle_t handle = getHandle();
    double overallStartTime = CycleTimer::currentSeconds();
    cudaMalloc(&y, x->rows * x->cols * sizeof(float));
    cudaMemcpy(y, x->getDev(), x->rows * x->cols * sizeof(float), cudaMemcpyDeviceToDevice);


    cublasSgeam(handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                x->rows, x->cols,
                &alpha,
                y, x->cols,
                &beta,
                NULL, x->rows,
                x->getDev(), x->rows);

    int temp_r = x->rows;
    x->rows = x->cols;
    x->cols = temp_r;
    cudaFree(y);
    double overallEndTime = CycleTimer::currentSeconds();
    return overallEndTime - overallStartTime;
}

__global__ void matrixTransKernel(float *A, int rows, int cols) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= cols || i >= rows) return;
    float tmp = A[i * cols + j];
    A[i * cols + j] = A[j * cols + i];
    A[j * cols + i] = tmp;
}

double matrixTranspose2(cuMatrix<float> *x) {
    double overallStartTime = CycleTimer::currentSeconds();
    dim3 blockDim(16, 16);
    dim3 gridDim((x->cols + blockDim.x - 1) / blockDim.x,
                 (x->rows + blockDim.y - 1) / blockDim.y);

    matrixTransKernel << < blockDim, gridDim >> > (x->getDev(), x->rows,x->cols);
    cudaThreadSynchronize();
    int temp_r = x->rows;
    x->rows = x->cols;
    x->cols = temp_r;
    double overallEndTime = CycleTimer::currentSeconds();
    return overallEndTime - overallStartTime;
}

__global__ void matrixSubKernel(float *A, float *B, float *C, float lambda, int N) {
    int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    int COL = blockIdx.x * blockDim.x + threadIdx.x;

    if (ROW < N && COL < N) {
        C[ROW * N + COL] = A[ROW * N + COL] + lambda * B[ROW * N + COL];
    }
}

double matrixSub2(cuMatrix<float> *x, cuMatrix<float> *y, cuMatrix<float> *z, float lambda) {
    double overallStartTime = CycleTimer::currentSeconds();
    lambda = -lambda;
    dim3 blockDim(16, 16);
    int N = x->rows;
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);

    matrixSubKernel << < blockDim, gridDim >> > (x->getDev(), y->getDev(), z->getDev(), lambda, N);
    cudaThreadSynchronize();
    double overallEndTime = CycleTimer::currentSeconds();
    return overallEndTime - overallStartTime;
}

double matrixSub(cuMatrix<float> *x, cuMatrix<float> *y, cuMatrix<float> *z, float lambda) {
    lambda = -lambda;
    float alpha = 1.0;
    cublasStatus_t stat;
    cublasHandle_t handle = getHandle();
    double overallStartTime = CycleTimer::currentSeconds();
    stat = cublasSgeam(handle,
                       CUBLAS_OP_N,
                       CUBLAS_OP_N,
                       x->cols, y->rows,
                       &alpha,
                       x->getDev(), x->cols,
                       &lambda,
                       y->getDev(), y->cols,
                       z->getDev(), z->cols);
    double overallEndTime = CycleTimer::currentSeconds();
    return overallEndTime - overallStartTime;
    //cudaStreamSynchronize(0);
//    getLastCudaError("matrixSub");
//    if (stat != CUBLAS_STATUS_SUCCESS) {
//        printf("matrixSub cublasSgemm error\n");
//        cudaFree(x->getDev());
//        cudaFree(y->getDev());
//        cudaFree(z->getDev());
//        exit(0);
//    }
}

/*matrix multiply*/
/*z = x * y*/
double matrixMul(cuMatrix<float> *x, cuMatrix<float> *y, cuMatrix<float> *z) {
    if (x->cols != y->rows || z->rows != x->rows || z->cols != y->cols) {
        printf("matrix mul chanels != 1\n");
        exit(0);
    }
    float alpha = 1.0;
    float beta = 0.0;
    cublasStatus_t stat;
    cublasHandle_t handle = getHandle();
    double overallStartTime = CycleTimer::currentSeconds();
    stat = cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            y->cols,
            x->rows,
            y->rows,
            &alpha,
            y->getDev(),
            y->cols,
            x->getDev(),
            x->cols,
            &beta,
            z->getDev(),
            z->cols);
    double overallEndTime = CycleTimer::currentSeconds();
    return overallEndTime - overallStartTime;
    //cudaStreamSynchronize(0);
    //getLastCudaError("matrixMul");
//    if (stat != CUBLAS_STATUS_SUCCESS) {
//        printf("matrixMul cublasSgemm error\n");
//        cudaFree(x->getDev());
//        cudaFree(y->getDev());
//        cudaFree(z->getDev());
//        exit(0);
//    }
}

__global__ void matrixMultiplicationKernel(float *A, float *B, float *C, int N) {

    int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    int COL = blockIdx.x * blockDim.x + threadIdx.x;

    float tmpSum = 0;

    if (ROW < N && COL < N) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < N; i++) {
            tmpSum += A[ROW * N + i] * B[i * N + COL];
        }
    }
    C[ROW * N + COL] = tmpSum;
}

double matrixMul2(cuMatrix<float> *x, cuMatrix<float> *y, cuMatrix<float> *z) {
    double overallStartTime = CycleTimer::currentSeconds();
    int N = x->rows;
    //printf("%d\n", N);
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);

    matrixMultiplicationKernel << < blockDim, gridDim >> > (x->getDev(), y->getDev(), z->getDev(), N);
    cudaThreadSynchronize();
    double overallEndTime = CycleTimer::currentSeconds();
    return overallEndTime - overallStartTime;
}


/*z = T(x) * y*/
void matrixMulTA(cuMatrix<float> *x, cuMatrix<float> *y, cuMatrix<float> *z) {
    if (x->rows != y->rows || z->rows != x->cols || z->cols != y->cols) {
        printf("matrix mul chanels != 1\n");
        exit(0);
    }
    cublasStatus_t stat;
    float alpha = 1.0;
    float beta = 0.0;
    stat = cublasSgemm(
            getHandle(),
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            y->cols,
            x->cols,
            y->rows,
            &alpha,
            y->getDev(),
            y->cols,
            x->getDev(),
            x->cols,
            &beta,
            z->getDev(),
            z->cols);
    cudaStreamSynchronize(0);
    getLastCudaError("matrixMulTA");
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("matrixMulTA cublasSgemm error\n");
        exit(0);
    }
}

/*z = x * T(y)*/
void matrixMulTB(cuMatrix<float> *x, cuMatrix<float> *y, cuMatrix<float> *z) {
    if (x->cols != y->cols || z->rows != x->rows || z->cols != y->rows) {
        printf("matrix mul chanels != 1\n");
        exit(0);
    }
    cublasStatus_t stat;
    float alpha = 1.0;
    float beta = 0.0;
    stat = cublasSgemm(
            getHandle(),
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            y->rows,
            x->rows,
            y->cols,
            &alpha,
            y->getDev(),
            y->cols,
            x->getDev(),
            x->cols,
            &beta,
            z->getDev(),
            z->cols);
    cudaStreamSynchronize(0);
    getLastCudaError("matrixMulTB");
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("matrixMulTB cublasSgemm error\n");
        exit(0);
    }
}