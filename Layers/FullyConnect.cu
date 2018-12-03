#include "../Common/cuMatrix.h"
#include "FullyConnect.h"
#include <cstdlib>

__global__ void relu(float *inout, float *bias, int rows, int cols) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= cols || i >= rows) return;
    inout[i * cols + j] = fmaxf(0.0, inout[i * cols + j] + bias[i]);
}

__global__ void relu_grad(float *pre_grad, float *output, int rows, int cols) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= cols || i >= rows) return;
    if (output[i * cols + j] <= 0)
        pre_grad[i * cols + j] = 0;
}

__global__ void bias_grad(float *pre_grad, float *output, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    output[i] = 0;
    for (int k = 0; k < cols; k++) {
        output[i] = pre_grad[i * cols + k];
    }
}

void FullyConnect::initRandom() {
    this->w = new cuMatrix<float>(this->units, this->inputs->rows);
    this->b = new cuMatrix<float>(this->units, 1);

    // this->w->setAllRandom(-1, 1);
    // this->b->setAllRandom(-1, 1);

    for (int j = 0; j < this->inputs->rows; j++) {
        for (int i = 0; i < this->units; i++) {
            this->w->set(i, j, j + 1);
        }
    }

    for (int i = 0; i < this->units; i++) {
        this->b->set(i, 0, i);
    }
}

void FullyConnect::feedforward() {
    this->w->toGpu();
    this->b->toGpu();
    this->inputs->toGpu();
    this->outputs->toGpu();
    matrixMul(this->w, this->inputs, this->outputs);

    dim3 blockDim(16, 16, 1);
    dim3 gridDim((this->outputs->cols + blockDim.x - 1) / blockDim.x,
                 (this->outputs->rows + blockDim.y - 1) / blockDim.y);
    relu << < blockDim, gridDim >> > (outputs->getDev(), this->b->getDev(), this->units, this->batch);
}

cuMatrix<float> *FullyConnect::getOutputs() {
    return this->outputs;
}

void FullyConnect::printParameter() {
    this->w->toCpu();
    this->b->toCpu();
    printf("weights:\n");
    this->w->printHost();
    printf("bias:\n");
    this->b->printHost();
    this->inputs->toCpu();
    printf("inputs:\n");
    this->inputs->printHost();
    this->outputs->toCpu();
    printf("outputs:\n");
    this->outputs->printHost();
}

void FullyConnect::backpropagation(cuMatrix<float> *pre_grad) {
    dim3 blockDim_r(16, 16, 1);
    dim3 gridDim_r((outputs->cols + blockDim.x - 1) / blockDim.x,
                 (outputs->rows + blockDim.y - 1) / blockDim.y);
    relu_grad << < blockDim_r, gridDim_r >> > (pre_grad->getDev(), outputs->getDev(), outputs->rows, outputs->cols);
    
    dim3 blockDim_b(256);
    dim3 gridDim_b((b->rows + blockDim.x - 1) / blockDim.x);
    bias_grad << < blockDim_b, gridDim_b >> > (pre_grad->getDev(), b_grad->getDev(), pre_grad->rows, pre_grad->cols);
    
    matrixMulTA(pre_grad, w, inputs_grad);
    matrixTranspose(inputs_grad);
    matrixMulTB(pre_grad, inputs, w_grad);
}

void FullyConnect::getGrad() {

}

void FullyConnect::updateWeight() {

}

void FullyConnect::clearGrad() {

}




