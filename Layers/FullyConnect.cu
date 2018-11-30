#include "../Common/cuMatrix.h"
#include "FullyConnect.h"
#include <cstdlib>

__global__ void relu(float *inout, float *bias, int rows, int cols) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= cols || i >= rows) return;
    inout[i * cols + j] = fmaxf(0.0, inout[i * cols + j] + bias[j]);
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
        this->b->set(i, 0, 1);
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
    relu <<< blockDim, gridDim >>> (outputs->getDev(), this->b->getDev(), this->units, this->batch);
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

void FullyConnect::backpropagation() {

}

void FullyConnect::getGrad() {

}

void FullyConnect::updateWeight() {

}

void FullyConnect::clearGrad() {

}




