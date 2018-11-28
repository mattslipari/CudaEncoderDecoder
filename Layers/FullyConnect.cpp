#include "FullyConnect.h"
#include "../Common/cuMatrix.h"


FullyConnect::FullyConnect(cuMatrix<float> *inputs, int units) {
    this->units = units;
    this->inputs = inputs;
    this->batch = inputs->cols;

    this->initRandom();
}

void FullyConnect::initRandom() {
    this->w = new cuMatrix<float>(this->units, this->inputs->rows);
    this->b = new cuMatrix<float>(this->units, 1);

    this->w->setAllRandom(-1, 1);
    this->b->setAllRandom(-1, 1);
}

__global__ void relu(cuMatrix<float>* inout, cuMatrix<float>* bias, int rows, int cols) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= cols || i >= rows) return;
    inout.set(i, j, fmaxf(0.0, inout.get(i, j) + bias.get(i, 1)));
}

void FullyConnect::feedforward() {
    matrixMul(this->w, this->inputs, this->outputs);
    dim3 blockDim(16,16,1);
    dim3 gridDim((this->outputs->cols+blockDim.x-1)/blockDim.x, (this->outputs->rows+blockDim.y-1)/blockDim.y);
    relu<<<blockDim, gridDim>>>(outputs, this->b, this->units, this->batch);
}
cuMatrix<float> *FullyConnect::getOutputs() {
    return this->outputs;
}

void FullyConnect::backpropagation() {

}


