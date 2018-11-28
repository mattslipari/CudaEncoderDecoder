#include "FullyConnect.h"
#include "../Common/cuMatrix.h"


FullyConnect::FullyConnect(cuMatrix<float> *inputs, int units) {
    this->units = units;
    this->inputs = inputs;
    this->batch=inputs->cols;

    this->initRandom();
}

void FullyConnect::initRandom() {
    this->w = new cuMatrix<float>(this->units, this->inputs->rows);
    this->b = new cuMatrix<float>(this->units, 1);

    this->w->setAllRandom(-1, 1);
    this->b->setAllRandom(-1, 1);
}

__global__ void relu(float *inout, float *bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.y;
    if (idx >= rows) return;
    float _bias = bias[idx];
    for (int j = 0; j < cols; j++) {
        inout[idx][j] = fmaxf(0.0, inout[idx][j] + _bias);
    }
}

void FullyConnect::feedforward() {
    matrixMul(this->w, this->inputs, this->outputs);
    dim3 blockDim(16, 16);
    relu(outputs, this->b, this->units, this->batch);
}

cuMatrix<float> *FullyConnect::getOutputs(){
    return this->outputs;
}
