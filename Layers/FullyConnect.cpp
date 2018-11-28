#include "FullyConnect.h"
#include <cstdlib>

__global__ void relu(float* inout, float* bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows) return;

    float _bias = bias[idx];
    float* _inout = &inout[idx];

    for(int j = 0; j < cols; j++) {
    	inout[idx][j] = fmaxf(0.0, _inout[j] + _bias);
    }
}

void FullyConnect::FullyConnect(cuMatrix<float> *inputs, int units) {
		this->units = units;
		this->inputs = inputs;

		this->initRandom();
}

void FullyConnect::initRandom() {
		this->w = cuMatrix(float, this->inputs->units, this->inputs->rows);
		this->b = cuMatrix(float, this->inputs->units, 1);

		this->w->setAllRandom(-1,1);
		this->b->setAllRandom(-1,1);
}

cuMatrix<float> *FullyConnect::feedforward() {
    
}
