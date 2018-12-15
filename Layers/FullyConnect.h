/*
Modified from
https://github.com/zhxfl/CUDA-CNN
*/

#ifndef CUDAENCODERDOCODER_FULLYCONNECT_H
#define CUDAENCODERDOCODER_FULLYCONNECT_H

#include "LayerBase.h"

class FullyConnect : public LayerBase {
public:

    enum Activation {
        RELU,
        SIGMOID,
        TANH
    };

    FullyConnect(int input_rows, int input_cols, int units, float lambda, Activation type) {
        this->units = units;
        this->batch = input_rows;
        this->lambda = lambda;
        this->type = type;
        this->outputs = new cuMatrix<float>(units, input_cols);

        this->inputs_grad = new cuMatrix<float>(input_rows, input_cols);
        this->w_grad = new cuMatrix<float>(units, input_rows);
        this->b_grad = new cuMatrix<float>(units, 1);

        this->outputs->cpuClear();

        this->initRandom();
    }

    ~FullyConnect() {

    }

    void forward(cuMatrix<float> *inputs);

    void backpropagation(cuMatrix<float> *pre_grad, cuMatrix<float> *inputs);

    cuMatrix<float> *getGrad();

    void updateWeight();

    void initRandom();

    void printParameter();

    cuMatrix<float> *w_grad;

    cuMatrix<float> *outputs; // units x batch
    cuMatrix<float> *inputs_grad; // units x batch
    int units;

private:
    cuMatrix<float> *inputs;
    cuMatrix<float> *b_grad;

    cuMatrix<float> *w; // units x input_row
    cuMatrix<float> *b; // units x 1

    Activation type;
    float lambda;
    int batch;
};

__global__ void tanh(float *inout, float *bias, int rows, int cols);
__global__ void tanh_grad(float *pre_grad, float *output, int rows, int cols);

#endif //CUDAENCODERDOCODER_FULLYCONNECT_H
