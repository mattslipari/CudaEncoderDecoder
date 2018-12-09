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
        this->outputs = new cuMatrix<float>(units, this->batch);

        this->inputs_grad = new cuMatrix<float>(input_rows, input_cols);
        this->w_grad = new cuMatrix<float>(units, input_rows);
        this->b_grad = new cuMatrix<float>(units, 1);

        this->outputs->cpuClear();

        this->initRandom();
    }

    ~FullyConnect() {

    }

    void forward();

    void backpropagation(cuMatrix<float> *pre_grad);

    cuMatrix<float> *getGrad();

    void updateWeight();

    cuMatrix<float> *getOutputs();

    void initRandom();

    void printParameter();

    cuMatrix<float> *getWeightsGrad();

private:
    cuMatrix<float> *inputs;
    cuMatrix<float> *outputs; // units x batch
    cuMatrix<float> *w_grad;
    cuMatrix<float> *b_grad;
    cuMatrix<float> *inputs_grad; // units x batch
    cuMatrix<float> *w; // units x input_row
    cuMatrix<float> *b; // units x 1
    int units;

    Activation type;
    float lambda;
    int batch;
};


#endif //CUDAENCODERDOCODER_FULLYCONNECT_H
