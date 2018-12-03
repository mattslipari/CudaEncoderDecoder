#ifndef CUDAENCODERDOCODER_FULLYCONNECT_H
#define CUDAENCODERDOCODER_FULLYCONNECT_H

#include "LayerBase.h"

class FullyConnect : public LayerBase {
public:

    FullyConnect(cuMatrix<float> *inputs, int units, float lambda) {
        this->units = units;
        this->inputs = inputs;
        this->batch = inputs->cols;
        this->lambda = lambda;
        this->outputs = new cuMatrix<float>(units, this->batch);

        this->inputs_grad = new cuMatrix<float>(inputs->rows, inputs->cols);
        this->w_grad = new cuMatrix<float>(units, inputs->rows);
        this->b_grad = new cuMatrix<float>(units, 1);

        this->outputs->cpuClear();

        this->initRandom();
    }

    ~FullyConnect() {

    }

    void feedforward();

    void backpropagation(cuMatrix<float> *pre_grad);

    void getGrad();

    void updateWeight();

    void clearGrad();

    cuMatrix<float> *getOutputs();

    void initRandom();

    void printParameter();

private:
    cuMatrix<float> *inputs;
    cuMatrix<float> *outputs; // units x batch
    cuMatrix<float> *w_grad;
    cuMatrix<float> *b_grad;
    cuMatrix<float> *inputs_grad; // units x batch
    cuMatrix<float> *w; // units x input_row
    cuMatrix<float> *b; // units x 1
    int units;

    float lambda;
    int batch;
};


#endif //CUDAENCODERDOCODER_FULLYCONNECT_H
