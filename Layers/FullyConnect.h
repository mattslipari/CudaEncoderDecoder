#ifndef CUDAENCODERDOCODER_FULLYCONNECT_H
#define CUDAENCODERDOCODER_FULLYCONNECT_H

#include "LayerBase.h"

class FullyConnect : public LayerBase {
public:

    FullyConnect(cuMatrix<float> *inputs, int units) {
        this->units = units;
        this->inputs = inputs;
        this->batch = inputs->cols;
        this->outputs = new cuMatrix<float>(units, batch);
        this->outputs->cpuClear();

        this->initRandom();
    }

    ~FullyConnect() {

    }

    void feedforward();

    void backpropagation();

    void getGrad();

    void updateWeight();

    void clearGrad();

    cuMatrix<float> *getOutputs();

    void initRandom();

    void printParameter();

private:
    cuMatrix<float> *inputs;
    cuMatrix<float> *outputs;

    cuMatrix<float> *w;

    cuMatrix<float> *b;
    int units;

    float lambda;
    int batch;
};


#endif //CUDAENCODERDOCODER_FULLYCONNECT_H
