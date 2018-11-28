#ifndef CUDAENCODERDOCODER_FULLYCONNECT_H
#define CUDAENCODERDOCODER_FULLYCONNECT_H

#include "LayerBase.h"

class FullyConnect : LayerBase {
public:

    FullyConnect(cuMatrix<float> *inputs, int units);

    void feedforward();

    void backpropagation();

    void getGrad();

    void updateWeight();

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
    int units;
};


#endif //CUDAENCODERDOCODER_FULLYCONNECT_H
