#ifndef CUDAENCODERDOCODER_FULLYCONNECT_H
#define CUDAENCODERDOCODER_FULLYCONNECT_H

#include "LayerBase.h"

class FullyConnect:LayerBase {
public:
    void feedforward();
    void backpropagation();
    void getGrad();
    void updateWeight();
    cuMatrix<float>* getOutputs();
    void initRandom();
    void printParameter();
private:
    cuMatrix<float>* inputs;
    cuMatrix<float>* outputs;

    cuMatrix<float>* w;

    cuMatrix<float>* b;

    int inputsize;
    int outputsize;
    float lambda;
    int batch;
};


#endif //CUDAENCODERDOCODER_FULLYCONNECT_H