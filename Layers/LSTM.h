#ifndef CUDAENCODERDOCODER_LSTM_H
#define CUDAENCODERDOCODER_LSTM_H


#include "LayerBase.h"

class LSTM : public LayerBase {
    void feedforward();

    void backpropagation(cuMatrix<float> *pre_grad);

    cuMatrix<float> *getGrad();

    void updateWeight();

    cuMatrix<float> *getOutputs();

    void initRandom();

    void printParameter();
};


#endif //CUDAENCODERDOCODER_LSTM_H
