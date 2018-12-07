#ifndef CUDAENCODERDOCODER_LSTM_H
#define CUDAENCODERDOCODER_LSTM_H


#include "LayerBase.h"

class LSTM : public LayerBase {


    void forward();

    void backpropagation(cuMatrix<float> *pre_grad);

    cuMatrix<float> *getGrad();

    void updateWeight();

    cuMatrix<float> *getOutputs();

    void initRandom();

    void printParameter();

private:

};


#endif //CUDAENCODERDOCODER_LSTM_H
