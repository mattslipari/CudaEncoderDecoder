#ifndef CUDAENCODERDOCODER_ENCODERDECODER_H
#define CUDAENCODERDOCODER_ENCODERDECODER_H


#include "../Common/cuMatrix.h"
#include "FullyConnect.h"
#include "LSTM.h"

class EncoderDecoder {
public:
    EncoderDecoder(cuMatrix<float> **inputs, cuMatrix<float> *pre_hidden,
                   cuMatrix<float> *pre_cell, int input_total, int units, float lambda) {
        encoder = new LSTM(inputs, pre_hidden, pre_cell, input_total, units, lambda, false);
        decoder = new LSTM(inputs, pre_hidden, pre_cell, input_total, units, lambda, true);
    }

    ~EncoderDecoder() {

    }

    void forward();

    cuMatrix<float> *getGrad();

    void updateWeight();

    void printParameter();

private:
    LSTM *encoder;
    LSTM *decoder;
};


#endif //CUDAENCODERDOCODER_ENCODERDECODER_H
