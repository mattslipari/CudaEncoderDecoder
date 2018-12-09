#include "Layers/FullyConnect.h"
#include "Layers/LSTM.h"
#include "Common/cuMatrix.h"

int main() {
    float data[8];
    for (int i = 0; i < 4; i++) {
        data[i] = i + 1;
        data[i + 4] = i + 1;
    }
    float grad[12];
    for (int i = 0; i < 12; i++) {
        grad[i] = i;
    }

    cuMatrix<float> inputs(data, 2, 4);
    cuMatrix<float> pre_hidden(grad, 3, 4);
    cuMatrix<float> pre_cell(grad, 3, 4);

    cuMatrix<float> *input_list[1];
    input_list[0] = &inputs;
    LSTM ls(input_list, &pre_hidden, &pre_cell, 1, 3, 1.0);
    ls.forward();
    ls.backpropagation(&pre_cell,input_list);
}