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
    printf("Starting...\n");
    cuMatrix<float> **input_list;
    *input_list = &inputs;
    printf("Past this stage\n");
    LSTM ls(input_list, &pre_hidden, &pre_cell, 1, 3, 1.0);
}