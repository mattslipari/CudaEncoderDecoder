#include "Layers/FullyConnect.h"
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
    cuMatrix<float> pre_grad(grad, 3, 4);
    pre_grad.toGpu();
    cuMatrix<float> inputs(data, 2, 4);
    FullyConnect fc(&inputs, 3, 0.01);
    fc.feedforward();
    fc.backpropagation(&pre_grad);
    fc.printParameter();
}