#include "Layers/FullyConnect.h"
#include "Common/cuMatrix.h"

int main() {
    float data[6];
    for(int i=0;i<3;i++){
        data[i]=i+1;
        data[i+3]=i+1;
    }
    cuMatrix<float> inputs(data,2,3);
    FullyConnect fc(&inputs, 3);
    cuMatrix<float> *outputs=fc.getOutputs();
    outputs->toCpu();
    outputs->printHost();
}