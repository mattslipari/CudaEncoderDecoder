#include "Layers/FullyConnect.h"
#include "Common/cuMatrix.h"

int main() {
    float data[8];
    for(int i=0;i<4;i++){
        data[i]=i+1;
        data[i+4]=i+1;
    }
    cuMatrix<float> inputs(data,2,4);
    FullyConnect fc(&inputs, 3);
    fc.feedforward();
    fc.printParameter();
}