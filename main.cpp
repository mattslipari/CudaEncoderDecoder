#include "Layers/FullyConnect.h"
#include "Layers/LSTM.h"
#include "Common/cuMatrix.h"
#include "Common/CycleTimer.h"

int main() {

    int total_inputs = 100;    
    int units = 140;
    int lambda = 1.0;
    int input_cols = 1000;
    int input_batch = 200; //real batch is: input_batch + units

    cuMatrix<float> pre_hidden(units, input_cols); //same # of cols as inputs
    cuMatrix<float> pre_cell(units, input_cols); //same # of cols as inputs

    cuMatrix<float> *input_list[total_inputs];
    for (int i = 0; i < total_inputs; i++) {
        cuMatrix<float> inputs(input_batch, input_cols);
        input_list[i] = &inputs;
    }

    //Start our computation
    double overallStartTime = CycleTimer::currentSeconds();

    LSTM ls(input_list, &pre_hidden, &pre_cell, total_inputs, units, lambda);

    double initEndTime = CycleTimer::currentSeconds();    
    double forwardStartTime = CycleTimer::currentSeconds();

    ls.forward();

    double forwardEndTime = CycleTimer::currentSeconds();
    double backStartTime = CycleTimer::currentSeconds();

    ls.backpropagation(&pre_cell,input_list);

    double overallEndTime = CycleTimer::currentSeconds(); 

    printf("Initialization Time Elapsed: %.3f ms\n", initEndTime - overallStartTime);
    printf("Forward Time Elapsed: %.3f ms\n", forwardEndTime - forwardStartTime);
    printf("Backpropagation Time Elapsed: %.3f ms\n", overallEndTime - backStartTime);
    printf("Overall Time Elapsed: %.3f ms\n", overallEndTime - overallStartTime);
}