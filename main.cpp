#include "Layers/FullyConnect.h"
#include "Layers/LSTM.h"
#include "Common/cuMatrix.h"
#include "Common/CycleTimer.h"

int main() {

    int total_runs = 5;

    int total_inputs = 200;    
    int units = 100;
    float lambda = 1.0;
    int input_cols = 1000;
    int input_rows = 200; //real batch is: input_batch + units

    double totalRuntime;
    double forwardRuntime;
    double backRuntime;
    for (int i = 0; i < total_runs; i++) {

        cuMatrix<float> pre_hidden(units, input_cols); //same # of cols as inputs
        cuMatrix<float> pre_cell(units, input_cols); //same # of cols as inputs
        pre_hidden.setAllRandom(-1.0, 1.0);
        pre_cell.setAllRandom(-1.0, 1.0);

        cuMatrix<float> *input_list[total_inputs];
        for (int i = 0; i < total_inputs; i++) {
            cuMatrix<float> inputs(input_rows, input_cols);
            inputs.setAllRandom(-1.0, 1.0);
            input_list[i] = &inputs;
        }

        //Start our computation
        double overallStartTime = CycleTimer::currentSeconds();

        LSTM ls(input_list, &pre_hidden, &pre_cell, total_inputs, units, lambda);

        double forwardStartTime = CycleTimer::currentSeconds();

        ls.forward();

        double forwardEndTime = CycleTimer::currentSeconds();
        double backStartTime = CycleTimer::currentSeconds();

        ls.backpropagation(&pre_cell,input_list);

        double overallEndTime = CycleTimer::currentSeconds(); 

        totalRuntime += overallEndTime - overallStartTime;
        forwardRuntime += forwardEndTime - forwardStartTime;
        backRuntime += overallEndTime - backStartTime;
    }


    printf("Forward Time Elapsed: %.3f s\n", forwardRuntime/total_runs);
    printf("Backpropagation Time Elapsed: %.3f s\n", backRuntime/total_runs);
    printf("Overall Time Elapsed: %.3f s\n", totalRuntime/total_runs);
}