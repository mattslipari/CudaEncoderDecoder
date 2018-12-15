#include "FullyConnect.h"
#include "LSTM.h"
#include "../Common/cuMatrix.h"
#include <algorithm>

__global__ void attentionKernel(float *x, int rows, int cols) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j >= cols) return;
    float sum = 0;
    for (int k = 0; k < rows; k++) {
        sum += x[k * cols + j];
    }
    for (int k = 0; k < rows; k++) {
        x[k * cols + j] *= sum;
    }
}

void LSTM::forward(cuMatrix<float> **encoder_hidden) {
    cuMatrix<float> *input_t;
    cuMatrix<float> *input_hidden = new cuMatrix<float>(total_rows, this->input_cols);
    cuMatrix<float> *cell_t = this->pre_cell;
    cuMatrix<float> *ia = new cuMatrix<float>(this->units, this->input_cols);
    cuMatrix<float> *fc = new cuMatrix<float>(this->units, this->input_cols);
    cuMatrix<float> *blank_bias = new cuMatrix<float>(cell_t->cols, 1);
    cuMatrix<float> *concat_tmp = new cuMatrix<float>(pre_hidden->rows + this->input_rows, this->input_cols);

    dim3 blockDim(16, 16, 1);
    dim3 gridDim((cell_t->cols + blockDim.x - 1) / blockDim.x,
                 (cell_t->rows + blockDim.y - 1) / blockDim.y);

    for (int t = 0; t < std::min(this->input_total, MAXTIMESTEP); t++) {
        input_t = this->inputs[t];
        if (useAttention) {
            matrixConcat(input_t, this->pre_hidden, concat_tmp);
            cuMatrix<float> *ct = attention(encoder_hidden);
            matrixConcat(concat_tmp, ct, input_hidden);
        } else
            matrixConcat(input_t, this->pre_hidden, input_hidden);

        this->a_layer->forward(input_hidden);
        this->i_layer->forward(input_hidden);
        this->f_layer->forward(input_hidden);
        this->o_layer->forward(input_hidden);
        at[t] = new cuMatrix<float>(a_layer->outputs->rows, a_layer->outputs->cols);
        at[t]->copyFromGpu(a_layer->outputs->getDev());
        it[t] = new cuMatrix<float>(i_layer->outputs->rows, i_layer->outputs->cols);
        it[t]->copyFromGpu(i_layer->outputs->getDev());
        ft[t] = new cuMatrix<float>(f_layer->outputs->rows, f_layer->outputs->cols);
        ft[t]->copyFromGpu(f_layer->outputs->getDev());
        ot[t] = new cuMatrix<float>(o_layer->outputs->rows, o_layer->outputs->cols);
        ot[t]->copyFromGpu(o_layer->outputs->getDev());
        ht[t] = new cuMatrix<float>(pre_hidden->rows, pre_hidden->cols);
        ot[t]->copyFromGpu(pre_hidden->getDev());
        matrixElementWiseMul(this->i_layer->outputs, this->a_layer->outputs, ia);
        matrixElementWiseMul(this->f_layer->outputs, cell_t, fc);
        matrixSub(ia, fc, cell_t, -1);
        ct[t] = new cuMatrix<float>(cell_t->rows, cell_t->cols);
        ct[t]->copyFromGpu(cell_t->getDev());
        tanh << < blockDim, gridDim >> > (cell_t->getDev(), blank_bias->getDev(), cell_t->rows, cell_t->cols);
        cudaThreadSynchronize();

        tanh_ct[t] = new cuMatrix<float>(cell_t->rows, cell_t->cols);
        tanh_ct[t]->copyFromGpu(cell_t->getDev());
        matrixElementWiseMul(this->o_layer->outputs, cell_t, this->pre_hidden);
    }
}

cuMatrix<float> *LSTM::attention(cuMatrix<float> **encoder_hidden) {
    dim3 blockDim(256);
    dim3 gridDim((pre_hidden->cols + blockDim.x - 1) / blockDim.x);
    cuMatrix<float> *ct = new cuMatrix<float>(pre_hidden->rows, pre_hidden->cols);
    cuMatrix<float> tmpt(pre_hidden->rows, pre_hidden->cols);
    for (int t = 0; t < std::min(input_total, MAXTIMESTEP); t++) {
        matrixElementWiseMul(encoder_hidden[t], pre_hidden, &tmpt);
        attentionKernel << < blockDim, gridDim >> > (tmpt.getDev(), tmpt.rows, tmpt.cols);
        matrixSub(ct, &tmpt, ct, -1);//addition
    }
    return ct;
}

void LSTM::backpropagation(cuMatrix<float> *pre_grad, cuMatrix<float> **inputs) {
    dim3 blockDim_r(16, 16, 1);
    dim3 gridDim_r((pre_cell->cols + blockDim_r.x - 1) / blockDim_r.x,
                   (pre_cell->rows + blockDim_r.y - 1) / blockDim_r.y);
    cuMatrix<float> *input_hidden = new cuMatrix<float>(total_rows, this->input_cols);
    cuMatrix<float> x_grad(input_rows, input_cols);
    cuMatrix<float> ht_grad(units, input_cols);

    cuMatrix<float> i_grad(units, input_cols);
    cuMatrix<float> a_grad(units, input_cols);
    cuMatrix<float> f_grad(units, input_cols);
    cuMatrix<float> o_grad(units, input_cols);
    cuMatrix<float> c_grad(units, input_cols);
    cuMatrix<float> o_weights_grad(units, pre_hidden->rows + input_rows);
    cuMatrix<float> a_weights_grad(units, pre_hidden->rows + input_rows);
    cuMatrix<float> f_weights_grad(units, pre_hidden->rows + input_rows);
    cuMatrix<float> i_weights_grad(units, pre_hidden->rows + input_rows);

    for (int t = std::min(input_total, MAXTIMESTEP) - 1; t >= 0; t--) {
        cuMatrix<float> *input_t = inputs[t];
        matrixConcat(input_t, ht[t], input_hidden);
        matrixElementWiseMul(pre_grad, tanh_ct[t], &o_grad);// ot gradient
        o_layer->backpropagation(&o_grad, input_hidden);
        matrixSub(&o_weights_grad, o_layer->w_grad, &o_weights_grad, -1); //  weights addition
        tanh_grad << < blockDim_r, gridDim_r >> >
                                   (pre_grad->getDev(), tanh_ct[t]->getDev(), tanh_ct[t]->rows, tanh_ct[t]->cols);
        cudaThreadSynchronize();

        matrixElementWiseMul(pre_grad, ot[t], pre_grad);
        matrixSub(&c_grad, pre_grad, &c_grad, -1);// ct gradient
        matrixElementWiseMul(&c_grad, at[t], &i_grad);// it gradient
        matrixElementWiseMul(&c_grad, it[t], &a_grad);//at gradient

        if (t - 1 < 0)
            matrixElementWiseMul(&c_grad, pre_cell, &f_grad);
        else
            matrixElementWiseMul(&c_grad, ct[t - 1], &f_grad);//ft gradient

        i_layer->backpropagation(&i_grad, input_hidden);
        matrixSub(&i_weights_grad, i_layer->w_grad, &i_weights_grad, -1); //  weights addition
        f_layer->backpropagation(&f_grad, input_hidden);
        matrixSub(&f_weights_grad, f_layer->w_grad, &f_weights_grad, -1); //  weights addition
        a_layer->backpropagation(&a_grad, input_hidden);
        matrixSub(&a_weights_grad, a_layer->w_grad, &a_weights_grad, -1); //  weights addition

        pre_grad->cpuClear();
        pre_grad->gpuClear();

        matrixSplit(i_layer->inputs_grad, &x_grad, &ht_grad);
        matrixSub(pre_grad, &ht_grad, pre_grad, -1);
        matrixSplit(a_layer->inputs_grad, &x_grad, &ht_grad);
        matrixSub(pre_grad, &ht_grad, pre_grad, -1);
        matrixSplit(f_layer->inputs_grad, &x_grad, &ht_grad);
        matrixSub(pre_grad, &ht_grad, pre_grad, -1);
        matrixSplit(o_layer->inputs_grad, &x_grad, &ht_grad);
        matrixSub(pre_grad, &ht_grad, pre_grad, -1);
    }
    // pre_grad->printHost();
}

cuMatrix<float> *LSTM::getGrad() {
    return this->pre_hidden;
}

void LSTM::updateWeight() {

}

void LSTM::printParameter() {

}