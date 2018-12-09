#include "FullyConnect.h"
#include "LSTM.h"
#include "../Common/cuMatrix.h"

void LSTM::forward() {
    cuMatrix<float> *input_t;
    cuMatrix<float> *input_hidden = new cuMatrix<float>(pre_hidden->rows + this->input_rows, this->input_cols);
    cuMatrix<float> *ia = new cuMatrix<float>(this->units, this->input_cols);
    cuMatrix<float> *fc = new cuMatrix<float>(this->units, this->input_cols);
    cuMatrix<float> *blank_bias = new cuMatrix<float>(cell_t->cols, 1);

    dim3 blockDim(16, 16, 1);
    dim3 gridDim((cell_t->cols + blockDim.x - 1) / blockDim.x,
                 (cell_t->rows + blockDim.y - 1) / blockDim.y);

    cuMatrix<float> *cell_t = this->pre_cell;

    for (int t = 0; t < std::min(this->input_total, t < MAXTIMESTEP); t++) {
        input_t = this->inputs[t];
        matrixConcat(input_t, this->pre_hidden, input_hidden);

        this->a_layer->forward(input_hidden);
        this->i_layer->forward(input_hidden);
        this->f_layer->forward(input_hidden);
        this->o_layer->forward(input_hidden);
        at[t] = cuMatrix<float>(a_layer->outputs->getHost(), a_layer->outputs->rows, a_layer->outputs->cols);
        it[t] = cuMatrix<float>(i_layer->outputs->getHost(), i_layer->outputs->rows, i_layer->outputs->cols);
        ft[t] = cuMatrix<float>(f_layer->outputs->getHost(), f_layer->outputs->rows, f_layer->outputs->cols);
        ot[t] = cuMatrix<float>(o_layer->outputs->getHost(), o_layer->outputs->rows, o_layer->outputs->cols);
        ht[t] = cuMatrix<float>(pre_hidden->getHost(), pre_hidden->rows, pre_hidden->cols);

        matrixElementWiseMul(this->i_layer->outputs, this->a_layer->outputs, ia);
        matrixElementWiseMul(this->f_layer->outputs, cell_t, fc);
        matrixSub(ia, fc, cell_t, -1);
        ct[t] = cuMatrix<float>(cell_t->getHost(), cell_t->rows, cell_t->cols);

        tanh << < blockDim, gridDim >> > (cell_t->getDev(), blank_bias->getDev(), cell_t->rows, cell_t->cols);
        tanh_ct[t] = cuMatrix<float>(cell_t->getHost(), cell_t->rows, cell_t->cols);

        matrixElementWiseMul(this->o_layer->outputs, cell_t, this->pre_hidden);
    }

    this->pre_hidden->printHost();
}

void LSTM::backpropagation(cuMatrix<float> *pre_grad, cuMatrix<float> inputs) {
    dim3 blockDim_r(16, 16, 1);
    dim3 gridDim_r((ct->cols + blockDim_r.x - 1) / blockDim_r.x,
                   (ct->rows + blockDim_r.y - 1) / blockDim_r.y);
    cuMatrix<float> *input_hidden = new cuMatrix<float>(pre_hidden->rows + this->input_rows, this->input_cols);
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

    for (int t = std::min(input_total, MAXTIMESTEP); t >= 0; t++) {
        input_t = inputs[t];
        matrixConcat(input_t, this->pre_hidden, input_hidden);
        matrixElementWiseMul(pre_grad, tanh_ct[t], o_grad);// ot gradient
        o_layer->backpropagation(o_grad);
        matrixSub(o_weights_grad, o_layer->getWeightsGrad(), o_weights_grad, -1); //  weights addition
        tanh_grad << < blockDim_r, gridDim_r >> >
                                   (pre_grad->getDev(), tanh_ct[t]->getDev(), tanh_ct[t]->rows, tanh_ct[t]->cols);
        matrixElementWiseMul(pre_grad, ot[t], pre_grad, pre_grad->rows, pre_grad->cols);
        matrixSub(c_grad, pre_grad, c_grad, -1);// ct gradient
        matrixElementWiseMul(c_grad, at[t], i_grad);// it gradient
        matrixElementWiseMul(c_grad, it[t], a_grad);//at gradient
        if (t - 1 < 0)
            matrixElementWiseMul(c_grad, pre_cell, f_grad);
        else
            matrixElementWiseMul(c_grad, ct[t - 1], f_grad);//ft gradient
        i_layer->backpropagation(i_grad, input_hidden);
        matrixSub(i_weights_grad, i_layer->getWeightsGrad(), i_weights_grad, -1); //  weights addition
        f_layer->backpropagation(f_grad, input_hidden);
        matrixSub(f_weights_grad, f_layer->getWeightsGrad(), f_weights_grad, -1); //  weights addition
        a_layer->backpropagation(a_grad, input_hidden);
        matrixSub(a_weights_grad, a_layer->getWeightsGrad(), a_weights_grad, -1); //  weights addition

        pre_grad->cpuClear();
        pre_grad->gpuClear();
        matrixSplit(i_layer->inputs_grad, x_grad, ht_grad);
        matrixSub(pre_grad, ht_grad, pre_grad, -1);
        matrixSplit(a_layer->inputs_grad, x_grad, ht_grad);
        matrixSub(pre_grad, ht_grad, pre_grad, -1);
        matrixSplit(f_layer->inputs_grad, x_grad, ht_grad);
        matrixSub(pre_grad, ht_grad, pre_grad, -1);
        matrixSplit(o_layer->inputs_grad, x_grad, ht_grad);
        matrixSub(pre_grad, ht_grad, pre_grad, -1);

    }
}

cuMatrix<float> *LSTM::getGrad() {
    return this->pre_hidden;
}

void LSTM::updateWeight() {

}

void LSTM::printParameter() {

}