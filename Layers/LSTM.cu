#include "LSTM.h"


void LSTM::backpropagation(cuMatrix<float> *pre_grad) {
    dim3 blockDim_r(16, 16, 1);
    dim3 gridDim_r((ct->cols + blockDim_r.x - 1) / blockDim_r.x,
                   (ct->rows + blockDim_r.y - 1) / blockDim_r.y);
    cuMatrix<float> i_grad();
    cuMatrix<float> a_grad();
    cuMatrix<float> f_grad();
    cuMatrix<float> c_grad();
    cuMatrix<float> o_weights_grad();
    cuMatrix<float> a_weights_grad();
    cuMatrix<float> f_weights_grad();
    cuMatrix<float> i_weights_grad();

    for (int t = 0; t < T; t++) {
        matrixElementWiseMul(pre_grad, tanh_ct, pre_grad);// ot gradient
        o_layer->backpropagation(pre_grad);
        matrixSub(o_weights_grad,o_layer->getWeightsGrad(),-1); //  weights addition
        tanh_grad << < blockDim_r, gridDim_r >> > (pre_grad->getDev(), tanh_ct->getDev(), tanh_ct->rows, tanh_ct->cols);//ct gradient
        matrixElementWiseMul(pre_grad->getDev(),ot,c_grad,pre_grad->rows,pre_grad->cols);
        matrixElementWiseMul(pre_grad,ot,i_grad);
        matrixElementWiseMul(pre_grad,it,a_grad);
        matrixElementWiseMul(pre_grad,ct-1,f_grad);
        i_layer->backpropagation(i_grad);
        matrixSub(i_weights_grad,i_layer->getWeightsGrad(),-1); //  weights addition
        f_layer->backpropagation(f_grad);
        matrixSub(f_weights_grad,f_layer->getWeightsGrad(),-1); //  weights addition
        a_layer->backpropagation(a_grad);
        matrixSub(a_weights_grad,a_layer->getWeightsGrad(),-1); //  weights addition
        matrixSub(c_grad,pre_grad,)

        matrixElementWiseMul(pre_grad,ft,c_grad);
    }
}