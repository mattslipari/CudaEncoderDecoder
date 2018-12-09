#include "FullyConnect.h"
#include "LSTM.h"
#include "../Common/cuMatrix.h"

void LSTM::forward() {
	cuMatrix<float> *input_t;
	cuMatrix<float> *input_hidden =
		new cuMatrix<float>(pre_hidden->rows + this->input_rows, this->input_cols);
	cuMatrix<float> *cell_t = new cuMatrix<float>(this->units, this->input_cols);
	cuMatrix<float> *ia = new cuMatrix<float>(this->units, this->input_cols);
	cuMatrix<float> *fc = new cuMatrix<float>(this->units, this->input_cols);
	cuMatrix<float> *blank_bias = new cuMatrix<float>(cell_t->cols, 1);

	dim3 blockDim(16, 16, 1);
    dim3 gridDim((cell_t->cols + blockDim.x - 1) / blockDim.x,
                 (cell_t->rows + blockDim.y - 1) / blockDim.y);

	for (int t = 0; t < this->input_total; t++) {
        printf("Loop front\n");
		input_t = this->inputs[t];
		matrixConcat(input_t, this->pre_hidden, input_hidden);

        printf("After input+hidden concat\n");

		this->a_layer->forward(input_hidden);
		this->i_layer->forward(input_hidden);
		this->f_layer->forward(input_hidden);
		this->o_layer->forward(input_hidden);

        printf("After Feed Forward\n");

		matrixElementWiseMul(this->i_layer->outputs, this->a_layer->outputs, ia);
		matrixElementWiseMul(this->f_layer->outputs, this->pre_cell, fc);
		matrixSub(ia, fc, cell_t, -1);

        printf("After Mul Sub\n");

        tanh <<< blockDim, gridDim >>> (cell_t->getDev(), blank_bias->getDev(), cell_t->rows, cell_t->cols);
        matrixElementWiseMul(this->o_layer->outputs, cell_t, this->pre_hidden);
	}
}

void LSTM::backpropagation(cuMatrix<float> *pre_grad) {
//     dim3 blockDim_r(16, 16, 1);
//     dim3 gridDim_r((ct->cols + blockDim_r.x - 1) / blockDim_r.x,
//                    (ct->rows + blockDim_r.y - 1) / blockDim_r.y);
//     cuMatrix<float> i_grad();
//     cuMatrix<float> a_grad();
//     cuMatrix<float> f_grad();
//     cuMatrix<float> c_grad();
//     cuMatrix<float> o_weights_grad();
//     cuMatrix<float> a_weights_grad();
//     cuMatrix<float> f_weights_grad();
//     cuMatrix<float> i_weights_grad();

//     for (int t = 0; t < T; t++) {
//         matrixElementWiseMul(pre_grad, tanh_ct, pre_grad);// ot gradient
//         o_layer->backpropagation(pre_grad);
//         matrixSub(o_weights_grad,o_layer->getWeightsGrad(),-1); //  weights addition
//         tanh_grad << < blockDim_r, gridDim_r >> > (pre_grad->getDev(), tanh_ct->getDev(), tanh_ct->rows, tanh_ct->cols);//ct gradient
//         matrixElementWiseMul(pre_grad->getDev(),ot,c_grad,pre_grad->rows,pre_grad->cols);
//         matrixElementWiseMul(pre_grad,ot,i_grad);
//         matrixElementWiseMul(pre_grad,it,a_grad);
//         matrixElementWiseMul(pre_grad,ct-1,f_grad);
//         i_layer->backpropagation(i_grad);
//         matrixSub(i_weights_grad,i_layer->getWeightsGrad(),-1); //  weights addition
//         f_layer->backpropagation(f_grad);
//         matrixSub(f_weights_grad,f_layer->getWeightsGrad(),-1); //  weights addition
//         a_layer->backpropagation(a_grad);
//         matrixSub(a_weights_grad,a_layer->getWeightsGrad(),-1); //  weights addition
//         matrixSub(c_grad,pre_grad,)

//         matrixElementWiseMul(pre_grad,ft,c_grad);
//     }
}

cuMatrix<float> *LSTM::getGrad() {
    return this->pre_hidden;
}

void LSTM::updateWeight() {

}

void LSTM::printParameter() {

}