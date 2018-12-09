#include "LSTM.h"
#include "cuMatrix.h"

void forward() {
	cuMatrix<float> *input_t;
	cuMatrix<float> *input_hidden = 
		new cuMatrix<float>(pre_hidden->rows + this->input_rows, this->input_cols);
	cuMatrix<float> *cell_t = new cuMatrix<float>(this->units, this->input_rows);
	cuMatrix<float> *ia = new cuMatrix<float>(this->units, this->input_rows);
	cuMatrix<float> *fc = new cuMatrix<float>(this->units, this->input_rows);
	cuMatrix<float> *blank_bias = new cuMatrix<float>(cell_t->rows, 1);

	dim3 blockDim(16, 16, 1);
  dim3 gridDim((this->cell_t->cols + blockDim.x - 1) / blockDim.x,
               (this->cell_t->rows + blockDim.y - 1) / blockDim.y);

	for (int t = 0; t < this->input_total; t++) {
		input_t = this->inputs[t]; 
		matrixConcat(input_t, this->pre_hidden, input_hidden);

		this->a_layer->feedForward(input_hidden);
		this->i_layer->feedForward(input_hidden);
		this->f_layer->feedForward(input_hidden);
		this->o_layer->feedForward(input_hidden);

		matrixElementwiseMul(i_layer->outputs, a_layer->outputs, ia);
		matrixElementwiseMul(f_layer->outputs, this->pre_cell, fc);
		matrixSub(ia, fc, cell_t, -1);
    
    tanh <<< blockDim, gridDim >>> (cell_t->getDev(), blank_bias, cell_t->rows, cell_t->cols);            
    matrixElementwiseMul(o_layer->outputs, cell_t, this->pre_hidden);
	}
}
