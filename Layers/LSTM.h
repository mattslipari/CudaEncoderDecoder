#ifndef CUDAENCODERDOCODER_LSTM_H
#define CUDAENCODERDOCODER_LSTM_H

#include "LayerBase.h"
#include "FullyConnect.h"

class LSTM : public LayerBase {
public:
		LSTM(cuMatrix<float> **inputs, cuMatrix<float> *pre_hidden, 
				 cuMatrix<float> *pre_cell, int input_total, int units, float lambda) {

      printf("Where is the segfault\n");

      this->inputs = inputs;
      this->pre_hidden = pre_hidden;
      this->pre_cell = pre_cell;
      this->input_total = input_total;
      this->units = units;
      this->lambda = lambda;

      printf("After setting\n");

      cuMatrix<float> *first_input = inputs[0];
      this->input_rows = first_input->rows;
      this->input_cols = first_input->cols;

      printf("Input size\n");

      int total_rows = pre_hidden->rows + this->input_rows;
      int total_cols = pre_hidden->cols + this->input_cols;

      FullyConnect *a_layer =  new FullyConnect(total_rows, total_cols, units, lambda, FullyConnect::TANH);
      FullyConnect *i_layer = new FullyConnect(total_rows, total_cols, units, lambda, FullyConnect::SIGMOID);
      FullyConnect *f_layer = new FullyConnect(total_rows, total_cols, units, lambda, FullyConnect::SIGMOID);
      FullyConnect *o_layer = new FullyConnect(total_rows, total_cols, units, lambda, FullyConnect::SIGMOID);
      
      printf("All layers done\n");

      this->a_layer = a_layer;
      this->i_layer = i_layer;
      this->f_layer = f_layer;
      this->o_layer = o_layer;
		}

    ~LSTM () {

    }

    void forward();

    void backpropagation(cuMatrix<float> *pre_grad);

    cuMatrix<float> *getGrad();

    void updateWeight();

    void printParameter();

private:
  	cuMatrix<float> **inputs;
  	cuMatrix<float> *pre_hidden;
  	cuMatrix<float> *pre_cell;
  
  	FullyConnect* a_layer;
  	FullyConnect* i_layer;
  	FullyConnect* f_layer;
  	FullyConnect* o_layer;
  
  	int units;
  	int input_total;
  	int input_rows;
  	int input_cols;
  	float lambda;
};


#endif //CUDAENCODERDOCODER_LSTM_H
