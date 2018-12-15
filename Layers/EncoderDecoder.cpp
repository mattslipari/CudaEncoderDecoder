#include "EncoderDecoder.h"

void EncoderDecoder::forward() {
    encoder->forward(NULL);
    //printf("forward done\n");
    decoder->forward(encoder->ht);
}

cuMatrix<float> *EncoderDecoder::getGrad() {
    return NULL;
}

void EncoderDecoder::updateWeight() {
    encoder->updateWeight();
    decoder->updateWeight();
}

void EncoderDecoder::printParameter() {
    encoder->printParameter();
    decoder->printParameter();
}
