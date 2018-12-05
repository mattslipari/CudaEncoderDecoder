#ifndef __LAYERS_BASE_CU_H__
#define __LAYERS_BASE_CU_H__

#include <string>
#include <vector>
#include "../Common/cuMatrix.h"

class LayerBase
{
public:
    virtual void feedforward() = 0;
    virtual void backpropagation(cuMatrix<float>* pre_grad) = 0;
    virtual cuMatrix<float> *getGrad() = 0;
    virtual void updateWeight() = 0;


    virtual cuMatrix<float>* getOutputs() = 0;

    virtual void printParameter() = 0;

    std::string m_name;
    std::vector<std::string> m_preLayer;
};

class Layers
{
public:
    LayerBase* get(std::string name);
    void set(std::string name, LayerBase* layer);
private:
    std::map<std::string, LayerBase*>m_maps;
};
#endif