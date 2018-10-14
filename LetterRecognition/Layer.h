#ifndef __LAYER_H
#define __LAYER_H

#include "Neuron.h"

enum LayerType {
	LT_INPUT = 0,
	LT_HIDDEN = 1,
	LT_OUPUT = 2
};

class Layer
{
public:
	Layer(int leftNeurons, int neurons, int rightNeurons);
	virtual void calcOutput(Layer* pLeft);
	virtual void calcError(Layer* pRight);
	virtual void updateWeights(Layer* pRight, double learningRate);
	void copyWeights(Layer* fromLayer);
	void setDesiredOuput(double* vals);
	double* getOutput();
	void saveWeights(FILE* pFile);
	void loadWeights(FILE* pFile);
public:
	int m_N, m_rightN;
	CArray<Neuron*> m_neurons;
	//weights to the next layer
	Neuron* m_pBias;
	double* m_err;
	double* m_output;
private:
	LayerType m_type;
};

#endif