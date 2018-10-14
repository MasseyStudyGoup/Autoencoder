#pragma once
#include "Layer.h"

class Autoencoder
{
public:
	Autoencoder(int inputNeurons, int hiddenNeurons, int outputNeurons, double learningRate);
	~Autoencoder();
	double train(double* input, double* desiredOutput);
	Layer* getInputLayer(){
		return m_pInput;
	}
private:
	Layer* m_pInput;
	Layer* m_pHidden;
	Layer* m_pOutput;

	double m_learningRate;
};

