#include "stdafx.h"
#include "NeuralNetwork.h"
#include "Autoencoder.h"
#include "SoftmaxLayer.h"

Autoencoder::Autoencoder(int inputNeurons, int hiddenNeurons, int outputNeurons, double learningRate)
{
	m_learningRate = learningRate;
	m_pInput = new Layer(0, inputNeurons, hiddenNeurons);
	m_pHidden = new Layer(inputNeurons, hiddenNeurons, inputNeurons);
	m_pOutput = new SoftmaxLayer(hiddenNeurons, outputNeurons, 0);
}

Autoencoder::~Autoencoder()
{
	if (m_pInput != nullptr)
		delete m_pInput;
	m_pInput = nullptr;
	if (m_pHidden != nullptr)
		delete m_pHidden;
	m_pHidden = nullptr;
	if (m_pOutput != nullptr)
		delete m_pOutput;
	m_pOutput = nullptr;
}

double Autoencoder::train(double* input, double* desiredOutput)
{
	for (int i = 0; i < m_pInput->m_N; i++) {
		m_pInput->m_neurons[i]->m_o = input[i];
	}

	double sum = 0;
	for (int i = 0; i < m_pOutput->m_N; i++) {
		sum += desiredOutput[i];
	}

	for (int i = 0; i < m_pOutput->m_N; i++) {
		m_pOutput->m_neurons[i]->m_t = desiredOutput[i] / sum;
	}

	//feed forward
	m_pHidden->calcOutput(m_pInput);
	m_pOutput->calcOutput(m_pHidden);

	//back propagate
	m_pOutput->calcError(nullptr);
	m_pHidden->updateWeights(m_pOutput, m_learningRate);
	m_pHidden->calcError(m_pOutput);
	m_pInput->updateWeights(m_pHidden, m_learningRate);

	double sse = 0;
	for (int k = 0; k < m_pOutput->m_N; k++) {
		sse += sqr(m_pOutput->m_err[k]);
	}

	return sse/2;
}


