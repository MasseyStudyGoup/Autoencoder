#include "stdafx.h"
#include "Layer.h"

Layer::Layer(int leftNeurons, int neurons, int rightNeurons)
{
	m_N = neurons;
	m_rightN = rightNeurons;
	
	if (leftNeurons == 0)
		m_type = LayerType::LT_INPUT;
	else if (rightNeurons == 0)
		m_type = LayerType::LT_OUPUT;
	else
		m_type = LayerType::LT_HIDDEN;

	//bias only feed to the current layer
	m_pBias = nullptr;
	if (m_rightN > 0)
		m_pBias = new Neuron(m_rightN);

	for (int i = 0; i < m_N; i++) {
		Neuron* n = new Neuron(m_rightN);
		m_neurons.Add(n);
	}

	m_output = new double[m_N];
	m_err = new double[m_N];
}

void Layer::saveWeights(FILE* pFile)
{
	for (int i = 0; i < m_N; i++) 
	{
		for (int j = 0; j < m_rightN; j++) {
			fprintf(pFile, "%lf ", m_neurons[i]->m_w[j]);
		}
		fprintf(pFile, "\n");
	}
	// save the Bias
	for (int j = 0; j < m_rightN; j++) {
		fprintf(pFile, "%lf ", m_pBias->m_w[j]);
	}
	fprintf(pFile, "\n");
}

void Layer::loadWeights(FILE* pFile)
{
	double w;
	for (int i = 0; i < m_N; i++)
	{
		for (int j = 0; j < m_rightN; j++) {
			fscanf(pFile, "%lf ", &w);
			m_neurons[i]->m_w[j] = w;
		}
		fscanf(pFile, "\n");
	}
	// load the Bias
	for (int j = 0; j < m_rightN; j++) {
		fscanf(pFile, "%lf ", &w);
		m_pBias->m_w[j] = w;
	}
	fscanf(pFile, "\n");
}

void Layer::copyWeights(Layer* fromLayer)
{
	for (int i = 0; i < m_N; i++)
		memcpy(m_neurons[i]->m_w, fromLayer->m_neurons[i]->m_w, sizeof(double)*m_rightN);
	memcpy(m_pBias->m_w, fromLayer->m_pBias->m_w, sizeof(double)*m_rightN);
}

void Layer::setDesiredOuput(double* vals)
{
	for (int i = 0; i < m_N; i++)
		m_neurons[i]->m_t = vals[i];
}

void Layer::calcOutput(Layer* pLeft)
{
	if (pLeft == nullptr)
		return;

	int size = pLeft->m_neurons.GetSize();
	for (int i = 0; i < m_N; i++) {
		double sum = 0;
		
		for (int j = 0; j < size; j++) {
			sum += pLeft->m_neurons[j]->m_o * pLeft->m_neurons[j]->m_w[i];
		}
		sum += pLeft->m_pBias->m_w[i];
		m_neurons[i]->m_o = Neuron::activate(sum);
	}
}

void Layer::calcError(Layer* pRight)
{
	if (m_type == LayerType::LT_OUPUT)
	{
		for (int out = 0; out < m_N; out++) {
			Neuron* p = m_neurons[out];
			m_err[out] = (p->m_o - p->m_t) * Neuron::activateD(p->m_o);
		}
	}
	else if (m_type == LayerType::LT_HIDDEN)
	{
		for (int i = 0; i < m_N; i++) {
			m_err[i] = 0.0;
			Neuron* p = m_neurons[i];
			for (int j = 0; j < pRight->m_neurons.GetSize(); j++) {
				m_err[i] += pRight->m_err[j] * p->m_w[j];
			}

			m_err[i] *= Neuron::activateD(p->m_o);
		}
	}
}

void Layer::updateWeights(Layer* pRight, double learningRate)
{
	if (pRight == nullptr)
		return;

	/* Update the weights (step 6 for hidden cell) */
	for (int i = 0; i < m_N; i++) {
		Neuron* p = m_neurons[i];
		for (int j = 0; j < pRight->m_neurons.GetSize(); j++) {
			p->m_w[j] -= (learningRate * pRight->m_err[j] * p->m_o);
		}
	}

	//bias
	for (int out = 0; out < m_N; out++) {
		m_pBias->m_w[out] -= (learningRate * pRight->m_err[out]);
	}
}


double* Layer::getOutput() {
	for (int i = 0; i < m_N; i++)
		m_output[i] = m_neurons[i]->m_o;

	return m_output;
}