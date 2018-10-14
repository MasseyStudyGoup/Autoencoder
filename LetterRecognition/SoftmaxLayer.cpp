#include "stdafx.h"
#include "SoftmaxLayer.h"


SoftmaxLayer::SoftmaxLayer(int leftNeurons, int neurons, int rightNeurons)
	:Layer(leftNeurons,  neurons, rightNeurons)
{
}
/*
void SoftmaxLayer::setDesiredOuput(double* vals)
{
	//const int sum_output = 3.4; //0.1 * 25 + 0.9;
	double sum = 0;
	for (int i = 0; i < m_N; i++)
		sum += vals[i];

	for (int i = 0; i < m_N; i++)
		m_neurons[i]->m_t = vals[i] / sum;
}
*/
void SoftmaxLayer::calcOutput(Layer* pLeft)
{
	double sumExp = 0;
	int size = pLeft->m_neurons.GetSize();
	for (int i = 0; i < m_N; i++) {
		double sum = 0;

		for (int j = 0; j < size; j++) {
			sum += pLeft->m_neurons[j]->m_o * pLeft->m_neurons[j]->m_w[i];
		}
		sum += pLeft->m_pBias->m_w[i];

		m_neurons[i]->m_o = exp(sum);
		sumExp += m_neurons[i]->m_o;
	}

	if (sumExp > 1000)
		printf("error");
	for (int i = 0; i < m_N; i++) {
		double o = m_neurons[i]->m_o / sumExp;
		m_neurons[i]->m_o = o;
	}
}

void SoftmaxLayer::calcError(Layer* pRight)
{
	for (int out = 0; out < m_N; out++) {
		m_err[out] = m_neurons[out]->m_o - m_neurons[out]->m_t;
	}
}
