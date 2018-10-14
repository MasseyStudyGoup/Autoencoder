#include "stdafx.h"
#include "Neuron.h"

NeuronFunc Neuron::activate = Neuron::sigmoid;
NeuronFunc Neuron::activateD = Neuron::sigmoidD;

Neuron::Neuron(int connections)
{
	m_connections = connections;
	if (m_connections > 0)
	{
		m_w = new double[m_connections];
		for (int i = 0; i < m_connections; i++)
			m_w[i] = RAND_WEIGHT;
	}
	m_o = 0;
	m_t = 0;
}

Neuron::~Neuron() {
	delete m_w;
}

void Neuron::setActivationFunction(ActivationFunction af)
{
	if (af == ActivationFunction::AF_RELU) {
		Neuron::activate = relu;
		Neuron::activateD = reluD;
	}
	else if (af == ActivationFunction::AF_SIGMOID) {
		Neuron::activate = sigmoid;
		Neuron::activateD = sigmoidD;
	}
	else
	{
		Neuron::activate = tanh;
		Neuron::activateD = tanhD;
	}
}

void Neuron::setDesiredOutput(double t)
{
	m_t = t;
}

double Neuron::sigmoid(double val)
{
	return (1.0 / (1.0 + exp(-val)));
}

double Neuron::sigmoidD(double val)
{
	return (val * (1.0 - val));
}


double Neuron::tanh(double val)
{
	return 1.0 * (exp(val) - exp(-1.0 * val)) / (exp(val) + exp(-1.0 * val));
}  

double Neuron::tanhD(double val)
{
	return 1.0 - tanh(val) * tanh(val);
}  

double Neuron::relu(double val)
{
	return fmax(0.0, val);
} 

double Neuron::reluD(double val)
{
	if (val > 0)
		return 1.0;
	else
		return 0.0;
} 



