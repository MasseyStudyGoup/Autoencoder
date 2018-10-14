#pragma once

#define RAND_WEIGHT	( ((float)rand() / (float)RAND_MAX) - 0.5)

enum ActivationFunction {
	AF_RELU = 0,
	AF_SIGMOID = 1,
	AF_TANH = 2
};

typedef double (*NeuronFunc)(double val);

class Neuron
{
public:
	Neuron(int connections);
	~Neuron();
	void setDesiredOutput(double t);
	static double sigmoid(double val);
	static double sigmoidD(double val);
	static double tanh(double val);
	static double tanhD(double val);
	static double relu(double val);
	static double reluD(double val);
	static void setActivationFunction(ActivationFunction af);
public:
	int m_connections;
	double m_o; //actual ouput
	double m_t; //desired output
	double* m_w; //weights

	static NeuronFunc activate;
	static NeuronFunc activateD;
};

