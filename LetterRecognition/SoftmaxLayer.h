#pragma once

#include "Layer.h"
class SoftmaxLayer : public Layer
{
public:
	SoftmaxLayer(int leftNeurons, int neurons, int rightNeurons);
	virtual void calcOutput(Layer* pLeft);
	virtual void calcError(Layer* pRight);
};

