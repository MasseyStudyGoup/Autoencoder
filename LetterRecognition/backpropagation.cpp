#include "stdafx.h"
#include "backpropagation.h"


#define sqr(x)	((x) * (x))

//FILE *out;

//#define LEARN_RATE		0.2

#define RAND_WEIGHT	( ((float)rand() / (float)RAND_MAX) - 0.5)

#define getSRand()	((float)rand() / (float)RAND_MAX)
#define getRand(x)      (int)((float)(x)*rand()/(RAND_MAX+1.0))

///////////////////////////////////////////////////////////////////

Backpropagation::Backpropagation()
{
   initialise();
}  //untouched

void Backpropagation::initialise()
{
    err=0;
    sample=0;
    iterations=0;
    sum = 0;


    /* Seed the random number generator */
    srand( unsigned (time(nullptr) ) );

    assignRandomWeights();
}  //untouched

/*
 *  assignRandomWeights()
 *
 *  Assign a set of random weights to the network.
 *
 */

void Backpropagation::assignRandomWeights(void) 
{
	int hid, inp, out;

	for (inp = 0; inp < INPUT_NEURONS + 1; inp++) {
		for (hid = 0; hid < HIDDEN_NEURONS; hid++) {
			wih[inp][hid] = RAND_WEIGHT;
		}
	}

	for (hid = 0; hid < HIDDEN_NEURONS + 1; hid++) {
		for (out = 0; out < OUTPUT_NEURONS; out++) {
			who[hid][out] = RAND_WEIGHT;
		}
	}

} //untouched

double Backpropagation::getError_SSE(){
    return err2;
}  //untouched, just change from err to err2

void Backpropagation::loadPatterns(const char* fileName) {
	FILE* pfile = fopen(fileName, "r");
	if (pfile == nullptr) {
		printf("Backpropagation::loadPatterns could not open file %s!", fileName);
		return;
	}
	Letter_S letter;
	while (true) {
		letter.symbol = 0;
		memset(letter.X, 0, sizeof(int)*INPUT_NEURONS);
		for (int i = 0; i < OUTPUT_NEURONS; i++)
			letter.O[i] = 0.1;

		char c;
		if (fscanf(pfile, "%c", &c) == EOF)
			return;
		if (c < 'A' || c > 'Z')
			continue;

		letter.symbol = c;
		int f = 0;
		for (int i = 0; i < INPUT_NEURONS; i++) {
			if (fscanf(pfile, ",%d", &f) == EOF)
				return;
			else
				letter.X[i] = f/15.0;
		}
		//only the corresponding output is 1, others are all 0
		letter.O[letter.symbol - 'A'] = 1;

		m_train.Add(letter);
	}
}
Letter_S* Backpropagation::parsePattern(CString& pattern, Letter_S* letter)
{
	if (letter == nullptr || pattern.GetLength() == 0)
		return nullptr;
	pattern.Trim();

	CString sToken = _T("");
	for (int i = 0; i < INPUT_NEURONS; i++)
	{
		AfxExtractSubString(sToken, pattern, i, ',');
		letter->X[i] = ::StrToInt(sToken)/15.0;
	}
	return letter;
}

double* Backpropagation::testNetwork(Letter_S* testPattern2){
    //retrieve input patterns
    for(int j=0; j < INPUT_NEURONS; j++){
       inputs[j] = testPattern2->X[j];
    }

    for(int i=0; i < OUTPUT_NEURONS; i++){
        target[i] = testPattern2->O[i];
    }

    feedForward();

    return actual;
}// untouched


double Backpropagation::trainNetwork(int NUMBER_OF_DESIRED_EPOCHS)
{
    double accumulatedErr=0.0;
    iterations=0;
    int epochs=0;
    err = 0.0;
    while (1) {

        if (sample == NUMBER_OF_TRAINING_PATTERNS) {
            sample = 0;
            err = 0.0;
            epochs++;
        }

        if(epochs >= NUMBER_OF_DESIRED_EPOCHS) {

            break;
        }

        //retrieve input patterns
        for(int j=0; j < INPUT_NEURONS; j++){
           inputs[j] = m_train[sample].X[j];
        }

        for(int i=0; i < OUTPUT_NEURONS; i++){
            target[i] = m_train[sample].O[i];
        }

        feedForward();

        /* need to iterate through all ... */

        //err = 0.0;
        for (int k = 0 ; k < OUTPUT_NEURONS ; k++) {

          err += sqr( (m_train[sample].O[k] - actual[k]) );
        }

        err = 0.5 * err;
        err2 = err;
        accumulatedErr = accumulatedErr + err;

        backPropagate();
        sample ++;

      }
      return accumulatedErr;
}  //untouched ??? sample ++ first (old version)

/*
 *  sigmoid()
 *
 *  Calculate and return the sigmoid of the val argument.
 *
 */

double Backpropagation::sigmoid( double val )
{
  return (1.0 / (1.0 + exp(-val)));
} //unchanged


/*
 *  sigmoidDerivative()
 *
 *  Calculate and return the derivative of the sigmoid for the val argument.
 *
 */

double Backpropagation::sigmoidDerivative( double val )
{
  return ( val * (1.0 - val) );
} //unchanged


/*
 *  feedForward()
 *
 *  Feedforward the inputs of the neural network to the outputs.
 *
 */

void Backpropagation::feedForward( )
{
  int inp, hid, out;
  double sum, sum2 = 0.0;

  /* Calculate input to hidden layer */
  //Experiment with Relu and tanh as the activation functions of your hidden units, include sigmoid with lecture code
  for (hid = 0 ; hid < HIDDEN_NEURONS ; hid++) {

    sum = 0.0;
    for (inp = 0 ; inp < INPUT_NEURONS ; inp++) {
      sum += inputs[inp] * wih[inp][hid];
    }

    /* Add in Bias */
    sum += wih[INPUT_NEURONS][hid];
    hidden[hid] = sigmoid( sum );
  }

  /* Calculate the hidden to output layer */ //Use softmax units at the output layer
  for (out = 0 ; out < OUTPUT_NEURONS ; out++) {

    sum = 0.0;
    for (int hid2 = 0 ; hid2 < HIDDEN_NEURONS ; hid2++) {
      sum += hidden[hid2] * who[hid2][out];
    }

    /* Add in Bias */
    sum += who[HIDDEN_NEURONS][out];

    // start softmax, so dont need relu, tanh, sigmoid in this layer
    actual[out] = sigmoid(sum);
   
  }

}  // google and learn softmax in output layer, test later, maybe ok


/*
 *  backPropagate()
 *
 *  Backpropagate the error through the network.
 *
 */

void Backpropagation::backPropagate( void ) //gradient ascent not work well, change to descent
{
  int inp, hid, out;

  /* Calculate the output layer error (step 3 for output cell) */
  for (out = 0 ; out < OUTPUT_NEURONS ; out++) {
      erro[out] = actual[out] - target[out];
      erro[out] *= sigmoidDerivative( actual[out] );
  }

  /* Update the weights for the output layer (step 4 for output cell) */
  for (out = 0 ; out < OUTPUT_NEURONS ; out++) {

    for (hid = 0 ; hid < HIDDEN_NEURONS ; hid++) {
      who[hid][out] -= (LEARNING_RATE * erro[out] * hidden[hid]);
    }

    /* Update the Bias */
    who[HIDDEN_NEURONS][out] -= (LEARNING_RATE * erro[out]);

  }
   
  /* Calculate the hidden layer1 error (step 3 for hidden cell) */
  for (int hid = 0 ; hid < HIDDEN_NEURONS ; hid++) {

    errh[hid] = 0.0;
    for (out = 0 ; out < OUTPUT_NEURONS ; out++) {
      errh[hid] += erro[out] * who[hid][out];
    }

     errh[hid] *= sigmoidDerivative( hidden[hid] );

  }

  /* Update the weights for the hidden layer1 (step 4 for hidden cell) */
  for (hid = 0 ; hid < HIDDEN_NEURONS ; hid++) {

    for (inp = 0 ; inp < INPUT_NEURONS ; inp++) {
      wih[inp][hid] -= (LEARNING_RATE * errh[hid] * inputs[inp]);
    }

    /* Update the Bias */
    wih[INPUT_NEURONS][hid] -= (LEARNING_RATE * errh[hid]);

  }

}  //change a bit
