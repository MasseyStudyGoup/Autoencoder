#ifndef GLOBALVARIABLES_H
#define GLOBALVARIABLES_H

#define INPUT_NEURONS		16
#define HIDDEN_NEURONS		16 
#define OUTPUT_NEURONS		26

struct Letter_S{
    char symbol;
    double O[OUTPUT_NEURONS];
    double X[INPUT_NEURONS];
};

//extern int NUMBER_OF_PATTERNS;
const int NUMBER_OF_PATTERNS = 20000;
const int NUMBER_OF_TRAINING_PATTERNS = 16000;
const int NUMBER_OF_TEST_PATTERNS = 4000;

const int DEFAULT_MAX_EPOCHS = 10;
const double DEFAULT_LEARNING_RATE = 1.0;
const double LEARNING_RATE = 0.1;
#endif // GLOBALVARIABLES_H
