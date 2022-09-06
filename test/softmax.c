#include <math.h>
#include <assert.h>
#include "softmax.h"

void softmax(double* input, size_t size) {

    int i;
    double sum = 0.0;
    double max = 0.0;

    for (i = 0; i < size; i++) {
        if (input[i] > max) {
            max = input[i];
        }
    }

    for (i = 0; i < size; i++) {
        input[i] = exp(input[i] - max);
        sum += input[i];
    }

    for (i = 0; i < size; i++) {
        input[i] /= sum;
    }

}