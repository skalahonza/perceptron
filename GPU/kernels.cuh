#pragma once
#include <stdio.h>
#include <stdlib.h>
#include "default.h"

// dot product
// scalar vector multiplication
// calssification
// evaluation reduction sum


float* update(float learn_rate, float* expected, float* data, float *bias, float *weights, int size);
float *dot(float* a, float* b, int size);
void scale(float *scaler, float* vector, float *result, int size);
float* classify(float* data, float* weights, float* bias, int length, int size);