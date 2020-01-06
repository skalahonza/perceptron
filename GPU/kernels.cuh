#pragma once
#include <stdio.h>
#include <stdlib.h>
#include "default.h"

// dot product
// scalar vector multiplication
// calssification
// evaluation reduction sum

/**
 * @brief Compute value for update variable 
 * 
 * @param learn_rate Learning rate 
 * @param expected Expected value for given data
 * @param data Data to classify
 * @param bias Perceptron bias
 * @param weights Perceptrpn weights
 * @param size Size of data and weights
 * @return float* Update value
 */
float* update(float learn_rate, float* expected, float* data, float *bias, float *weights, int size);
/**
 * @brief Compute dot product of two vectors
 * 
 * @param a First vector
 * @param b Second vector
 * @param size Size of vectors
 * @return float* Result vector
 */
float *dot(float* a, float* b, int size);
/**
 * @brief Scale vector with given scaler
 * 
 * @param scaler Scaler (number)
 * @param vector Vector to scale
 * @param result Result will be stored here
 * @param size Size of vector
 */
void scale(float *scaler, float* vector, float *result, int size);
/**
 * @brief Classify matrix of data with given weights and bias
 * 
 * @param data Data to classify
 * @param weights Perceptron weights
 * @param bias Perceptron bias
 * @param length Length of data vector
 * @param size Ammount of data 
 * @return float* Vector of classification
 */
float* classify(float* data, float* weights, float* bias, int length, int size);