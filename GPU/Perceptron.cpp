#include <iostream>
#include "Perceptron.h"

Perceptron::Perceptron(float eta, int epochs) : 
	m_epochs(epochs), 
	learn_rate(eta) {}

void Perceptron::fit(vector<vector<float>> &data, vector<float> &classes) {
    weights.resize(data[0].size(), 0);

    for (int i = 0; i < m_epochs; i++) {
        cout << "Starting epoch: " << i << " | ";

        for (size_t j = 0; j < data.size(); j++) {
            float update = learn_rate * (classes[j] - predict(data[j]));
            for (size_t w = 0; w < weights.size(); w++) {
                weights[w] += update * data[j][w];				
            }
			
            bias = update;
        }
		for(auto x : weights)
			cout << x << " ";
		cout << endl;
    }
}

void Perceptron::fit_gpu(float** data, float* classes, int data_len, int classes_len, int size)
{
	 for (int i = 0; i < m_epochs; i++) {
		//parallel data - kernel

        for (size_t j = 0; j < data_len; j++) {
			// parallel prediction
			// u = lr * error
            float update = learn_rate * (classes[j] - predict_gpu(data[j],data_len));

			// parallel wight adjustment
			// W = u*D
            for (size_t w = 0; w < weights.size(); w++) {
                weights[w] += update * data[j][w];				
            }
			
            bias = update;
        }
    }
}

float Perceptron::net_input(const vector<float> &input) {
    float prob = bias;
    for (size_t i = 0; i < input.size(); i++) {
        prob += input[i] * weights[i];
    }
    return prob;
}

float Perceptron::net_input_gpu(float* input, int length)
{
	//prop = I'W + b
	return 0.0f;
}

float Perceptron::predict(const vector<float> &X) {
    return net_input(X) > 0 ? 1.f : -1.f;
}

float Perceptron::predict_gpu(float* x, int lenght)
{
	return net_input_gpu(x,lenght) > 0 ? 1.f : -1.f;
}
