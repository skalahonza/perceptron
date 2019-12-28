#include <iostream>
#include "Perceptron.h"
#include "kernels.cuh"

Perceptron::Perceptron(float eta, int epochs) :
	m_epochs(epochs),
	learn_rate(eta)
{
	// bias gpu
	cudaMalloc(&bias_gpu, 1 * sizeof(float));
}

void Perceptron::fit(vector<vector<float>>& data, vector<float>& classes) {
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
		for (auto x : weights)
			cout << x << " ";
		cout << endl;
	}
}

void Perceptron::fit_gpu(float** data, float* classes, int data_len, int size)
{
	// resize weights
	if (weights_gpu != nullptr)
	{
		gpuErrchk(cudaFree(weights_gpu));
	}
	gpuErrchk(cudaMalloc(&weights_gpu, size * sizeof(float)));

	for (int i = 0; i < m_epochs; i++) {
		//parallel data - kernel

		for (size_t j = 0; j < data_len; j++) {
			// parallel prediction			            
			float* u = update(learn_rate, (classes + j), data[j], bias, weights_gpu, data_len);
			// parallel weight adjustment
			// W = u*D
			scale(u, data[j], weights_gpu, size);
			gpuErrchk(cudaMemcpy(bias_gpu,u,sizeof(float),cudaMemcpyDeviceToDevice));			
			gpuErrchk(cudaFree(u));
		}
	}
}

float Perceptron::net_input(const vector<float>& input) {
	float prob = bias;
	for (size_t i = 0; i < input.size(); i++) {
		prob += input[i] * weights[i];
	}
	return prob;
}

float Perceptron::predict(const vector<float>& X) {
	return net_input(X) > 0 ? 1.f : -1.f;
}
