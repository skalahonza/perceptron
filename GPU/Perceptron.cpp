#include <iostream>
#include "Perceptron.h"
#include "kernels.cuh"
#include "debug.h"

Perceptron::Perceptron(float eta, int epochs, bool _verbose) :
	m_epochs(epochs),
	learn_rate(eta),
	verbose(_verbose)
{
	// bias gpu
	cudaMalloc(&bias_gpu, 1 * sizeof(float));
	cudaMemset(bias_gpu, 0, sizeof(float));
}

void Perceptron::fit(vector<vector<float>>& data, vector<float>& classes) {
	weights.resize(data[0].size(), 0);

	for (int i = 0; i < m_epochs; i++) {
		if (verbose)
			cout << "Starting epoch: " << i << " | ";

		for (size_t j = 0; j < data.size(); j++) {
			float update = learn_rate * (classes[j] - predict(data[j]));
			for (size_t w = 0; w < weights.size(); w++) {
				weights[w] += update * data[j][w];
			}
			bias = update;
		}

		if (verbose) {
			cout << "Weights: ";
			for (auto x : weights)
				cout << x << " ";
			cout << "| Bias: " << bias;
			cout << endl;
		}
	}
}

void Perceptron::fit_gpu(float* data, float* classes, int data_len, int size)
{
	// resize weights
	if (weights_gpu != nullptr)
	{
		gpuErrchk(cudaFree(weights_gpu));
	}
	gpuErrchk(cudaMalloc(&weights_gpu, size * sizeof(float)));
	gpuErrchk(cudaMemset(weights_gpu, 0, size * sizeof(float)));

	for (int i = 0; i < m_epochs; i++) {
		if (verbose)
			cout << "Starting GPU epoch: " << i << " | ";
		//parallel data - kernel
		for (size_t j = 0; j < data_len; j++) {
			// parallel prediction			            
			float* u = update(learn_rate, (classes + j), (data + size * j), bias_gpu, weights_gpu, size);
			// parallel weight adjustment
			// W = u*D
			scale(u, (data + size * j), weights_gpu, size);
			gpuErrchk(cudaMemcpy(bias_gpu, u, sizeof(float), cudaMemcpyDeviceToDevice));
			gpuErrchk(cudaFree(u));
		}

		if (verbose) {
			float* ws = (float*)calloc(size, sizeof(float));
			float* b = (float*)calloc(1, sizeof(float));
			gpuErrchk(cudaMemcpy(ws, weights_gpu, sizeof(float) * size, cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(b, bias_gpu, sizeof(float) * 1, cudaMemcpyDeviceToHost));
			cout << "Weights GPU: ";
			for (size_t i = 0; i < size; i++)
			{
				cout << ws[i] << " ";
			}
			cout << "| Bias GPU: " << *b;
			cout << endl;
			free(ws);
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

float* Perceptron::predict_gpu(float* data, int length, int size)
{
	float* result = classify(data, weights_gpu, bias_gpu, length, size);
	return result;
}

int Perceptron::verify(float* predictions, float* classes, int size)
{
	int* x = eval(predictions, classes, size);	
	int result = 0;
	int *tmp = (int*)calloc(size,sizeof(int));
	gpuErrchk(cudaMemcpy(tmp, x, size*sizeof(int), cudaMemcpyDeviceToHost));			
	cudaFree(x);	
	for (size_t i = 0; i < size; i++)
	{
		result += tmp[i];
	}
	return result;
}
