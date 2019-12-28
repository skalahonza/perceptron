#pragma once
#include <vector>
using namespace std;

class Perceptron
{
public:
	Perceptron(float eta, int epochs, bool _verbose);
	// Train Perceptron on given dataset
	void fit(vector<vector<float>> &data, vector<float> &classes);
	void fit_gpu(float* data, float* classes, int data_len, int size);
	// Classify input data
	float predict(const vector<float> &X);
	float *predict_gpu(float *data, int length, int size);
private:
	int m_epochs;
	float learn_rate;
	float bias;
	bool verbose;
	vector<float> weights;
	float net_input(const vector<float>& input);
	float *weights_gpu = nullptr;
	float *bias_gpu = nullptr;
};

