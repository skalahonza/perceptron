#pragma once
#include <vector>
using namespace std;

class Perceptron
{
public:
	Perceptron(float eta, int epochs);
	// Train Perceptron on given dataset
	void fit(vector<vector<float>> &data, vector<float> &classes);
	void fit_gpu(float **data, float *classes, int data_len, int classes_len, int size);
	// Classify input data
	float predict(const vector<float> &X);
	float predict_gpu(float *x, int lenght);
private:
	int m_epochs;
	float learn_rate;
	float bias;
	vector<float> weights;
	float net_input(const vector<float>& input);
};

