#pragma once
#include <vector>
using namespace std;

class Perceptron
{
public:
	Perceptron(float eta, int epochs);
	void Fit(vector<vector<float>> data, vector<float> classes);
	float NetInput(const vector<float>& input);
	float Predict(const vector<float>& X);
private:
	int m_epochs;
	float learn_rate;
	float bias;
	vector<float> weights;
};

