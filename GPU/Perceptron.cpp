#include <iostream>
#include "Perceptron.h"

Perceptron::Perceptron(float eta, int epochs) : m_epochs(epochs), learn_rate(eta) {}

void Perceptron::Fit(vector<vector<float>> data, vector<float> classes) {
    weights.resize(data[0].size(), 0);

    for (int i = 0; i < m_epochs; i++) {
        cout << "Starting epoch: " << i << endl;

        for (size_t j = 0; j < data.size(); j++) {
            float update = learn_rate * (classes[j] - Predict(data[j]));
            for (size_t w = 0; w < weights.size(); w++) {
                weights[w] += update * data[j][w];
            }
            bias = update;
        }
    }
}

float Perceptron::NetInput(const vector<float> &input) {
    float prob = bias;

    for (size_t i = 0; i < input.size(); i++) {
        prob += input[i] * weights[i];
    }

    return prob;
}

float Perceptron::Predict(const vector<float> &X) {
    return NetInput(X) > 0 ? 1.f : -1.f;
}