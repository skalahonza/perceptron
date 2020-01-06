#pragma once
#include <vector>
using namespace std;

class Perceptron
{
public:
	/**
	 * @brief Construct a new Perceptron object
	 * 
	 * @param eta Learning rate, float value between 0 and 1
	 * @param epochs Number of traning epochs
	 * @param _verbose Specify true if verbose input is required, affects performance
	 */
	Perceptron(float eta, int epochs, bool _verbose);	
	/**
	 * @brief Train Perceptron using CPU on given traning dataset
	 * 
	 * @param data Training data
	 * @param classes Expected classes
	 */
	void fit(vector<vector<float>> &data, vector<float> &classes);
	/**
	 * @brief Train Perceptron using GPU on given traning dataset
	 * 
	 * @param data Training data - GPU memory pointer
	 * @param classes Expected classes - GPU memory pointer
	 * @param data_len Number of data
	 * @param size Size of data
	 */
	void fit_gpu(float* data, float* classes, int data_len, int size);
	/**
	 * @brief Classify input data using CPU
	 * 
	 * @param X Vector of input data
	 * @return float 1 or -1
	 */
	float predict(const vector<float> &X);
	/**
	 * @brief Classify input data using GPU
	 * 
	 * @param data Matrix of input data, data are rows of the matrix - GPU pointer
	 * @param length Lenth of data vector
	 * @param size Number of data vectors
	 * @return float* Vector of classifications with values 1 or -1 GPU POINTER
	 */
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

