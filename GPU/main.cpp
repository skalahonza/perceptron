#include <string>
#include <vector>
#include <sstream>
#include "CLI11.hpp"
#include "Perceptron.h"
#include "CSV.h"
#include "cuda_runtime.h"
#include "default.h"
#include <chrono>
#include <ctime>

using namespace std;
using namespace std::chrono;
string train, eval;
int iterations;
float learning_rate;
bool verbose = false;


void print_time(time_point<system_clock> start, time_point<system_clock> end, const string& message) {
	duration<double> elapsed_seconds = end - start;
	time_t end_time = system_clock::to_time_t(end);
	cout << message << elapsed_seconds.count() << "s" << endl;
}

void print_time(time_point<system_clock> start, time_point<system_clock> end) {
	print_time(start, end, "elapsed time: ");
}

void print(vector<float> const& input) {
	copy(input.begin(),
		input.end(),
		ostream_iterator<int>(cout, ", "));
}

void training(Perceptron* p) {
	cout << "==============" << endl;
	cout << "CPU Training" << endl;
	cout << "==============" << endl;

	vector<vector<float>> data;
	vector<float> classes;
	tie(data, classes) = CSV::parse_data_with_classes(train);

	auto start = std::chrono::system_clock::now();
	p->fit(data, classes);
	auto end = std::chrono::system_clock::now();
	print_time(start, end);
}

void training_gpu(Perceptron* p) {
	cout << "==============" << endl;
	cout << "GPU Training" << endl;
	cout << "==============" << endl;

	vector<vector<float>> data;
	vector<float> classes;
	tie(data, classes) = CSV::parse_data_with_classes(train);

	float* g_classes;
	gpuErrchk(cudaMalloc(&g_classes, classes.size() * sizeof(float)));


	//copy to GPU
	auto startCopy = std::chrono::system_clock::now();
	gpuErrchk(cudaMemcpy(g_classes, &classes[0], classes.size() * sizeof(float), cudaMemcpyHostToDevice));

	float* g_data;
	size_t pitch;
	auto width = data[0].size();
	cudaMallocPitch((void**)&g_data, &pitch, width * sizeof(float), data.size());
	for (size_t i = 0; i < data.size(); i++)
	{
		auto& current = data[i];
		gpuErrchk(cudaMemcpy(g_data + i * width, &(current[0]), width * sizeof(float), cudaMemcpyHostToDevice));
	}

	auto start = std::chrono::system_clock::now();
	p->fit_gpu(g_data, g_classes, data.size(), width);
	cudaDeviceSynchronize();
	auto end = std::chrono::system_clock::now();	
	print_time(start, end);
	print_time(startCopy, end, "elapsed time with copying: ");
}

void evaluation(Perceptron* p) {
	vector<vector<float>> data;
	vector<float> classes;
	tie(data, classes) = CSV::parse_data_with_classes(eval);

	unsigned int correct = 0;

	cout << "==============" << endl;
	cout << "CPU EVALUATION" << endl;
	cout << "==============" << endl;

	auto start = std::chrono::system_clock::now();
	for (size_t i = 0; i < data.size(); i++)
	{
		auto result = p->predict(data[i]);
		auto expected = classes[i];
		if (verbose)
			cout << "Classification: " << result << " Expected: " << expected << endl;
		if (result == expected) correct++;
	}
	auto end = std::chrono::system_clock::now();
	print_time(start, end);

	unsigned int wrong = data.size() - correct;

	cout << "==============" << endl;
	cout << "Total: " << data.size() << endl;
	cout << "Correct: " << correct << " " << (correct / (float)data.size()) * 100 << "%" << endl;
	cout << "Wrong: " << wrong << " " << (wrong / (float)data.size()) * 100 << "%" << endl;
	cout << "==============" << endl;
}

void evaluation_gpu(Perceptron* p) {
	cout << "==============" << endl;
	cout << "GPU EVALUATION" << endl;
	cout << "==============" << endl;
	vector<vector<float>> data;
	vector<float> classes;
	tie(data, classes) = CSV::parse_data_with_classes(eval);

	// allocation GPU
	float* g_classes;
	gpuErrchk(cudaMalloc(&g_classes, classes.size() * sizeof(float)));
	float* g_data;
	size_t pitch;
	auto width = data[0].size();
	cudaMallocPitch((void**)&g_data, &pitch, width * sizeof(float), data.size());

	// copy to GPU
	auto startCopy = std::chrono::system_clock::now();
	gpuErrchk(cudaMemcpy(g_classes, &classes[0], classes.size() * sizeof(float), cudaMemcpyHostToDevice));

	for (size_t i = 0; i < data.size(); i++)
	{
		auto& current = data[i];
		gpuErrchk(cudaMemcpy(g_data + i * width, &(current[0]), width * sizeof(float), cudaMemcpyHostToDevice));
	}

	auto start = std::chrono::system_clock::now();
	float* result = p->predict_gpu(g_data, data.size(), width);
	cudaDeviceSynchronize();
	int correct = 0;	
	auto end = std::chrono::system_clock::now();

	float* result_cpu = (float*)calloc(data.size(), sizeof(float));
	gpuErrchk(cudaMemcpy(result_cpu, result, data.size() * sizeof(float), cudaMemcpyDeviceToHost));
	auto endCopy = std::chrono::system_clock::now();

	print_time(start, end);
	print_time(startCopy, endCopy, "elapsed time with copying: ");


	for (size_t i = 0; i < data.size(); i++)
	{
		auto expected = classes[i];
		if (verbose)
			cout << "Classification: " << result_cpu[i] << " Expected: " << expected << endl;
		if (result_cpu[i] == expected) correct++;
	}
	int wrong = data.size() - correct;

	cout << "==============" << endl;
	cout << "Total: " << data.size() << endl;
	cout << "Correct: " << correct << " " << (correct / (float)data.size()) * 100 << "%" << endl;
	cout << "Wrong: " << wrong << " " << (wrong / (float)data.size()) * 100 << "%" << endl;
	cout << "==============" << endl;
}

int main(int argc, char* argv[])
{
	CLI::App app{ "CUDA Perceptron" };
	app.add_option("-t,--train", train, "Training dataset, CSV file with data and expected value.")->required();
	app.add_option("-e,--eval", eval, "Dataset for evaluation, CSV file with data with expected output.")->required();
	app.add_option("-i,--iterations", iterations, "Number of iterations for training.")->default_val("5");
	app.add_option("-l,--lrate", learning_rate, "Learning rate.")->default_val("0.1");
	app.add_flag("-v,--verbose", verbose, "Specify for verbose output");
	CLI11_PARSE(app, argc, argv);

	auto p = new Perceptron(learning_rate, iterations, verbose);
	training(p);
	evaluation(p);
	training_gpu(p);
	evaluation_gpu(p);
	return 0;
}