#include <string>
#include <vector>
#include <sstream>
#include "CLI11.hpp"
#include "Perceptron.h"
#include "CSV.h"
#include "cuda_runtime.h"
#include "default.h"

using namespace std;
string train, eval;
int iterations;
float learning_rate;
bool verbose = false;

void print(vector<float> const& input) {
	copy(input.begin(),
		input.end(),
		ostream_iterator<int>(cout, ", "));
}

void training(Perceptron* p) {
	vector<vector<float>> data;
	vector<float> classes;
	tie(data, classes) = CSV::parse_data_with_classes(train);
	p->fit(data, classes);
}

void training_gpu(Perceptron* p) {
	vector<vector<float>> data;
	vector<float> classes;
	tie(data, classes) = CSV::parse_data_with_classes(train);

	float* g_classes;
	gpuErrchk(cudaMalloc(&g_classes, classes.size() * sizeof(float)));
	gpuErrchk(cudaMemcpy(g_classes, &classes[0], classes.size() * sizeof(float), cudaMemcpyHostToDevice));

	float* g_data;	
	size_t pitch;
	auto width = data[0].size();
	cudaMallocPitch((void**)&g_data, &pitch, width * sizeof(float), data.size());	
	for (size_t i = 0; i < data.size(); i++)
	{
		auto& current = data[i];
		gpuErrchk(cudaMemcpy(g_data + i*width,&(current[0]),width * sizeof(float), cudaMemcpyHostToDevice));		
	}
	p->fit_gpu(g_data, g_classes, data.size(), width);
}

void evaluation(Perceptron* p) {
	vector<vector<float>> data;
	vector<float> classes;
	tie(data, classes) = CSV::parse_data_with_classes(eval);

	unsigned int correct = 0;

	for (size_t i = 0; i < data.size(); i++)
	{
		auto result = p->predict(data[i]);
		auto expected = classes[i];
		if (verbose)
			cout << "Classification: " << result << " Expected: " << expected << endl;
		if (result == expected) correct++;
	}

	unsigned int wrong = data.size() - correct;

	cout << "==============" << endl;
	cout << "Total: " << data.size() << endl;
	cout << "Correct: " << correct << " " << (correct / data.size()) * 100 << "%" << endl;
	cout << "Wrong: " << wrong << " " << (wrong / data.size()) * 100 << "%" << endl;
	cout << "==============" << endl;
}

int main(int argc, char* argv[])
{
	CLI::App app{ "CUDA Perceptron" };
	app.add_option("-t,--train", train, "Training dataset, data and expected value.")->required();
	app.add_option("-e,--eval", eval, "Dataset for evaluation, contains data with expected output.")->required();
	app.add_option("-i,--iterations", iterations, "Number of iterations for training.")->default_val("5");
	app.add_option("-l,--lrate", learning_rate, "Learning rate.")->default_val("0.1");
	app.add_flag("-v,--verbose", verbose, "Specify for verbose output");
	CLI11_PARSE(app, argc, argv);

	auto p = new Perceptron(learning_rate, iterations, verbose);
	training(p);
	training_gpu(p);
	//evaluation(p);
	return 0;
}