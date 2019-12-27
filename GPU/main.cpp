#include "Default.h"
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include "CLI11.hpp"
#include "Perceptron.h"
#include "CSV.h"

using namespace std;
string train, eval;
int iterations;
float learning_rate;
bool verbose = false;

void print(vector<float> const &input){
	copy(input.begin(),
			input.end(),
			ostream_iterator<int>(cout, ", "));
}

void training(Perceptron *p){
	vector<vector<float>> data;
	vector<float> classes;
	tie(data, classes) =  CSV::parse_data_with_classes(train);
	p->fit(data, classes);
}

void evaluation(Perceptron *p){
	vector<vector<float>> data;
	vector<float> classes;
	tie(data, classes) =  CSV::parse_data_with_classes(eval);

	unsigned int correct = 0;

	for (size_t i = 0; i < data.size(); i++)
	{
		auto result = p->predict(data[i]);
		auto expected = classes[i];
		if(verbose)
			cout << "Classification: " << result << " Expected: " << expected << endl;
		if(result == expected) correct++;
	}

	unsigned int wrong = data.size() - correct;

	cout << "==============" << endl;
	cout << "Total: " << data.size() <<endl;
	cout << "Correct: " << correct << " " << (correct/data.size()) * 100 << "%" << endl;
	cout << "Wrong: " << wrong << " " << (wrong/data.size()) * 100 << "%" << endl;
	cout << "==============" << endl;
}

int main(int argc, char* argv[])
{
	CLI::App app{"CUDA Perceptron"};	
    app.add_option("-t,--train", train, "Training dataset, data and expected value.")->required();    
    app.add_option("-e,--eval", eval, "Dataset for evaluation, contains data with expected output.")->required();
    app.add_option("-i,--iterations", iterations, "Number of iterations for training.")->default_val("100");
    app.add_option("-l,--lrate", learning_rate, "Learning rate.")->default_val("0.1");
    app.add_flag("-v,--verbose", verbose, "Specify for verbose output");	
    CLI11_PARSE(app, argc, argv);

	auto p = new Perceptron(learning_rate, iterations);
	training(p);
	evaluation(p);
	return 0;
}