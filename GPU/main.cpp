#include "Default.h"
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include "Perceptron.h"

using namespace std;

vector<float> parseline() {
	vector<float> dest;
	string line;
	getline(cin, line);
	stringstream ss(line);

	for (float i; ss >> i;) {
		dest.push_back(i);
		if (ss.peek() == ',')
			ss.ignore();
	}
	return dest;
}

int main()
{
	//train
	//clasify
	//test
	vector<vector<float>> data;
	vector<float> classes;
	vector<float> line;
	for (;;) {
		line = parseline();
		if (line.empty()) {
			break;
		}
		else { 
			data.push_back(vector<float>(line.begin(), line.end() - 1)); 
			classes.push_back(line.back());
		}
	}

	auto p = new Perceptron(1, 10);	
	p->Fit(data,classes);
	cout << p->Predict(data[0]);
	return 0;
}