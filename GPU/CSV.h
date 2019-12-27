#pragma once
#include <tuple>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
using namespace std;

class CSV
{
public:
	static tuple<vector<vector<float>>, vector<float>> parse_data_with_classes(const string& filename);
private:		
	static vector<float> parse_line(ifstream& file, char separator = ',');
};

