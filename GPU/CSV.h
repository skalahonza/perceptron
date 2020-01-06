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
	/**
	 * @brief Parse data from a CSV file and return 2D vector of data and 1D vector of classes
	 * 
	 * @param Path to CSV file
	 * @return tuple<vector<vector<float>>, vector<float>> Data and classes
	 */
	static tuple<vector<vector<float>>, vector<float>> parse_data_with_classes(const string& filename);
private:		
	static vector<float> parse_line(ifstream& file, char separator = ',');
};

