#include "CSV.h"

tuple<vector<vector<float>>, vector<float>> CSV::parse_data_with_classes(const string &filename)
{
	ifstream in(filename);
	vector<vector<float>> data;
	vector<float> classes;
	vector<float> line;	
	for (;;) {
		line = parse_line(in);
		if (line.empty()) {
			break;
		}
		else {
			data.push_back(vector<float>(line.begin(), line.end() - 1));
			classes.push_back(line.back());
		}
	}
	return {data, classes};
}

vector<float> CSV::parse_line(ifstream &file, char separator)
{
	string line;
	getline(file, line);
	stringstream ss(line);
	vector<float> dest;
	for (float i; ss >> i;) {
		dest.push_back(i);
		if (ss.peek() == separator)
			ss.ignore();
	}
	return dest;
}
