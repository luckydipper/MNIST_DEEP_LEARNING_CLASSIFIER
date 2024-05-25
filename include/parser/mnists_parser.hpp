#include <fstream>
#include <iostream>
#include <string> 
#include <vector>
using namespace std;

vector< vector<unsigned char> > getInputImgs(const std::string& path);
vector<int> getLabels(const std::string& path);
