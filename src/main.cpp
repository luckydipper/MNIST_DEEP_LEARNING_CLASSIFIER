#include <iostream>
#include "mnists_parser.hpp"
using namespace std;

int main(){
    cin.tie(nullptr);
    ios_base::sync_with_stdio(false);
    cout << "hello world";

    const string train_X_path = "../dataset/train/train-images.idx3-ubyte";
    const string train_y_path = "../dataset/train/train-labels.idx1-ubyte";
    const string test_X_path  = "../dataset/test/t10k-images.idx3-ubyte";
    const string test_y_path  = "../dataset/test/t10k-labels.idx1-ubyte";
    vector<vector<unsigned char> > train_X = getData(train_X_path);

    for (const vector<unsigned char> &image : train_X){
        for(int i = 0; i< 28*28; i++){
            if( (i % 28) == 0)
                cout <<"\n"; 
            cout << image[i] << " ";
        }
    }
}