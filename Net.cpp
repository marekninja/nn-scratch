#include "Net.h"

using namespace std;
#include <iostream>


Net::Net(const vector<int>& arch) {
    architecture = arch;
    int size = architecture.size();

    weightMatrices.resize(size);
    activationMatrices.resize(size);
    biasMatrices.resize(size);

    for (int i = 1; i -1 < arch.size()-1; ++i) {
        cout << "layer i:" << i << "=" << arch[i]<<endl;
        Matrix<double> weights(architecture[i],architecture[i-1]);
        weightMatrices[i]=(weights);
        Matrix<double> biases(architecture[i],1);
        biasMatrices[i]=(biases);
    }
    cout << "Ahoj"<< endl;
}

void Net::forward(const vector<vector<double>> &input) {

    cout << "forw" << endl;
}

void Net::backward(const vector<double> &target) {
    cout << "back" << endl;
}

