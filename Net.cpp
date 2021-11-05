#include "Net.h"

using namespace std;
#include <iostream>
#include <algorithm>


Net::Net(const vector<int>& arch, const int& batch_size, const double& learning_rate) {
    architecture = arch;
    int size = architecture.size();

    batchSize = batch_size;
    learningRate = learning_rate;

    weightMatrices.resize(size);
    activations.resize(size);
    biasMatrices.resize(size);

    //init of weights and biases - input neurons do not have it
    for (int i = 1; i -1 < arch.size()-1; ++i) {
        cout << "layer i:" << i << "=" << arch[i]<<endl;
        Matrix<double> weights(architecture[i-1],architecture[i]);
        weightMatrices[i]=weights;

        Matrix<double> biases(1,architecture[i]);
        biasMatrices[i]=biases;

//        for (int j = 0; j < architecture[i]; ++j) {
//            activations(j,1).resize(batch_size,0.0);
//        }
//        activationMatrices[i]=activations;
    }

    //init of activations - input layer activation is input into network
//    for (int i = 0; i < arch.size(); ++i) {
//        activations[i].resize(architecture[i]);
//        for (int j = 0; j < architecture[i]; ++j) {
//            vector<double> batch_examples({});
//            batch_examples.resize(batch_size,0.0);
//            activations[i][j] = batch_examples;
//        }
//    }
    cout << "Ahoj"<< endl;
}

double Net::relu(const double &example) {
    return 0 > example ? example : 0;
}



void Net::forward(const Matrix<double> &input) {
    cout << "forw" << endl;
    input.printShape();

    for (int i = 0; i < architecture.size(); ++i) {
        if (i == 0){
            activations[i] = input;
        }

        if (i != 0 && i < architecture.size()){
            activations[i] = activations[i-1].multiply(weightMatrices[i]).addToRow(biasMatrices[i]);
            activations[i].apply(relu);
        }
    }

//    weightMatrices[1].printShape();
//    biasMatrices[1].print();
//    biasMatrices[1].printShape();
//    activations[1].print();
//    activations[1].printShape();
}

void Net::backward(const vector<double> &target) {
    cout << "back" << endl;

}


