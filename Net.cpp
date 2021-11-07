using namespace std;
#include "Net.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#pragma GCC optimize("Ofast")



Net::Net(const vector<int> &arch, const int &batch_size, const double &learning_rate, double beta_1, double beta_2, double epsilon_v) {
    architecture = arch;
    int size = architecture.size();

    batchSize = batch_size;
    learningRate = learning_rate;
    Vdw = 0.0;
    Sdw = 0.0;
    Vdb = 0.0;
    Sdb = 0.0;
    beta1 = beta_1;
    beta2 = beta_2;
    epsilon = epsilon_v;

    weightMatrices.resize(size);
    activations.resize(size);
    innerPotentials.resize(size);
    biasMatrices.resize(size);

    //init of weights and biases - input neurons do not have it
    for (int i = 1; i -1 < arch.size()-1; ++i) {
        Matrix<double> weights(architecture[i],architecture[i-1]);
        Matrix<double> biases(1,architecture[i]);
        if (i < arch.size()-1){
            weights.apply(random);
            biases.apply(random);
        } else {
            weights.apply(random);
            biases.apply(random);
        }

        weightMatrices[i] = weights;
        biasMatrices[i]=biases.transpose();
    }
}

double Net::random(const double &example){
    return (double)rand()/RAND_MAX + 1e-15;
}

//double Net::randomRelu(const double &example){
//    return (double)rand()/RAND_MAX + 1e-15;
//}
//
//double Net::randomSoft(const double &example){
//    return (double)rand()/RAND_MAX + 1e-15;
//}
//TODO: pozri ci je to spravne
double Net::relu(const double &example) {
    return example > 0 ? example : 0;
}

double Net::scale(const double &example) {
    return example / (double)255;
}
//TODO: pozri ci je to spravne
double Net::drelu(const double &ex){
    return ex > 0 ? 1 : 0;
}

void Net::softmax(vector<double>& output) {
    double sum = 0;

    double max = 0;
    for (int i = 0; i < output.size(); ++i) {
        if (output[i] > max){
            max = output[i];
        }
    }

    for (int i = 0; i < output.size(); ++i) {
        output[i] -= max;
        sum += exp(output[i]);

    }

    for (int i = 0; i < output.size(); ++i) {
        output[i] = exp(output[i])/sum;
    }
}

void Net::dsoftmax(vector<double>& output) {
    double sum = 0;
    for (int i = 0; i < output.size(); ++i) {
        output[i] = output[i] *( 1 - output[i]);
    }
}

double Net::batchCrossEntropy(const Matrix<double>& target) {
    double sum = 0;

    for (int i = 0; i < target.getNumRows(); ++i) {
        for (int j = 0; j < activations.back().getNumRows(); ++j) {
            if (activations.back()(j,i) <= 0){
//                cout << "activation i="<<i<<" j="<<j<<" :"<<activations.back()(j,i)<<"\n";
//                activations.back()(j,i) = 1e-15;
                sum += target(i,j) * log(1e-15);
            } else {
                sum += target(i,j) * log(activations.back()(j,i));
            }
//            sum += target(i,j) * activations.back()(j,i);
        }
    }

    return sum * -1/target.getNumRows();
}

double Net::accuracy(const Matrix<int> &result, const Matrix<double> &target) {
    if (result.getNumRows() != target.getNumRows()){
        throw runtime_error("Can not compare, different numRows !");
    }
    if (result.getNumCols() != target.getNumCols()){
        throw runtime_error("Can not compare, different numCols !");
    }
    double correct = 0;
    for (int i = 0; i < result.getNumRows(); ++i) {
        if (result(i,0) == target(i,0)){
            correct++;
        }
    }
    return correct / result.getNumRows();
}

double Net::accuracy(const Matrix<double> &target) {
    Matrix<int> result = results();
    int correct = 0;
    for (int i = 0; i < result.getNumRows(); ++i) {
        for (int j = 0; j < target.getNumCols(); ++j) {
            if (target(i,j) == 1 && result(i,0) == j){
                correct++;
            }
        }
    }

    return ((double)correct/result.getNumRows())*100;
}

void Net::forward(const Matrix<double> &input) {
//    cout << "forw" << endl;
//    input.printShape();

    for (int i = 0; i < architecture.size(); ++i) {
        if (i == 0){
//            cout << "input" <<endl;
            activations[i] = input;
            activations[i] = activations[i].transpose();
            activations[i].apply(scale);
        }

        if (i != 0 && i < architecture.size()){

//            TODO: mozno toto treba naopak nasobit, teda w * activation, a nie activation * w
            innerPotentials[i] = weightMatrices[i]
                    .multiply(activations[i-1]);
            innerPotentials[i].addToCol(biasMatrices[i]);
//            innerPotentials[i] = weightMatrices[i]
//                    .multiply(activations[i-1]).addToColAlloc(biasMatrices[i]);
//            innerPotentials[i].addToCol(biasMatrices[i]);

            activations[i] = innerPotentials[i];


            if (i < architecture.size()-1){
                activations[i].apply(relu);
            } else {
//                activations[i].apply(softmax);
//                activations[i] = activations[i].transpose().applySoftmax(softmax);
                activations[i].applySoftmax(softmax);
            }
        }
    }
//    activations.back().print();
}

//attempt at ADAM optimizer
double Net::backward(Matrix<double> &target){
//    cout << "back" << endl;
//    cout<< "L: " << loss<<" ";

//    Matrix<double> dZ = activations.back().minusAlloc(target.transpose());
    Matrix<double> dZ = activations.back();
    dZ.minus(target.transpose());

    Matrix<double> dW;
    Matrix<double> dB;

    for (int i = architecture.size()-1; i > 0 ; --i) {
//        z = y = vysledok aktivacnej f.
//        dz(y=4) = dz(y=5) * deriv_activ4(inner pot. 5) * w(4->5)
//        dw(3->4) = dz(y=4) * deriv_activ4(inner potential 4) *y3

//        dW = dZ.multiply(activations[i-1].transpose());
        dW = dZ.multiply(activations[i-1].transpose());
        //TODO: toto je navyse lebo batch size
        dW.multiplyNum(1/batchSize);
//        dW = activations[i-1].transpose().multiply(dZ);
        dB = dZ;


//        weightMatrices[i] = weightMatrices[i].minusAlloc(dW.multiplyNum(learningRate));
//        weightMatrices[i].minus(dW.multiplyNumAlloc(learningRate));
        dW.multiplyNum(learningRate);
        weightMatrices[i].minus(dW);
//        biasMatrices[i] = biasMatrices[i].minusAlloc(dB.flatMeanRows().multiplyNum(learningRate));
//        biasMatrices[i].minus(dB.flatMeanRowsAlloc().multiplyNum(learningRate));
        dB.flatMeanRows();
//        biasMatrices[i].minus(dB.multiplyNumAlloc(learningRate));
        dB.multiplyNum(learningRate);
        biasMatrices[i].minus(dB);

        if (i > 1){
            innerPotentials[i-1].apply(drelu);
//            dZ = weightMatrices[i].transpose().multiply(dZ).multiplyCellsAlloc(innerPotentials[i-1]);
            dZ = weightMatrices[i].transpose().multiply(dZ);
            dZ.multiplyCells(innerPotentials[i-1]);
        }
    }
    double loss = batchCrossEntropy(target);
//    cout<< "L: " << loss<<" ";
//    double acc = accuracy(target);
//    cout<< "A: " <<acc<<" " << endl;
    return loss;
}



Matrix<int> Net::results() {
//    activations.back().print();
    Matrix<int> results(activations.back().getNumCols(),1);
    for (int i = 0; i < activations.back().getNumCols(); ++i) {
        int max_n = 0;
        double max_v = 0;
        for (int j = 0; j < activations.back().getNumRows(); ++j) {
            if (activations.back()(j,i) > max_v){
                max_n = j;
                max_v = activations.back()(j,i);
            }
        }
        results(i,0) = max_n;
    }
    return results;
}

void Net::setLearningRate(const double &learning_rate) {
    learningRate = learning_rate;
}



