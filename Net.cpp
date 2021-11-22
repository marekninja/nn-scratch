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
//    Vdw = 0.0;
//    Sdw = 0.0;
//    Vdb = 0.0;
//    Sdb = 0.0;

    seed = 5;
    weightMatrices.resize(size);
    activations.resize(size);
    innerPotentials.resize(size);
    biasMatrices.resize(size);

    // ADAM
    beta1 = beta_1;
    beta2 = beta_2;
    epsilon = epsilon_v;

    /// All initialized with zeros
    mW.resize(size);     // First moment vectors    -   weights
    vW.resize(size);     // Second moment vectors   -   weights
    mB.resize(size);     // First moment vectors    -   biases
    vB.resize(size);     // Second moment vectors   -   biases


    //init of weights and biases - input neurons do not have it
    /*
     *
     * Weight matrices initialization
     *
     * - bias matrices are initialized to zero (and left that way)
     * - weights are initialized with using namespace std;
     *
     * */
    for (int i = 1; i -1 < arch.size()-1; ++i) {
        Matrix<double> weights(architecture[i],architecture[i-1]);
        Matrix<double> biases(1,architecture[i]);

        // ADAM
        Matrix<double> mW_matrix(architecture[i],architecture[i-1]);
        Matrix<double> vW_matrix(architecture[i],architecture[i-1]);
        Matrix<double> mB_biases(1,architecture[i]);
        Matrix<double> vB_biases(1,architecture[i]);

        if (i < arch.size()-1){
            // Initialize ReLU layers
            kaiming_initializer(weights,seed*i,architecture[i-1],architecture[i]);
        } else {
            // Initialize Softmax layer
            xavier_initializer(weights,seed*i,architecture[i-1],architecture[i]);
        }

        weightMatrices[i] = weights;
        biasMatrices[i]=biases.transpose();

        // ADAM
        mW[i] = mW_matrix;
        vW[i] = vW_matrix;
        mB[i] = mB_biases.transpose();
        vB[i] = vB_biases.transpose();

    }
}


/*
 * Network activation functions
 *
 * Implemented: reLU, softmax
 *
 * */


double Net::relu(const double &example) {
    return example > 0 ? example : 0;
}

double Net::drelu(const double &ex){
    return ex > 0 ? 1 : 0;
}

double Net::leakyRelu(const double &example) {
    const double alpha=0.05;
    return example > 0 ? example : alpha*example;
}

double Net::dleakyRelu(const double &ex) {
    const double alpha=0.05;
    return ex > 0 ? 1 : alpha;
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

//TODO: dava divne vysledky, ako riesit zaporne aktivacie a 0?
//TODO: ako je vobec mozne ze vystupna aktivacia je zaporna/nulova??
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

    for (int i = 0; i < architecture.size(); ++i) {

        if (i == 0) {
            activations[i] = input;
            activations[i] = activations[i].transpose();
        }

        if (i != 0 && i < architecture.size()){

            innerPotentials[i] = weightMatrices[i].multiply(activations[i-1]);
            innerPotentials[i].addToCol(biasMatrices[i]);
            activations[i] = innerPotentials[i];

            if (i < architecture.size()-1){
                activations[i].apply(relu);
            } else {
                activations[i].applySoftmax(softmax);
            }
        }
    }

}

double Net::backward(Matrix<double> &target, int &epoch){

    // Init helper matrices
    Matrix<double> dW;
    Matrix<double> dB;
    Matrix<double> dZ;

    // Derivation of Softmax & Cross Entropy loss
    //  => dLoss/dZ = predictions - truth
    dZ = activations.back();    // <== returns the last element in vector == in this case model's predictions
    dZ.minus(target.transpose());

    for (int i = architecture.size()-1; i > 0 ; --i) {

        dW = dZ.multiply(activations[i-1].transpose());

        dB = dZ;
        dB.flatMeanRows();

        // ADAM
        biasMatrices[i].minus(adam(mB[i], vB[i], dB, beta1, beta2, epsilon, epoch, learningRate));

//        dB.multiplyNum(learningRate);
//        biasMatrices[i].minus(dB);

        // For all hidden layers:
        if (i > 1){
            innerPotentials[i-1].apply(drelu);

            dZ = weightMatrices[i].transpose().multiply(dZ);
            dZ.multiplyCells(innerPotentials[i-1]);
        }

        dW.multiplyNum(1/batchSize);

        // ADAM
        weightMatrices[i].minus(adam(mW[i], vW[i], dW, beta1, beta2, epsilon, epoch, learningRate));

//        dW.multiplyNum(learningRate);
//        weightMatrices[i].minus(dW);

    }

    double loss = batchCrossEntropy(target);
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



