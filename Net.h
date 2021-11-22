//
// Created by marek on 10/25/2021.
//

#ifndef SRC_NET_H
#define SRC_NET_H

using namespace std;

#include <vector>
#include "OperationsThreads.hpp"

#include "xavier.h"
#include "kaiming.h"
#include "adam.h"


class Net {
private:
    ///network architecture {1,1,1} means 1 input neuron, 1 hidden, 1 output
    vector<int> architecture;
    double batchSize;
    double learningRate;
    int seed;

    ///vector of matrices of (incoming) weights
    vector<Matrix<double>> weightMatrices;

    ///vector of matrices of biases of neurons
    ///weight Matrix
    ///numRows = number of neurons in previous layer
    ///numCols = number of neurons in actual layer
    vector<Matrix<double>> biasMatrices;

    ///vector of activations of forward pass of network
    ///this depends also on batch-size
    ///for each layer, for each example,
    /// Matrix is [NumNeurons x Example Cols]
    /// 1 row = 1 neuron on layer
    ///     examples
    ///n1 0.3 0.4 0.6 ...
    ///n2
    ///n3
    ///...
    vector<Matrix<double>> activations;

    ///same as activations, but without activation function applied
    /// only for storage purps.
    vector<Matrix<double>> innerPotentials;

//    double Vdw = 0.0;
//    double Sdw = 0.0;
//    double Vdb = 0.0;
//    double Sdb = 0.0;

    /// ADAM
    double beta1;
    double beta2;
    double epsilon;

    vector<Matrix<double>> mW;
    vector<Matrix<double>> vW;
    vector<Matrix<double>> mB;
    vector<Matrix<double>> vB;

    static double random(const double &example);

    static double scale(const double &example);

    // Activation functions
    static double relu(const double &example);
    static double drelu(const double &ex);
    static double leakyRelu(const double &example);
    static double dleakyRelu(const double &ex);
    static void softmax(vector<double> &output);
    static void dsoftmax(vector<double> &output);


public:

    /*
     * Neural Net constructor
     *
     * Input is network architecture:
     * {1,1,1} means 1 input neuron, 1 hidden, 1 output
     *
     * hidden neurons use ReLU activations
     * output is SoftMax
     */
    Net(const vector<int> &arch,
        const int &batch_size,
        const double &learning_rate,
        double beta_1=0.9,
        double beta_2=0.999,
        double epsilon_v=0.00000008);

    /*
     * Forward function
     *
     * Takes input BATCH of training examples
     * Matrix:
     *      vals
     * e1 1 2 0 3
     * e2 0 4 0 0
     * e3 ...
     */

    void forward(const Matrix<double> &input);

    /*
     * Backwards pass - defines gradient descent
     *
     * Takes input batch of target values one-hot encoded
     *      classes
     * ex1 0 0 0 1 0 0 0 0 0
     * ex2 1 ...
     * ex3 ...
     */
    double backward(Matrix<double> &target, int &epoch);

    /*
     * Returns predictions of network
     * row = example
     * col = class
     *
     * format:
     *      class
     * e1   3
     * e2   1
     * e3   0
     * e4   ...
     */
    Matrix<int> results();

    void setLearningRate(const double& learning_rate);
    double batchCrossEntropy(const Matrix<double>& target);
    double accuracy(const Matrix<double> &target);
    static double accuracy(const Matrix<int>& result, const Matrix<double> &target);
};


#endif //SRC_NET_H
