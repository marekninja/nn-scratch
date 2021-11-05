//
// Created by marek on 10/25/2021.
//

#ifndef SRC_NET_H
#define SRC_NET_H

using namespace std;
#include <vector>
#include "Operations.hpp"


class Net {
private:
    ///network architecture {1,1,1} means 1 input neuron, 1 hidden, 1 output
    vector<int> architecture;
    double batchSize;
    double learningRate;

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
    /// Matrix is [1row x NumNeurons Cols]
    /// 1 row = 1 example
    vector<Matrix<double>> activations;

    static double relu(const double& example);


public:
//    topology
// batch size treba implementovat

    /*
     * Neural Net constructor
     *
     * Input is network architecture:
     * {1,1,1} means 1 input neuron, 1 hidden, 1 output
     */
    Net(const vector<int>& arch, const int& batch_size, const double& learning_rate);

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
    void forward(const Matrix<double>& input);

    /*
     * Backwards pass - defines gradient descent
     *
     * Takes as input batch of target values
     */
    void backward(const vector<double>& target);
};


#endif //SRC_NET_H
