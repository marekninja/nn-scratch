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

    ///vector of matrices of (incoming) weights
    vector<Matrix<double>> weightMatrices;

    ///vector of matrices of biases of neurons
    vector<Matrix<double>> biasMatrices;

    ///vector of activations of forward pass of network
    ///this depends also on batch-size
    vector<Matrix<double>> activationMatrices;

public:
//    topology
// batch size treba implementovat

    /*
     * Neural Net constructor
     *
     * Input is network architecture:
     * {1,1,1} means 1 input neuron, 1 hidden, 1 output
     */
    Net(const vector<int>& arch);

    /*
     * Forward function
     *
     * Takes as input BATCH of training examples
     */
    void forward(const vector<vector<double>>& input);

    /*
     * Backwards pass - defines gradient descent
     *
     * Takes as input batch of target values
     */
    void backward(const vector<double>& target);
};


#endif //SRC_NET_H
