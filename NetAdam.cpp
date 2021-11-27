#include "Net.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <random>




Net::Net(const vector<int> &arch, const int &batch_size, const double &learning_rate, double beta_1, double beta_2, double epsilon_v) {
    architecture = arch;
    int size = architecture.size();

    batchSize = batch_size;
    learningRate = learning_rate;

    beta1 = beta_1;
    beta2 = beta_2;
    epsilon = epsilon_v;
    seed = 5;
    epoch = 0;


    weightMatrices.resize(size);
    activations.resize(size);
    innerPotentials.resize(size);
    biasMatrices.resize(size);

    Vdw.resize(size);
    Sdw.resize(size);
    Vdb.resize(size);
    Sdb.resize(size);

    //init of weights and biases - input neurons do not have it
    for (int i = 1; i -1 < arch.size()-1; ++i) {
        Matrix<double> weights(architecture[i],architecture[i-1]);
        Matrix<double> biases(1,architecture[i]);
        if (i < arch.size()-1){
            initRelu(weights,seed*i,architecture[i-1],architecture[i]);
//            bias set to 0 for relu
        } else {
//            init softmax

            initSoftmax(weights,seed*i,architecture[i-1],architecture[i]);
//            bias set to 0 for softmax
        }

        weightMatrices[i] = weights;
        biasMatrices[i]=biases.transpose();
    }

    for (int i = 1; i -1 < arch.size()-1; ++i) {
        Matrix<double> weights1(architecture[i],architecture[i-1]);
        Matrix<double> weights2(architecture[i],architecture[i-1]);
        Matrix<double> biases1(architecture[i],1);
        Matrix<double> biases2(architecture[i],1);
        Vdw[i] = weights1;
        Sdw[i] = weights2;
        Vdb[i] = biases1;
        Sdb[i] = biases2;
    }
}



void Net::initRelu(Matrix<double>& weights, const int& seed, const int& incoming, const int& cols){
//kaiming initialization
//    random(number incoming, cols) * math.sqrt( 2 / number incoming)

    std::mt19937 gen(seed);
    std::normal_distribution<> distrib(0, 1);

    for (int i = 0; i < weights.getNumRows(); ++i) {
        for (int j = 0; j < weights.getNumCols(); ++j) {
            double num = distrib(gen) * sqrt( (double)2/incoming);
            weights(i,j) = num;
        }
    }
//    return (double)rand()/RAND_MAX + 1e-15;
//    return 0;
}

void Net::initSoftmax(Matrix<double>& weights, const int& seed, const int& incoming, const int& cols){
//xavier initialization
//  uniform_random(-1,1) * match.sqrt(6/(incoming*cols)
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> distrib(-1, 1);

    for (int i = 0; i < weights.getNumRows(); ++i) {
        for (int j = 0; j < weights.getNumCols(); ++j) {
            double num = distrib(gen) * sqrt( (double)6/(incoming * cols));
            weights(i,j) = num;
        }
    }
//    double num = distrib(gen) * sqrt( (double)6/(incoming * cols));
//    double num1 = distrib(gen) * sqrt( (double)6/(incoming * cols));
//    double num2 = distrib(gen) * sqrt( (double)6/(incoming * cols));
//    double num3 = distrib(gen) * sqrt( (double)6/(incoming * cols));
//    return num;
//    return (double)rand()/RAND_MAX + 1e-15;
}

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

//TODO: dava divne vysledky, ako riesit zaporne aktivacie a 0?
//TODO: ako je vobec mozne ze vystupna aktivacia je zaporna/nulova??
double Net::batchCrossEntropy(const Matrix<double>& target) {
    double sum = 0;

    for (int i = 0; i < target.getNumRows(); ++i) {
        for (int j = 0; j < activations.back().getNumRows(); ++j) {
//            if (activations.back()(j,i) <= 0){
////                cout << "activation i="<<i<<" j="<<j<<" :"<<activations.back()(j,i)<<"\n";
////                activations.back()(j,i) = 1e-15;
//                sum += target(i,j) * log(1e-15);
//            } else {
//                sum += target(i,j) * log(activations.back()(j,i));
//            }
            sum += target(i,j) * log(activations.back()(j,i));
        }
    }

    return sum * -1/target.getNumRows();
}

//accuracy of one hot
double Net::accuracy(const Matrix<double> &target) {
    int correct = 0;
//    cout<<"activations:\n";
//    activations.back().print();
//    cout << "target:\n";
//    target.print();
    for (int i = 0; i < target.getNumRows(); ++i) {
        for (int j = 0; j < activations.back().getNumRows(); ++j) {
            if (target(i,j) == 1 && activations.back()(j,i) == target(i,j)){
                correct++;
            }
        }
    }
    return ((double)correct/target.getNumRows())*100;
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

    Matrix<double> dZ = activations.back();
    dZ.minus(target.transpose());

    Matrix<double> dW;
    Matrix<double> dB;
    epoch ++;

    for (int i = architecture.size()-1; i > 0 ; --i) {
//        z = y = vysledok aktivacnej f.
//        dz(y=4) = dz(y=5) * deriv_activ4(inner pot. 5) * w(4->5)
//        dw(3->4) = dz(y=4) * deriv_activ4(inner potential 4) *y3


        dW = dZ.multiply(activations[i-1].transpose());
        //TODO: toto je navyse lebo batch size

        dB = dZ;
        dB.flatMeanRows();

        Vdw[i].multiplyNum(beta1);
        Vdw[i] = Vdw[i].add(dW.multiplyNumAlloc(1-beta1));
        Vdb[i].multiplyNum(beta1);
        Vdb[i] = Vdb[i].add(dB.multiplyNumAlloc(1-beta1));

        Sdw[i].multiplyNum(beta2);
        Sdw[i] = Sdw[i].add(dW.multiplyCellsAlloc(dW).multiplyNumAlloc(1-beta2));
        Sdb[i].multiplyNum(beta2);
        Sdb[i] = Sdb[i].add(dB.multiplyCellsAlloc(dB).multiplyNumAlloc(1-beta2));

        //correction
        Matrix<double> Vdw_corr =  Vdw[i].multiplyNumAlloc(1/(1- pow(beta1,epoch)));
        Matrix<double> Vdb_corr = Vdb[i].multiplyNumAlloc(1/(1- pow(beta1,epoch)));

        Matrix<double> Sdw_corr = Sdw[i].multiplyNumAlloc(1/(1-pow(beta2,epoch)));
        Matrix<double> Sdb_corr = Sdb[i].multiplyNumAlloc(1/(1-pow(beta2,epoch)));

        if (i > 1){
            innerPotentials[i-1].apply(drelu);

            dZ = weightMatrices[i].transpose().multiply(dZ);
            dZ.multiplyCells(innerPotentials[i-1]);
        }
//        updates
//        TODO: toto moze byt problem - sqrtl vracia long double
        Sdw_corr.apply(sqrtl);
        Sdw_corr.addNum(epsilon);
        Vdw_corr.divideCells(Sdw_corr);
        Vdw_corr.multiplyNum(learningRate);
        weightMatrices[i].minus(Vdw_corr);

        Sdb_corr.apply(sqrtl);
        Sdb_corr.addNum(epsilon);
        Vdb_corr.divideCells(Sdb_corr);
        Vdb_corr.multiplyNum(learningRate);
        biasMatrices[i].minus(Vdb_corr);

//        // update biases
//        dB.flatMeanRows();
//        dB.multiplyNum(learningRate);
//        biasMatrices[i].minus(dB);
//
//        // update weights
//        dW.multiplyNum(1/batchSize);
//        dW.multiplyNum(learningRate);
//        weightMatrices[i].minus(dW);

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



