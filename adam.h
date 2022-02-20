//
// Created by krcma on 21. 11. 2021.
//

#ifndef PV021_PROJECT_ADAM_H
#define PV021_PROJECT_ADAM_H

using namespace std;

#include "OperationsThreads.hpp"

Matrix<double> adam(Matrix<double>& mB,
                    Matrix<double>& vB,
                    Matrix<double>& gradients,
                    const double& beta1,
                    const double& beta2,
                    const double& epsilon,
                    const int& epoch,
                    const double& learningRate);

#endif //PV021_PROJECT_ADAM_H
