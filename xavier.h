//
// Created by krcma on 20. 11. 2021.
//

#ifndef PV021_PROJECT_XAVIER_H
#define PV021_PROJECT_XAVIER_H

#include "OperationsThreads.hpp"

void xavier_initializer(Matrix<double>& weights, const int& seed, const int& incoming, const int& cols);

#endif //PV021_PROJECT_XAVIER_H
