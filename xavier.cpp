//
// Created by krcma on 20. 11. 2021.
//

#include "xavier.h"

#include "OperationsThreads.hpp"
#include <random>

void xavier_initializer(Matrix<double>& weights, const int& seed, const int& incoming, const int& cols) {

    std::mt19937 gen(seed);
    std::uniform_real_distribution<> distrib(-1, 1);

    for (int i = 0; i < weights.getNumRows(); ++i) {
        for (int j = 0; j < weights.getNumCols(); ++j) {
            double num = distrib(gen) * sqrt( (double)6/(incoming * cols));
            weights(i,j) = num;
        }
    }

}
