using namespace std;

#include "kaiming.h"

#include "OperationsThreads.hpp"
#include <random>

void kaiming_initializer(Matrix<double>& weights, const int& seed, const int& incoming, const int& cols) {

    std::mt19937 gen(seed);
    std::normal_distribution<> distrib(0, 1);

    for (int i = 0; i < weights.getNumRows(); ++i) {
        for (int j = 0; j < weights.getNumCols(); ++j) {
            double num = distrib(gen) * sqrt( (double)2/incoming);
            weights(i,j) = num;
        }
    }

}
