using namespace std;

#include "OperationsThreads.hpp"
// TODO moze sa pouzivat "libka" random na generovanie hodnot z normalneho rozdelenia?
#include <random>
// TODO moze sa pouzivat cmath na funkciu sqrt?
#include <cmath>

Matrix<double> glorot_uniform_initializer(Matrix<double> &weights) {

    // Random values generator
    double sd = sqrt((double)6 / (weights.getNumRows() + weights.getNumCols()));
    random_device rd;  // Will be used to obtain a seed for the random number engine
    mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    uniform_real_distribution<> distrib(-sd, +sd);

    // TODO zmenit hodnoty priamo v danej matici, nie vytvorit novu
    vector<vector<double>> data = weights.getData();

    // Compute
    for (int i = 0; i < weights.getNumRows(); ++i) {
        for (int j = 0; j < weights.getNumCols(); ++j) {
            data[i][j] = distrib(gen);
        }
    }

    return Matrix<double>(weights.getNumRows(), weights.getNumCols(), data);
}