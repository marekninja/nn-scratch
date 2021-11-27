//using namespace std;

#include "Scaler.h"


/*
 *
 * Constructor
 *
 * */

Scaler::Scaler(const double &scaling_factor) {
    scalingFactor = scaling_factor;
}

/*
 *
 * Functions
 *
 * */

Matrix<double> Scaler::scale(Matrix<double> &matrix) {

    // Declare variable data
    vector<vector<double>> data = matrix.getData();

    for (int i = 0; i < matrix.getNumRows(); ++i) {
        for (int j = 0; j < matrix.getNumCols(); ++j) {
            data[i][j] /= scalingFactor;
        }
    }

    return Matrix<double>(matrix.getNumRows(), matrix.getNumCols(), data);
}
