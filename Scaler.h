#ifndef PV021_PROJECT_SCALER_H
#define PV021_PROJECT_SCALER_H

using namespace std;

#include "OperationsThreads.hpp"

class Scaler {
private:
    double scalingFactor;

public:

    /*
     * Constructor
     * */
    Scaler(const double &sf);

    Matrix<double> scale(Matrix<double> &matrix);
};

#endif //PV021_PROJECT_SCALER_H
