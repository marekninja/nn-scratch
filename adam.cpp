//
// Created by krcma on 21. 11. 2021.
//

#include "adam.h"
#include <cmath>

Matrix<double> adam(Matrix<double>& m,
                    Matrix<double>& v,
                    Matrix<double>& gradients,
                    const double& beta1,
                    const double& beta2,
                    const double& epsilon,
                    const int& epoch,
                    const double& learningRate) {

    /*
     * ADAM optimizer
     *
     * source: https://arxiv.org/pdf/1412.6980v9.pdf
     *
     * */

    // TODO naive implementations used for most matrix operations (change for threads?)

    // mt ← β1 ·mt−1 + (1 −β1) ·gt (Update biased first moment estimate)
    m.multiplyNum(beta1);
//    m = m.addAlloc(gradients.multiplyNumAlloc((1-beta1)));
    m.add(gradients.multiplyNumAlloc((1-beta1)));

    // vt ← β2 ·vt−1 + (1 −β2) ·g2t (Update biased second raw moment estimate)
    v.multiplyNum(beta2);
//    v = v.addAlloc(gradients.matrixPowAlloc(2).multiplyNumAlloc((1-beta2)));
    v.add(gradients.matrixPowAlloc(2).multiplyNumAlloc((1-beta2)));

    // αt = α ·√(1 −βt2)/(1 −βt1)
    double stepSize = learningRate * (sqrt((1-pow(beta2, (epoch+1)))) / (1 - pow(beta1, (epoch+1))));

    // return (αt ·mt)/(√vt + epsilon)
//    return m.multiplyNumAlloc(stepSize).devideNaiveAlloc(v.matrixSqrtAlloc().addConstantAlloc(epsilon));

    return m.multiplyNumAlloc(stepSize).devideNaiveAlloc(v.matrixSqrtAlloc().addNumAlloc(epsilon));

}