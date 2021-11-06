//
// Created by marek on 10/26/2021.
//

#ifndef PV021_PROJECT_CSV_H
#define PV021_PROJECT_CSV_H
using namespace std;

#include "Operations.hpp"
#include <vector>
#include <string>

class Csv {
public: Csv();
    ///Loads csv file into vector of vectors
    ///Does not support headers
    /// Matrix is [1row x NumExamples Cols]
    Matrix<double> load(const string& path_to_file, const int& part=INT32_MAX);

    Matrix<double> loadOneHot(const string& path_to_file, const int& part=INT32_MAX);

    ///Saves data into csv file
    ///Does not create headers
    void save(string path_to_file,const vector<vector<double>>& data);

    ///Scales dataset by scaleVal
    ///Scaling is division: old/scaleVal=scaledVal
    ///result is in interval [0,255]
    void scaleData(vector<vector<double>> &vector, double scaleVal);

    void scaleOne(vector<double> &vector, double scaleVal);
};


#endif //PV021_PROJECT_CSV_H
