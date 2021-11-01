//
// Created by marek on 10/26/2021.
//

#ifndef PV021_PROJECT_CSV_H
#define PV021_PROJECT_CSV_H
using namespace std;

#include <vector>
#include <string>

class Csv {
public: Csv();
    ///Loads csv file into vector of vectors
    ///Does not support headers
    vector<vector<double>> load(string path_to_file);

    ///Saves data into csv file
    ///Does not create headers
    void save(string path_to_file, vector<vector<double>> data);

    void scaleData(vector<vector<double>> &vector, double scaleVal);

    void scaleOne(vector<double> &vector, double scaleVal);
};


#endif //PV021_PROJECT_CSV_H
