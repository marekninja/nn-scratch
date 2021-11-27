//
// Created by marek on 10/26/2021.
//
//using namespace std;

#include "Csv.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>


Csv::Csv() = default;

Matrix<double> Csv::load(const string& path_to_file, const int& part) {
    cout << "loading dataset... => " << path_to_file << endl;

    ifstream inputFile(path_to_file);

    if (!inputFile.is_open()) {
        throw runtime_error("Could not open file!");
    }

    vector <vector<double>> output;
    string line;

    while (getline(inputFile, line)) {
        vector<double> oneLine;

        stringstream ss(line);
        string valString;

        while (getline(ss, valString, ',')) {
            oneLine.push_back(stod(valString, nullptr));
        }

        output.push_back(oneLine);
        if (output.size()+1 == part+1){
            break;
        }

    }
    inputFile.close();

    Matrix<double> dataset(output.size(),output[0].size(),output);

    return dataset;
}

Matrix<double> Csv::loadOneHot(const string& path_to_file, const int& part) {
    cout << "loading dataset..." << endl;

    ifstream inputFile(path_to_file);

    if (!inputFile.is_open()) {
        throw runtime_error("Could not open file!");
    }

    vector <vector<double>> output;
    string line;

    while (getline(inputFile, line)) {
        vector<double> oneLine(10,0.0);

        stringstream ss(line);
        string valString;

        while (getline(ss, valString, ',')) {
            oneLine[stoi(valString)] = 1;
        }

        output.push_back(oneLine);
        if (output.size()+1 == part+1){
            break;
        }
    }
    inputFile.close();

    Matrix<double> dataset(output.size(),output[0].size(),output);

    return dataset;
}

void Csv::save(const string& path_to_file,const Matrix<int>& data) {
    cout << "saving dataset..." << endl;

    ofstream outputFile(path_to_file);

    if (!outputFile.is_open()) {
        throw runtime_error("Could not open file!");
    }

    vector <vector<int>> output;
    string line;

    for (vector<int> vec: data.getData()) {
        for (int i = 0; i < vec.size(); ++i) {
            outputFile << vec[i];
            if (i != vec.size() - 1) {
                outputFile << ",";
            }
            outputFile << endl;
        }
    }
    outputFile.close();
}


