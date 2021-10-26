//
// Created by marek on 10/26/2021.
//
using namespace std;
#include "Csv.h"
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>


Csv::Csv() {
}

vector<vector<double>> Csv::load(string path_to_file) {
    cout << "loading dataset..." << endl;

    ifstream inputFile(path_to_file);

    if (!inputFile.is_open()){
        throw runtime_error("Could not open file!");
    };

    vector<vector<double>> output;
    string line;

    while(getline(inputFile,line)){
        vector<double> oneLine;

        stringstream ss(line);
        string valString;

        while (getline(ss, valString , ',')){
            oneLine.push_back(stod(valString, nullptr));
        }

        output.push_back(oneLine);
    }
    inputFile.close();

    return output;
}

void Csv::save(string path_to_file, vector<vector<double>> data) {
    cout << "saving dataset..." << endl;

    ofstream outputFile(path_to_file);

    if (!outputFile.is_open()){
        throw runtime_error("Could not open file!");
    };

    vector<vector<double>> output;
    string line;

    for (vector<double> vec : data){
        for (int i = 0; i < vec.size(); ++i) {
            outputFile << vec[i];
            if (i != vec.size()-1){
                outputFile << ",";
            }
            outputFile << endl;
        }
    }
    outputFile.close();
}