//
// Created by marek on 11/10/2021.
//

#ifndef PV021_PROJECT_SCENARIO_H
#define PV021_PROJECT_SCENARIO_H

#include "OperationsThreads.hpp"
#include "Csv.h"
#include "Net.h"

void runTests();

void runBenchmarks();

void runTraining(const string& trainDataPath,const string& trainLabelsPath, const string& testDataPath, const string& testLabelsPath,
                 const vector<int>& topology= {784,1024,10}, const double&learning_rate=0.001,
                 const int& trainPart=INT32_MAX, const int &batchSize=50,const int&numEpochs=5, const int& testPart=INT32_MAX, const int& testBatch=200);

void runXor(const vector<int>& topology= {2,1,1}, const double&learning_rate=0.001, const int& batchSize=1);


#endif //PV021_PROJECT_SCENARIO_H
