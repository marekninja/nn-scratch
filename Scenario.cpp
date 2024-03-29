//
// Created by marek on 11/10/2021.
//

#include "Scenario.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include <chrono>
//#include <cstdlib>


double randomne(const double &example){
    return (double)rand()/RAND_MAX + 1e-15;
}

void runTests(){
    int size = 2;
    vector<vector<double>> data;
    for (int j = 0; j < size; ++j) {
        vector<double> row(size);
        std::transform(row.begin(), row.end(), row.begin(), ::randomne);
        data.push_back(row);
    }
    Matrix<double> mat(size,size,data);
    Matrix<double> mat2(mat);
    Matrix<double>::compare(mat,mat2,"init..");

    Matrix<double> mN = mat.multiplyNaive(mat2);
    Matrix<double> mT = mat.multiply(mat2);

    Matrix<double>::compare(mN,mT,"multiply");

    mN = Matrix<double>(mat);
    mT = Matrix<double>(mat2);
    Matrix<double>::compare(mN,mT,"init flatMeanRows");
    mN = mN.flatMeanRowsAlloc();
    mT.flatMeanRows();
    Matrix<double>::compare(mN,mT,"flatMeanRows");

    mN = Matrix<double>(mat);
    mT = Matrix<double>(mat2);
    Matrix<double>::compare(mN,mT,"ini multiplyCells");

    mN = mN.multiplyCellsAlloc(mat);
    mT.multiplyCells(mat2);
    Matrix<double>::compare(mN,mT,"multiplyCells");

    mN = Matrix<double>(mat);
    mT = Matrix<double>(mat2);
    Matrix<double>::compare(mN,mT,"init multiplyNum");

    mN = mN.multiplyNumAlloc(4.0);
    mT.multiplyNum(4.0);
    Matrix<double>::compare(mN,mT,"multiplyNum");

    mN = Matrix<double>(mat);
    mT = Matrix<double>(mat2);
    Matrix<double>::compare(mN,mT,"init minus");
    mN = mN.minusAlloc(mat);
    mT.minus(mat2);
    Matrix<double>::compare(mN,mT,"minus");

    mN = Matrix<double>(mat);
    mT = Matrix<double>(mat2);
    Matrix<double>::compare(mN,mT,"init addToCol");
    mN = mN.addToColAlloc(mat);
    mT.addToCol(mat2);
    Matrix<double>::compare(mN,mT,"addToCol");
}

void runBenchmarks(){
    for (int i = 64; i <= 2048; i *= 2) {
        cout << "*** Size: "<<i<<" ***\n";
        vector<vector<double>> data;
        for (int j = 0; j < i; ++j) {
            vector<double> row(i,1.0);
            data.push_back(row);
        }
        Matrix<double> mat(i,i,data);
        Matrix<double> mat2(mat);
        auto start = chrono::high_resolution_clock::now();
        Matrix<double> mN = mat.multiplyNaive(mat2);
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
        cout << "Naive:     "<<duration.count() << "\n";

        auto start2 = chrono::high_resolution_clock::now();

        Matrix<double> mT = mat.multiply(mat2);

        auto stop2 = chrono::high_resolution_clock::now();
        auto duration2 = chrono::duration_cast<chrono::microseconds>(stop2 - start2);
        cout << "Threads:   "<< duration2.count() << "\n";

    }

}

void runTraining(const string& trainDataPath,const string& trainLabelsPath, const string& testDataPath, const string& testLabelsPath,
                 const vector<int>& topology, const double&learningRate, const int& lr_change_interval, const int& lr_change_stop_int,
                 const int& trainPart, const int &batchSize,const int&numEpochs, const int& testPart, const int& testBatch){

    auto start3 = chrono::high_resolution_clock::now();

    Csv csv = Csv();
    Scaler sc = Scaler(255);

    Matrix<double> trainVectors = csv.load(trainDataPath,trainPart);
    trainVectors = sc.scale(trainVectors);

    vector<Matrix<double>> batches = trainVectors.splitToBatches(batchSize);

    Matrix<double> trainLabels = csv.loadOneHot(trainLabelsPath,trainPart);
    vector<Matrix<double>> batchesLabels = trainLabels.splitToBatches(batchSize);


    auto start = chrono::high_resolution_clock::now();
    Net myNet = Net(topology,batchSize,learningRate);


    int epochs = numEpochs;
    double learning_rate = learningRate;
    for (int i = 0; i < epochs; ++i) {

        if (i < lr_change_stop_int &&(i % lr_change_interval == 0) && (i != 0)){
            learning_rate /= 10;
            myNet.setLearningRate(learning_rate);
        }
        cout << "***** Epoch: " << i << " start ******"<<endl;
        double sum_losses = 0.0;

        for (int j = 0; j < batchesLabels.size(); ++j) {
            cout << j << " ";
            myNet.forward(batches[j]);

            sum_losses += myNet.backward(batchesLabels[j], i);
//            cout << "s: "<< sum_losses;
        }
        cout << endl;

        sum_losses /= batchesLabels.size();

        cout << "Loss: " << sum_losses << endl;
//        csv.save(".\\results\\train_attempt"+ to_string(i)+".txt",myNet.results());
    }

    vector<int> trainResults;
    for (int j = 0; j < batchesLabels.size(); ++j) {
        cout << j << " ";
        myNet.forward(batches[j]);
        vector<int> out = myNet.results().transpose().getData()[0];
        trainResults.insert(trainResults.end(), out.begin(), out.end());
    }
    Matrix<int> trainRes(1,trainResults.size(), {trainResults});
    trainRes = trainRes.transpose();

    Matrix<double> trainLabelsRes = csv.load(trainLabelsPath,trainPart);
    double trainAcc = myNet.accuracy(trainRes,trainLabelsRes);
    cout << "Accuracy: "<< trainAcc;

    csv.save("../../trainPredictions.txt",trainRes);

    auto stop = chrono::high_resolution_clock::now();
    auto durationMicro = chrono::duration_cast<chrono::microseconds>(stop - start);
    auto durationSec = chrono::duration_cast<chrono::seconds>(stop - start);
    auto durationMin = chrono::duration_cast<chrono::minutes>(stop - start);

    cout << "Training took: "<<durationMicro.count() << " microseconds => "<<durationSec.count()<< " seconds => " << durationMin.count() << " minutes\n";

    Matrix<double> testVectors = csv.load(testDataPath,testPart);
    testVectors = sc.scale(testVectors);

    vector<Matrix<double>> batchesTest = testVectors.splitToBatches(testBatch);
    Matrix<double> testLabels = csv.load(testLabelsPath,testPart);


    vector<int> results;
    auto start2 = chrono::high_resolution_clock::now();
    for (int i = 0; i < testVectors.getNumRows() / testBatch; ++i) {

        myNet.forward(batchesTest[i]);
        vector<int> out = myNet.results().transpose().getData()[0];
        results.insert(results.end(), out.begin(), out.end());
    }
    auto stop2 = chrono::high_resolution_clock::now();
    auto durationMicro2 = chrono::duration_cast<chrono::microseconds>(stop2 - start2);
    auto durationSec2 = chrono::duration_cast<chrono::seconds>(stop2 - start2);
    auto durationMin2 = chrono::duration_cast<chrono::minutes>(stop2 - start2);
    cout << "Eval took: "<<durationMicro2.count() << " microseconds => "<<
         durationSec2.count()<< " seconds => " << durationMin2.count() << " minutes\n";

    Matrix<int> res(1,results.size(), {results});
    res = res.transpose();

    double acc = myNet.accuracy(res,testLabels);
    cout << "Accuracy: "<< acc;


//    csv.save("../results/test1.txt",res);
    csv.save("../../actualTestPredictions.txt",res);

    auto stop3 = chrono::high_resolution_clock::now();
    auto durationSec3 = chrono::duration_cast<chrono::seconds>(stop3 - start3);
    auto durationMin3 = chrono::duration_cast<chrono::minutes>(stop3 - start3);
    cout << "Running time: "<<
         durationSec3.count()<< " seconds => " << durationMin3.count() << " minutes\n";
}

//does not work
void runXor(const int& numEpochs, const int& batchSize, const vector<int>& topology, const double&learningRate) {
    auto start3 = chrono::high_resolution_clock::now();

    vector<vector<double>> vectors = {{0, 0}, {1, 1}, {1, 0}, {0, 1}};
    vector<vector<double>> labelsOneHot = {{1, 0}, {1, 0}, {0, 1}, {0, 1}};

    vector<vector<double>> labels = {{0}, {0}, {1}, {1}};

    Matrix<double> trainVectors = Matrix<double>(4, 2, vectors);

    vector<Matrix<double>> batches = trainVectors.splitToBatches(batchSize);

    Matrix<double> trainLabels = Matrix<double>(4, 2, labelsOneHot);
    vector<Matrix<double>> batchesLabels = trainLabels.splitToBatches(batchSize);

    auto start = chrono::high_resolution_clock::now();
    Net myNet = Net(topology,batchSize,learningRate);


    int epochs = numEpochs;
    double learning_rate = learningRate;
    for (int i = 0; i < epochs; ++i) {

//        if ((i % 3 == 0) && (i != 0)){
//            learning_rate /= 10;
//            myNet.setLearningRate(learning_rate);
//        }
        cout << "***** Epoch: " << i << " start ******"<<endl;
        double sum_losses = 0.0;

        for (int j = 0; j < batchesLabels.size(); ++j) {
            cout << j << " ";
            myNet.forward(batches[j]);

            sum_losses += myNet.backward(batchesLabels[j],i);
//            cout << "s: "<< sum_losses;
        }
        cout << endl;

        sum_losses /= batchesLabels.size();

        cout << "Loss: " << sum_losses << endl;
//        csv.save(".\\labelsOneHot\\train_attempt"+ to_string(i)+".txt",myNet.labelsOneHot());
    }
    auto stop = chrono::high_resolution_clock::now();
    auto durationMicro = chrono::duration_cast<chrono::microseconds>(stop - start);
    auto durationSec = chrono::duration_cast<chrono::seconds>(stop - start);
    auto durationMin = chrono::duration_cast<chrono::minutes>(stop - start);

    cout << "Training took: "<<durationMicro.count() << " microseconds => "<<durationSec.count()<< " seconds => " << durationMin.count() << " minutes\n";



    Matrix<double> testVectors = Matrix<double>(4, 2, vectors);
    vector<Matrix<double>> batchesTest = trainVectors.splitToBatches(batchSize);
    Matrix<double> testLabels = Matrix<double>(4, 1, labels);


    vector<int> results;
    auto start2 = chrono::high_resolution_clock::now();
    for (int i = 0; i < testVectors.getNumRows() / batchSize; ++i) {

        myNet.forward(batchesTest[i]);
        vector<int> out = myNet.results().transpose().getData()[0];
        results.insert(results.end(), out.begin(), out.end());
    }
    auto stop2 = chrono::high_resolution_clock::now();
    auto durationMicro2 = chrono::duration_cast<chrono::microseconds>(stop2 - start2);
    auto durationSec2 = chrono::duration_cast<chrono::seconds>(stop2 - start2);
    auto durationMin2 = chrono::duration_cast<chrono::minutes>(stop2 - start2);
    cout << "Eval took: "<<durationMicro2.count() << " microseconds => "<<
         durationSec2.count()<< " seconds => " << durationMin2.count() << " minutes\n";

    Matrix<int> res(1, results.size(), {results});
    res = res.transpose();

    double acc = myNet.accuracy(res,testLabels);
    cout << "Accuracy: "<< acc;


//    csv.save("..\\labelsOneHot\\test1.txt",res);

    auto stop3 = chrono::high_resolution_clock::now();
    auto durationSec3 = chrono::duration_cast<chrono::seconds>(stop3 - start3);
    auto durationMin3 = chrono::duration_cast<chrono::minutes>(stop3 - start3);
    cout << "Running time: "<<
         durationSec3.count()<< " seconds => " << durationMin3.count() << " minutes\n";
}
