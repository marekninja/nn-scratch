using namespace std;

#include "main.h"

#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <algorithm>

double random(const double &example){
    return (double)rand()/RAND_MAX + 1e-15;
}

//void runTests(){
//    int size = 2;
//    vector<vector<double>> data;
//    for (int j = 0; j < size; ++j) {
//        vector<double> row(size);
//        std::transform(row.begin(), row.end(), row.begin(), random);
//        data.push_back(row);
//    }
//    Matrix<double> mat(size,size,data);
//    Matrix<double> mat2(mat);
//    Matrix<double>::compare(mat,mat2,"init..");
//
//    Matrix<double> mN = mat.multiplyNaive(mat2);
//    Matrix<double> mT = mat.multiply(mat2);
//
//    Matrix<double>::compare(mN,mT,"multiply");
//
//    mN = Matrix<double>(mat);
//    mT = Matrix<double>(mat2);
//    Matrix<double>::compare(mN,mT,"init flatMeanRows");
//    mN = mN.flatMeanRowsAlloc();
//    mT.flatMeanRows();
//    Matrix<double>::compare(mN,mT,"flatMeanRows");
//
//    mN = Matrix<double>(mat);
//    mT = Matrix<double>(mat2);
//    Matrix<double>::compare(mN,mT,"ini multiplyCells");
//
//    mN = mN.multiplyCellsAlloc(mat);
//    mT.multiplyCells(mat2);
//    Matrix<double>::compare(mN,mT,"multiplyCells");
//
//    mN = Matrix<double>(mat);
//    mT = Matrix<double>(mat2);
//    Matrix<double>::compare(mN,mT,"init multiplyNum");
//
//    mN = mN.multiplyNumAlloc(4.0);
//    mT.multiplyNum(4.0);
//    Matrix<double>::compare(mN,mT,"multiplyNum");
//
//    mN = Matrix<double>(mat);
//    mT = Matrix<double>(mat2);
//    Matrix<double>::compare(mN,mT,"init minus");
//    mN = mN.minusAlloc(mat);
//    mT.minus(mat2);
//    Matrix<double>::compare(mN,mT,"minus");
//
//    mN = Matrix<double>(mat);
//    mT = Matrix<double>(mat2);
//    Matrix<double>::compare(mN,mT,"init addToCol");
//    mN = mN.addToColAlloc(mat);
//    mT.addToCol(mat2);
//    Matrix<double>::compare(mN,mT,"addToCol");
//
//
//
////    Matrix<double> matrixA(2,3,A);
//////////    (int,int,class std::vector<class std::vector<double,class std::allocator<double> >,class std::allocator<class std::vector<double,class std::allocator<double> > > > const &)
//////    matrixA.fillup(2,3,A);
////    matrixA.print();
//////////
////    Matrix<double> matrixB(3,2,B);
////    cout << "\n";
////    matrixB.print();
////////
////    cout << endl;
////    matrixA.multiplyNaive(matrixB).print();
////
////    cout << endl;
////
////    matrixA.multiply(matrixB).print();
////
////
////    vector<vector<double>> C{{1.0,0.0},{-1.0,3.0}};
////    vector<vector<double>> D{{3,1},{2,1}};
////
////    Matrix<double> matrixC(2,2,C);
////    Matrix<double> matrixD(2,2,D);
////    matrixC.add(matrixD).print();
//}
//
//void runBenchmarks(){
//    for (int i = 64; i <= 2048; i *= 2) {
//        cout << "*** Size: "<<i<<" ***\n";
//        vector<vector<double>> data;
//        for (int j = 0; j < i; ++j) {
//            vector<double> row(i,1.0);
//            data.push_back(row);
//        }
//        Matrix<double> mat(i,i,data);
//        Matrix<double> mat2(mat);
//        auto start = chrono::high_resolution_clock::now();
//        Matrix<double> mN = mat.multiplyNaive(mat2);
//        auto stop = chrono::high_resolution_clock::now();
//        auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
//        cout << "Naive:     "<<duration.count() << "\n";
//
//        auto start2 = chrono::high_resolution_clock::now();
//
//        Matrix<double> mT = mat.multiply(mat2);
//
//        auto stop2 = chrono::high_resolution_clock::now();
//        auto duration2 = chrono::duration_cast<chrono::microseconds>(stop2 - start2);
//        cout << "Threads:   "<< duration2.count() << "\n";
//
//    }
//
//}

void runTraining(const int& trainPart, const int &batchSize,const int&numEpochs, const int& testPart, const int& testBatch){

    auto start3 = chrono::high_resolution_clock::now();

    /*
     * Load training data
     * */

    Csv csv = Csv();
    Matrix<double> trainVectors = csv.load("../data/fashion_mnist_train_vectors.csv", trainPart);
    Matrix<double> trainLabels = csv.loadOneHot("../data/fashion_mnist_train_labels.csv", trainPart);

    cout << "- training data loaded ==> ";

    //  Preprocess training data
    //  - scale image vectors by /255

    Scaler sc = Scaler(255);
    trainVectors = sc.scale(trainVectors);

    cout << "scaled ==> ";

    vector<Matrix<double>> batches = trainVectors.splitToBatches(batchSize);
    vector<Matrix<double>> batchesLabels = trainLabels.splitToBatches(batchSize);

    cout << "split to batches" << endl;

    auto start = chrono::high_resolution_clock::now();

    Net myNet = Net(
            {784,512,10},
            batchSize,
            0.001);

    cout << "model initialized" << endl;

    int epochs = numEpochs;
    double learning_rate = 0.01;

    for (int i = 0; i < epochs; ++i) {

        // TODO zmenit za "change lr on plateau"
        if (i % 27 == 0){
            learning_rate /= 10;
            myNet.setLearningRate(learning_rate);
            cout << "Changin lr to " << learning_rate << endl;
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
//        csv.save("./results/train_attempt"+ to_string(i)+".txt",myNet.results());
    }
    auto stop = chrono::high_resolution_clock::now();
    auto durationMicro = chrono::duration_cast<chrono::microseconds>(stop - start);
    auto durationSec = chrono::duration_cast<chrono::seconds>(stop - start);
    auto durationMin = chrono::duration_cast<chrono::minutes>(stop - start);



    cout << "Training took: "<<durationMicro.count() << " microseconds => "<<durationSec.count()<< " seconds => " << durationMin.count() << " minutes\n";

    Matrix<double> testVectors = csv.load("../data/fashion_mnist_test_vectors.csv",testPart);

    testVectors = sc.scale(testVectors);

    vector<Matrix<double>> batchesTest = testVectors.splitToBatches(testBatch);
//    csv.scaleData(trainVectors,255);
    Matrix<double> testLabels = csv.load("../data/fashion_mnist_test_labels.csv",testPart);
//    vector<Matrix<double>> batchesTestLabels = trainVectors.splitToBatches(100);

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
//    Matrix<int> res(1,results.size(), {results});
    res = res.transpose();

    double acc = myNet.accuracy(res,testLabels);
    cout << "Accuracy: "<< acc << endl;


//    csv.scaleData(trainLabels);

//    csv.save("../results/test1.txt",res);

    auto stop3 = chrono::high_resolution_clock::now();
    auto durationSec3 = chrono::duration_cast<chrono::seconds>(stop3 - start3);
    auto durationMin3 = chrono::duration_cast<chrono::minutes>(stop3 - start3);
    cout << "Running time: "<<
         durationSec3.count()<< " seconds => " << durationMin3.count() << " minutes\n";


//    double acc = myNet.accuracy(testLabels);
//    cout << "Accuracy: "<< acc;


//
//    vector<double> vec({2});
//    vector<double> vec2({1,2,3});
//    vector<double> vec3 = vec * vec2;
}


int main(){
    runTraining(60000,200,10,10000, 1000);
}
