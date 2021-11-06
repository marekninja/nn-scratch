using namespace std;

#include "main.h"


#include <iostream>
#include <vector>
#include <chrono>


int main(){

    Csv csv = Csv();
    Matrix<double> trainVectors = csv.load("..\\..\\data\\fashion_mnist_train_vectors.csv",6000);
//    csv.scaleData(trainVectors,255);
    vector<Matrix<double>> batches = trainVectors.splitToBatches(50);

    Matrix<double> trainLabels = csv.loadOneHot("..\\..\\data\\fashion_mnist_train_labels.csv",6000);
//    csv.scaleData(trainLabels);
    vector<Matrix<double>> batchesLabels = trainLabels.splitToBatches(50);

//    vector<vector<double>> A{{1.0,0.0,2.0},{-1.0,3.0,1.0}};
//    vector<vector<double>> B{{3,1},{2,1},{1,0}};
//////
//////
//    Matrix<double> matrixA(2,3,A);
////////    (int,int,class std::vector<class std::vector<double,class std::allocator<double> >,class std::allocator<class std::vector<double,class std::allocator<double> > > > const &)
////    matrixA.fillup(2,3,A);
//    matrixA.print();
////////
//    Matrix<double> matrixB(3,2,B);
//    cout << "\n";
//    matrixB.print();
//////
//    Matrix<double> result = matrixA.multiply(matrixB);
//    cout << "\n mult: \n";
//    result.print();
//    cout << "\n";
//
//    matrixA.multiplyThreads(matrixB).print();


//    vector<vector<double>> C{{1.0,0.0},{-1.0,3.0}};
//    vector<vector<double>> D{{3,1},{2,1}};

//    Matrix<double> matrixC(2,2,C);
//    Matrix<double> matrixD(2,2,D);
//    matrixC.add(matrixD).print();



    auto start = chrono::high_resolution_clock::now();
    Net myNet = Net({784,1024,10},100,0.01);
//    myNet.forward(batches[0]);
//    myNet.results().print();
//    myNet.backward(batchesLabels[0]);
//    myNet.forward(batches[1]);
//    myNet.results().print();
//    myNet.backward(batchesLabels[1]);
    int epochs = 5;
    double learning_rate = 0.01;
    for (int i = 0; i < epochs; ++i) {

        if (i % 3 == 0){
            learning_rate /= 10;
            myNet.setLearningRate(learning_rate);
        }
        cout << "***** Epoch: " << i << " start ******"<<endl;
        double sum_losses = 0.0;

        for (int j = 0; j < batchesLabels.size(); ++j) {
            cout << j << " ";
            myNet.forward(batches[j]);

            sum_losses += myNet.backward(batchesLabels[j]);
//            cout << "s: "<< sum_losses;
        }
        cout << endl;

        sum_losses /= batchesLabels.size();

        cout << "Loss: " << sum_losses << endl;
//        csv.save(".\\results\\train_attempt"+ to_string(i)+".txt",myNet.results());
    }
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);

    cout << "Training took: "<<duration.count() << endl;

    Matrix<double> testVectors = csv.load("..\\..\\data\\fashion_mnist_test_vectors.csv");
    vector<Matrix<double>> batchesTest = testVectors.splitToBatches(100);
//    csv.scaleData(trainVectors,255);
    Matrix<double> testLabels = csv.load("..\\..\\data\\fashion_mnist_test_labels.csv");
//    vector<Matrix<double>> batchesTestLabels = trainVectors.splitToBatches(100);

    vector<int> results;
    for (int i = 0; i < 100; ++i) {

        myNet.forward(batchesTest[i]);
        vector<int> out = myNet.results().transpose().getData()[0];
        results.insert(results.end(), out.begin(), out.end());
    }
    Matrix<int> res(1,10000, {results});
    res = res.transpose();

    double acc = myNet.accuracy(res,testLabels);
    cout << "Accuracy: "<< acc;


//    csv.scaleData(trainLabels);

    csv.save("..\\results\\test1.txt",res);

//    double acc = myNet.accuracy(testLabels);
//    cout << "Accuracy: "<< acc;


//
//    vector<double> vec({2});
//    vector<double> vec2({1,2,3});
//    vector<double> vec3 = vec * vec2;

}
