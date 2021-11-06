using namespace std;

#include "main.h"


#include <iostream>
#include <vector>


int main(){

    Csv csv = Csv();
    Matrix<double> trainVectors = csv.load("..\\..\\data\\fashion_mnist_train_vectors.csv",2000);
    vector<Matrix<double>> batches = trainVectors.splitToBatches(100);
    Matrix<double> trainLabels = csv.loadOneHot("..\\..\\data\\fashion_mnist_train_labels.csv",2000);
    vector<Matrix<double>> batchesLabels = trainLabels.splitToBatches(100);
//    vector<double> last = trainVectors[trainVectors.size()-1];
//    vector<vector<double>> results{{10},{20},{30}};

//    vector<double> last{results[0]};
//
//    for (int i = 0; i < last.size(); ++i) {
//        cout << last[i];
//    }
//    cout << endl;
//
//    csv.scaleOne(last,255);
//
//    for (int i = 0; i < last.size(); ++i) {
//        cout << last[i];
//    }
//    cout << endl;

//    csv.scaleData(results,30);
//    for (int i = 0; i < results.size(); ++i) {
//        for (int j = 0; j < results[i].size(); ++j) {
//            cout << results[i][j] << " ";
//        }
//    }
//    vector<vector<double>> A{{1.0,0.0,2.0},{-1.0,3.0,1.0}};
//    vector<vector<double>> B{{3,1},{2,1},{1,0}};
//
//
//    Matrix<double> matrixA(2,3,A);
////    (int,int,class std::vector<class std::vector<double,class std::allocator<double> >,class std::allocator<class std::vector<double,class std::allocator<double> > > > const &)
////    matrixA.fillup(2,3,A);
//    matrixA.print();
////
//    Matrix<double> matrixB(3,2,B);
//    matrixB.print();
//
//    Matrix<double> result = matrixA.multiply(matrixB);
//    result.print();
//
//    Matrix<double> result2 = matrixB.multiply(matrixA);
//    result2.print();
//
//    vector<vector<double>> C{{1.0,0.0},{-1.0,3.0}};
//    vector<vector<double>> D{{3,1},{2,1}};
//
//    Matrix<double> matrixC(2,2,C);
//    Matrix<double> matrixD(2,2,D);
//    matrixC.add(matrixD).print();


    Net myNet = Net({784,256,10},1000,0.001);
//    myNet.forward(batches[0]);
//    myNet.results().print();
//    myNet.backward(batchesLabels[0]);
//    myNet.forward(batches[1]);
//    myNet.results().print();
//    myNet.backward(batchesLabels[1]);
    int epochs = 1;

    for (int i = 0; i < epochs; ++i) {
        cout << "***** Epoch: " << i << " start ******"<<endl;
        double sum_losses = 0.0;
        for (int j = 0; j < batchesLabels.size(); ++j) {
            cout << j << " ";
            myNet.forward(batches[j]);
            sum_losses += myNet.backward(batchesLabels[j]);
            cout << "s: "<< sum_losses;
        }
        cout << endl;

//        sum_losses /= batchesLabels.size();

        cout << "Loss: " << sum_losses << endl;
    }




//
//    vector<double> vec({2});
//    vector<double> vec2({1,2,3});
//    vector<double> vec3 = vec * vec2;

}
