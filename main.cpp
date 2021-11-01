using namespace std;

#include "main.h"


#include <iostream>
#include <vector>


int main(){

//    Csv csv = Csv();
//    vector<vector<double>> trainVectors = csv.load("..\\..\\data\\fashion_mnist_train_vectors.csv");
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
    std::vector<std::vector<double>> A{{1.0,0.0,2.0},{-1.0,3.0,1.0}};
    vector<vector<double>> B{{3,1},{2,1},{1,0}};


    Matrix<double> matrixA(2,3,A);
//    (int,int,class std::vector<class std::vector<double,class std::allocator<double> >,class std::allocator<class std::vector<double,class std::allocator<double> > > > const &)
//    matrixA.fillup(2,3,A);
    matrixA.print();
//
    Matrix<double> matrixB(3,2,B);
    matrixB.print();

    Matrix<double> result = matrixA.multiply(matrixB);
    result.print();

    Matrix<double> result2 = matrixB.multiply(matrixA);
    result2.print();




    Net myNet = Net();
    cout << "Ahoj!";
}
