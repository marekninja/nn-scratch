using namespace std;

#include "main.h"


int main() {
//    runTests();
//    runBenchmarks();
//    runTraining("..\\..\\data\\fashion_mnist_train_vectors.csv", "..\\..\\data\\fashion_mnist_train_labels.csv",
//                "..\\..\\data\\fashion_mnist_test_vectors.csv","..\\..\\data\\fashion_mnist_test_labels.csv");



//
    runTraining("..\\..\\data\\fashion_mnist_train_vectors.csv", "..\\..\\data\\fashion_mnist_train_labels.csv",
                "..\\..\\data\\fashion_mnist_test_vectors.csv", "..\\..\\data\\fashion_mnist_test_labels.csv",
                {784, 1024, 10}, 0.1, 60000, 50, 6, 10000, 200);

//    runXor(1000,2,{2,1,2},0.001);
}
