using namespace std;

#include "main.h"


int main() {
//    runTests();
//    runBenchmarks();
//    runTraining("..\\..\\data\\fashion_mnist_train_vectors.csv", "..\\..\\data\\fashion_mnist_train_labels.csv",
//                "..\\..\\data\\fashion_mnist_test_vectors.csv","..\\..\\data\\fashion_mnist_test_labels.csv");


    runTraining("..\\..\\data\\fashion_mnist_train_vectors.csv", "..\\..\\data\\fashion_mnist_train_labels.csv",
                "..\\..\\data\\fashion_mnist_test_vectors.csv", "..\\..\\data\\fashion_mnist_test_labels.csv",
                {784, 512, 10}, 0.001, 2000, 50, 5, 1000, 200);

//    runXor(1,1,{2,3,2},0.001);
}
