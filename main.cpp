using namespace std;

#include "main.h"


int main(){
    runTests();
    runBenchmarks();
    runTraining("..\\..\\data\\fashion_mnist_train_vectors.csv", "..\\..\\data\\fashion_mnist_train_labels.csv",
                "..\\..\\data\\fashion_mnist_test_vectors.csv","..\\..\\data\\fashion_mnist_test_labels.csv");

}
