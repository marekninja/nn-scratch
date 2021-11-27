#include "main.h"


int main() {
//    runTests();
//    runBenchmarks();
//    runTraining("..\\..\\data\\fashion_mnist_train_vectors.csv", "..\\..\\data\\fashion_mnist_train_labels.csv",
//                "..\\..\\data\\fashion_mnist_test_vectors.csv","..\\..\\data\\fashion_mnist_test_labels.csv");

//    runTraining("..\\..\\data\\mnist_train_vector.csv", "..\\..\\data\\mnist_train_labels.csv",
//                "..\\..\\data\\mnist_test_vector.csv", "..\\..\\data\\mnist_test_labels.csv",
//                {784, 512, 256, 64, 10}, 0.1,2 ,7,
//                60000, 50, 6, 10000, 200);

//
    runTraining("..\\..\\data\\fashion_mnist_train_vectors.csv", "..\\..\\data\\fashion_mnist_train_labels.csv",
                "..\\..\\data\\fashion_mnist_test_vectors_mod.csv", "..\\..\\data\\fashion_mnist_test_labels.csv",
//                {784, 512, 256, 10}, 0.01,3 ,7,
                {784, 512, 256, 10}, 0.01,3 ,7,
                60000, 50, 0, 10000, 200);



//    runTraining("..\\..\\data\\fashion_mnist_train_vectors.csv", "..\\..\\data\\fashion_mnist_train_labels.csv",
//                "..\\..\\data\\fashion_mnist_test_vectors.csv", "..\\..\\data\\fashion_mnist_test_labels.csv",
//                {784, 512, 256, 64, 10}, 0.1,4 ,7,
//                2000, 50, 5, 10000, 200);


//    runTraining("../data/fashion_mnist_train_vectors.csv", "../data/fashion_mnist_train_labels.csv",
//                "../data/fashion_mnist_test_vectors.csv", "../data/fashion_mnist_test_labels.csv",
//                {784, 1024, 10}, 0.1, 60000, 50, 6, 10000, 200);

//    runXor(1000,2,{2,1,2},0.001);
}
