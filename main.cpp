using namespace std;

#include "main.h"
#include "Net.h"
#include "Csv.h"

#include <iostream>
#include <vector>


int main(){

    Csv csv = Csv();
    vector<vector<double>> trainVectors = csv.load("..\\..\\data\\fashion_mnist_train_vectors.csv");
    vector<double> last = trainVectors[trainVectors.size()-1];
    vector<vector<double>> results{{10},{20},{30}};


    csv.save("..\\results\\test1.txt", results);

    Net myNet = Net();
    cout << "Ahoj!";
}
