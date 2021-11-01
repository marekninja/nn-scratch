//
// Created by marek on 10/27/2021.
//
using namespace std;
#include <vector>
#include <iostream>
#include <stdexcept>



#ifndef PV021_PROJECT_OPERATIONS_H
#define PV021_PROJECT_OPERATIONS_H

template<typename T > class Matrix {
private:
    std::vector< std::vector<T> > matData;
    int numRows;
    int numCols;

//public:
public:
    Matrix(int num_rows, int num_cols){
        matData.resize(num_rows);

        for (int i = 0; i < matData.size(); ++i) {
            matData[i].resize(num_cols,0.0);
        }

        numRows = num_rows;
        numCols = num_cols;
    };

    //    Matrix(int num_rows, int num_cols, vector<double> mat_data);
    Matrix(int num_rows, int num_cols, const std::vector<std::vector<double>>& mat_data){
        matData.resize(num_rows);

        for (int i = 0; i < matData.size(); ++i) {
            matData[i].resize(num_cols);
            matData[i] = mat_data[i];
        }

        numRows = num_rows;
        numCols = num_cols;
    };

    Matrix(const Matrix<T>& copy){
        matData = copy.matData;
        numRows = copy.getNumRows();
        numCols = copy.getNumCols();
    };

//    Matrix();
    virtual ~Matrix(){};

//    void fillup(int num_rows, int num_cols, const vector<vector<T>> &mat_data);
    int getNumRows() const {
        return this->numRows;
    };
    int getNumCols() const{
        return this->numCols;
    };


    Matrix<T> multiply(const Matrix &other){
        if (numRows != other.getNumCols()){
            throw runtime_error("Can not multiply matrices, mat1cols != mat2rows !");
        }

        int cols = other.getNumCols();
        int rows = numRows;

        Matrix result(rows, cols);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                for (int k = 0; k < numCols; ++k) {
//                result in one cell is sum of mymatrix[row,col] othetmatrix[col]
//                result is multiplication of
                    result.matData[i][j] += matData[i][k] * other.matData[k][j];
                }
            }
        }

        return result;
    };

    Matrix<T> add(const Matrix &other){
        if (this->getNumRows() != other.getNumRows() or this->getNumCols() != other.getNumCols()){
            throw runtime_error("Can not add matrices of different shape!");
        }
        int cols = numCols;
        int rows = numRows;
        Matrix result(rows, cols);
        for (int i = 0; i < cols; ++i) {
            for (int j = 0; j < rows; ++j) {
                result.matData[i][j] += this->matData[i][j] * other.matData[i][j];
            }
        }
        return result;
    };

    void print(){
        for (int i = 0; i < this->getNumRows(); ++i) {
            cout << "( ";
            for (int j = 0; j < this->getNumCols(); ++j) {
                cout << this->matData[i][j] << " ";
            }
            cout << ")" << endl;
        }
    };
};

#endif //PV021_PROJECT_OPERATIONS_H
