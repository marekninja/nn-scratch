//
// Created by marek on 10/27/2021.
//
using namespace std;
#include <vector>
#include <iostream>
#include <stdexcept>
#include <functional>


#ifndef PV021_PROJECT_OPERATIONS_H
#define PV021_PROJECT_OPERATIONS_H

template<typename T > class Matrix {
private:
    std::vector< std::vector<T> > matData;
    int numRows;
    int numCols;

public:
    Matrix(int num_rows, int num_cols){
        matData.resize(num_rows);

        for (int i = 0; i < matData.size(); ++i) {
            //TODO: treba skontrolovat, ze ci nema byt nula nastavena
            //matData[i].resize(num_cols,0.0);
            matData[i].resize(num_cols);
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

    Matrix(){
        numRows=0;
        numCols=0;
        matData={};
    }
    virtual ~Matrix(){};

    T& operator()(int row, int col) {
        return this->matData[row][col];
    }

    const T& operator()(int row, int col) const{
        return this->matData[row][col];
    }

//    void fillup(int num_rows, int num_cols, const vector<vector<T>> &mat_data);
    int getNumRows() const {
        return this->numRows;
    };
    int getNumCols() const{
        return this->numCols;
    };

    ///multiplication of two matrices
    ///Type T must have defined * operator
    Matrix<T> multiply(const Matrix &other){
//        if (numRows != other.getNumCols()){
//            throw runtime_error("Can not multiply matrices, mat1cols != mat2rows !");
//        }
        if (numCols != other.getNumRows()){
            throw runtime_error("Can not multiply matrices, mat1cols != mat2rows !");
        }

//        int cols = numCols;
        int cols = other.getNumCols();
        int rows = numRows;
//        int rows = other.getNumRows();

        Matrix result(rows, cols);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                for (int k = 0; k < numCols; ++k) {
//                result in one cell is sum of mymatrix[row,col] othermatrix[col]
//                result is multiplication of
//                    result.matData[i][j] += matData[i][k] * other.matData[k][j];
                    result(i,j) += matData[i][k] * other(k,j);
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
//                result.matData[i][j] = this->matData[i][j] + other.matData[i][j];
                result(i,j) = this->matData[i][j] + other(i,j);
            }
        }
        return result;
    };

    //Addition of 1 row matrix to every row
    Matrix<T> addToRow(const Matrix& row){
        int cols = numCols;
        int rows = numRows;
        Matrix result(rows, cols);

        for (int i = 0; i < numRows; ++i) {
            for (int j = 0; j < numCols; ++j) {
                result(i,j) = this->matData[i][j] + row(0,j);
            }
        }
        return result;
    }


    void apply(std::function<T(T&)> func){
        for (int i = 0; i < numRows; ++i) {
            for (int j = 0; j < numCols; ++j) {
                matData[i][j] = func(matData[i][j]);
            }
        }
    }

    vector<Matrix<T>> splitToBatches(int batch_size){
        if (numRows% batch_size != 0){
            throw runtime_error("Can not divide into batches!");
        }

        vector<Matrix<T>> results({});
        int iters = numRows/batch_size;

        for (int i = 0; i < iters; ++i) {
//            Matrix<double> batch({});
            vector<vector<T>> batch;
            for (int j = 0; j < batch_size; ++j) {
                batch.push_back(matData[i * batch_size+j]);
            }
            Matrix<T> batchMat(batch_size,numCols,batch);
            results.push_back(batchMat);
        }
        return results;
    }

    void printShape() const{
        cout <<"numCols: "<<numCols<< " numRows: "<<numRows<<endl;
    }


    void print() const{
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
