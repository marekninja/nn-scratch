//
// Created by marek on 10/27/2021.
//
#pragma GCC optimize("Ofast")
using namespace std;
#include <vector>
#include <iostream>
#include <stdexcept>
#include <functional>
#include <thread>
#define THREAD_ROWS 20


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
    Matrix(int num_rows, int num_cols, const std::vector<std::vector<T>>& mat_data){
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

    vector<vector<T>> getData() const{
        return this->matData;
    }

    static void multiplyRow(const Matrix<T>& m1, const Matrix<T>& m2, Matrix<T>& result,
                            const vector<int>& rowPartitions){
//        const size_t rowblock = 128;
//        const size_t cblock = 128;
//        const size_t iblock = 64;
        const size_t rowblock = 64;
        const size_t cblock = 128;
        const size_t iblock = 32;
// loop over tiles of the matrixes
        size_t start = rowPartitions[0];
        size_t end = rowPartitions[1] +1;
        for (size_t rr = start; rr < end; rr += rowblock) {
            size_t rlim = rr + rowblock < start ? rr+rowblock : end;

            for (size_t cc = 0; cc < m2.numCols; cc += cblock) {

                size_t clim = cc + cblock < m2.numCols ? cc+cblock : m2.numCols;

                for (size_t ii = 0; ii < m1.numCols; ii += iblock) {

//                    size_t ilim = std::min(ii + iblock, m1.cols);
                    size_t ilim = ii + iblock < m1.numCols ? ii + iblock : m1.numCols;
                    // multiply tile by tile
                    for (size_t row = rr; row < rlim; row++) {
                        for (size_t c = cc; c < clim; ++c) {
                            T t = result(row, c);
                            for (size_t i = ii; i < ilim; ++i)

                                t += m1(row, i) * m2(i, c);
                            result(row, c) = t;
                        }
                    }
                }
            }
        }

//        for (int row = rowPartitions[0]; row <= rowPartitions[1]; ++row) {
//            for (int i = 0; i < m2.numCols; ++i) {
//                for (int j = 0; j < m1.numCols; ++j) {
//                    result.matData[row][i] += m1(row,j) * m2(j,i);
//                }
//            }
//        }
    }


    static vector<vector<int>> partitionRows(Matrix<T>& m1, int thread_count){
        vector<vector<int>> rowsPartitions;
        rowsPartitions.resize(thread_count);

        int countOne = (int)m1.getNumRows() / thread_count;
        int remaining = (int)m1.getNumRows() % thread_count;

        int i = 0;
        while( i < m1.getNumRows()){
            vector<int> pair;
            if (i < (thread_count) * countOne){
                pair.push_back(i);

                if (remaining > 0){
                    pair.push_back(i + countOne);
                    remaining --;
                    i = i + countOne+1;
                } else {
                    pair.push_back(i + countOne - 1);
                    i = i + countOne;
                }
                rowsPartitions[(i / countOne)-1] = pair;
            } else {
                break;
            }

        }
        return rowsPartitions;
    }

    Matrix<T> multiply(const Matrix &other){
        if (numCols != other.getNumRows()){
            throw runtime_error("Can not multiply matrices, mat1cols != mat2rows !");
        }

        int cols = other.getNumCols();
        int rows = numRows;

        Matrix result(rows, cols);
        vector<thread> threads;
        int threadCount = thread::hardware_concurrency();

        int neededThreads = numRows % THREAD_ROWS == 0 ? numRows / THREAD_ROWS  : numRows / THREAD_ROWS + 1;

        if (neededThreads > threadCount){
            neededThreads = threadCount;
        }

        vector<vector<int>> rowPartitions = partitionRows(*this, neededThreads);

        for (int i = 0; i < neededThreads; ++i) {
            threads.push_back(std::thread(multiplyRow, *this, other, std::ref(result), rowPartitions[i]));
        }

        for (int i = 0; i < neededThreads; ++i) {
            threads[i].join();
        }


        return result;
    }

    ///multiplication of two matrices
    ///Type T must have defined * operator
    Matrix<T> multiplyNaive(const Matrix &other){
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

    Matrix<T> transpose(){
        Matrix result(numCols, numRows);

        for (int i = 0; i < numRows; ++i) {
            for (int j = 0; j < numCols; ++j) {
                result(j,i) = matData[i][j];
            }
        }
        return result;
    }

    Matrix<T> flatMeanCols(){
        Matrix result(1, numCols);

        for (int i = 0; i < numCols; ++i) {
            for (int j = 0; j < numRows; ++j) {
                result(0,i) += matData[j][i];
            }
            result(0,i) = result(0,i) / numRows;
        }
        return result;
    }

    Matrix<T> flatMeanRowsAlloc(){
        Matrix result(numRows, 1);

        for (int i = 0; i < numRows; ++i) {
            for (int j = 0; j < numCols; ++j) {
                result(i,0) += matData[i][j];
            }
            result(i,0) = result(i,0) / numCols;
        }
        return result;
    }

    void flatMeanRows(){
        for (int i = 0; i < numRows; ++i) {
            for (int j = 1; j < numCols; ++j) {
//                result(i,0) += matData[i][j];
                matData[i][0] += matData[i][j];
            }
            matData[i][0] /= numCols;
        }
        numCols = 1;
    }

    Matrix<T> multiplyCellsAlloc(const Matrix<T>& other){
        if (this->getNumRows() != other.getNumRows() || this->getNumCols() != other.getNumCols()){
            throw runtime_error("Can not multiplyCells() matrices of different shape!");
        }
        Matrix result(numRows, numCols);
        for (int i = 0; i < numRows; ++i) {
            for (int j = 0; j < numCols; ++j) {
//                result.matData[i][j] = this->matData[i][j] + other.matData[i][j];
                result(i,j) = matData[i][j] * other(i,j);
            }
        }
        return result;
    }

    void multiplyCells(const Matrix<T>& other){
        if (this->getNumRows() != other.getNumRows() || this->getNumCols() != other.getNumCols()){
            throw runtime_error("Can not multiplyCells() matrices of different shape!");
        }
        for (int i = 0; i < numRows; ++i) {
            for (int j = 0; j < numCols; ++j) {
//                result.matData[i][j] = this->matData[i][j] + other.matData[i][j];
                matData[i][j] *= other(i,j);
            }
        }
    }

    Matrix<T> multiplyNumAlloc(const double& num){
        Matrix result(numRows, numCols);
        for (int i = 0; i < numRows; ++i) {
            for (int j = 0; j < numCols; ++j) {
//                result.matData[i][j] = this->matData[i][j] + other.matData[i][j];
                result(i,j) = matData[i][j] * num;
            }
        }
        return result;
    }

    void multiplyNum(const double& num){
        for (int i = 0; i < numRows; ++i) {
            for (int j = 0; j < numCols; ++j) {
//                result.matData[i][j] = this->matData[i][j] + other.matData[i][j];
                matData[i][j] *= num;
            }
        }
    }


    Matrix<T> add(const Matrix &other){
        if (this->getNumRows() != other.getNumRows() || this->getNumCols() != other.getNumCols()){
            throw runtime_error("Can not add() matrices of different shape!");
        }
        int cols = numCols;
        int rows = numRows;
        Matrix result(rows, cols);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
//                result.matData[i][j] = this->matData[i][j] + other.matData[i][j];
                result(i,j) = this->matData[i][j] + other(i,j);
            }
        }
        return result;
    };

    static void minusRow(const Matrix<T>& m1, const Matrix<T>& m2, Matrix<T>& result,const vector<int>& rowPartitions){
        for (int row = rowPartitions[0]; row <= rowPartitions[1]; ++row) {
            for (int i = 0; i < m1.numCols; ++i) {
                result(row,i) = m1(row,i) - m2(row,i);
            }
        }
    }

//    Matrix<T> minus(const Matrix &other){
    Matrix<T> minusAlloc(const Matrix &other){
        if (this->getNumRows() != other.getNumRows() || this->getNumCols() != other.getNumCols()){
            throw runtime_error("Can not add() matrices of different shape!");
        }
        int cols = numCols;
        int rows = numRows;

        Matrix result(rows, cols);
        vector<thread> threads;
        int threadCount = thread::hardware_concurrency();

        int neededThreads = numRows % THREAD_ROWS == 0 ? numRows / THREAD_ROWS  : numRows / THREAD_ROWS + 1;

        if (neededThreads > threadCount){
            neededThreads = threadCount;
        }

        vector<vector<int>> rowPartitions = partitionRows(*this, neededThreads);

        for (int i = 0; i < neededThreads; ++i) {
            threads.push_back(std::thread(minusRow, *this, other, std::ref(result), rowPartitions[i]));
        }

        for (int i = 0; i < neededThreads; ++i) {
            threads[i].join();
        }
        return result;
    };

    void minus(const Matrix &other){
        if (this->getNumRows() != other.getNumRows() || this->getNumCols() != other.getNumCols()){
            throw runtime_error("Can not add() matrices of different shape!");
        }
        int cols = numCols;
        int rows = numRows;

        vector<thread> threads;
        int threadCount = thread::hardware_concurrency();

        int neededThreads = numRows % THREAD_ROWS == 0 ? numRows / THREAD_ROWS  : numRows / THREAD_ROWS + 1;

        if (neededThreads > threadCount){
            neededThreads = threadCount;
        }

        vector<vector<int>> rowPartitions = partitionRows(*this, neededThreads);

        for (int i = 0; i < neededThreads; ++i) {
            threads.push_back(std::thread(minusRow, *this, other, std::ref(*this), rowPartitions[i]));
        }

        for (int i = 0; i < neededThreads; ++i) {
            threads[i].join();
        }
    };

    //Addition of 1 row matrix to every row
    void addToCol(Matrix& col){
        int cols = numCols;
        int rows = numRows;
//        Matrix result(rows, cols);

        for (int i = 0; i < numRows; ++i) {
            for (int j = 0; j < numCols; ++j) {
//                (*this)(i,j) = (*this)(i,j) + col(i,0);
                matData[i][j] +=  col(i,0);
            }
        }
//        return result;
    }

    Matrix<T> addToColAlloc(const Matrix& col){
        int cols = numCols;
        int rows = numRows;
        Matrix result(rows, cols);

        for (int i = 0; i < numRows; ++i) {
            for (int j = 0; j < numCols; ++j) {
//                result(i,j) = result(i,j) + col(i,0);
                result(i,j) = matData[i][j] + col(i,0);
            }
        }
        return result;
    }


    void apply(std::function<T(const T&)> func){
//        cout << "appy relu" << endl;

        for (int i = 0; i < numRows; ++i) {
            for (int j = 0; j < numCols; ++j) {
                matData[i][j] = func(matData[i][j]);
            }
        }
    }

    void apply(std::function<T(const int& seed, const int& incoming, const int& cols)> func, const int& seed, const int& incoming, const int& cols){
//        cout << "apply softmax" << endl;
        for (int i = 0; i < numRows; ++i) {
            for (int j = 0; j < numCols; ++j) {
                matData[i][j] = func(seed, incoming, cols);
            }
        }
    }

    void applySoftmax(std::function<void (vector<T>&)> func){
//        cout << "apply softmax" << endl;

        Matrix<T> copy = (*this).transpose();

        for (int i = 0; i < numCols; ++i) {
            func(copy.matData[i]);
        }
        copy = copy.transpose();
        matData = copy.matData;
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

    static bool compare(const Matrix<T>& m1, const Matrix<T>& m2, const string& operation) {
        cout << operation << ": ";
        if (m1.getNumRows() != m2.getNumRows()){
            cout << "not equal! m1 rows: "<< m1.getNumRows() <<" m2 rows: "<<m2.getNumRows() << "\n";
            return false;
        }
        if (m1.getNumCols() != m2.getNumCols()){
            cout << "not equal! m1 rows: "<< m1.getNumCols() <<" m2 rows: "<<m2.getNumCols() << "\n";
            return false;
        }


        for (int i = 0; i < m1.numRows; ++i) {
            for (int j = 0; j < m1.numCols; ++j) {
                if (m1(i,j) != m2(i,j)){
                    cout << "not equal! (i= "<< i <<", j= "<<j<<  " ) m1: "<< m1(i,j);
                    cout << " m2: " << m2(i,j) << "\n";
                    return false;
                }
            }
        }
        cout<<"matrices same!\n";
        return true;
    }
};

#endif //PV021_PROJECT_OPERATIONS_H
