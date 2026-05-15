#pragma once
#include<vector>

template <typename T>
class Matrix {
    int rows, cols;
    std::vector<T> elements;
public:

    Matrix(int rows, int cols) : rows(rows), cols(cols) {
        this->elements.resize(rows * cols);
    }
    Matrix(int rows, int cols, const std::vector<T>& elements) : rows(rows), cols(cols) {
        //tutaj bedzie logika sprawdzania, czy dane są w porządku
        this->elements = elements;
    }
};