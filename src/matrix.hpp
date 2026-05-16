#pragma once
#include <vector>
#include <stdexcept>
#include <cmath>

template <typename T>
class Matrix {
private:
    int rows_count;
    int cols_count;
    std::vector<T> elements;

public:
    Matrix(int r, int c) : rows_count(r), cols_count(c) {
        if (rows_count <= 0 || cols_count <= 0) {
            throw std::invalid_argument("Matrix dimensions must be positive!");
        }
        elements.resize(r * c, T{});
    }

    Matrix(int r, int c, const std::vector<T>& elems) : rows_count(r), cols_count(c), elements(elems) {
        if (elements.size() != static_cast<size_t>(r * c)) {
            throw std::invalid_argument("Number of elements does not match!");
        }
    }

    int rows() const { return rows_count; }
    int cols() const { return cols_count; }

    // Operatory dostępu do elementów
    T& operator()(int r, int c) {
        return elements[r * cols_count + c];
    }

    const T& operator()(int r, int c) const {
        return elements[r * cols_count + c];
    }

    const std::vector<T>& get_elements() const {
        return elements;
    }

    // Dodawanie macierzy
    Matrix<T> operator+(const Matrix<T>& other) const {
        if (rows_count != other.rows_count || cols_count != other.cols_count) {
            throw std::invalid_argument("Matrix dimensions must be matching!");
        }

        Matrix<T> result(rows_count, cols_count);
        for (size_t i = 0; i < elements.size(); ++i) {
            result.elements[i] = this->elements[i] + other.elements[i];
        }
        return result;
    }

    // Transpozycja
    Matrix<T> transpose() const {
        Matrix<T> result(cols_count, rows_count);
        for (int r = 0; r < rows_count; ++r) {
            for (int c = 0; c < cols_count; ++c) {
                result(c, r) = (*this)(r, c);
            }
        }
        return result;
    }

    // Mnożenie macierzy (Dot product)
    Matrix<T> dot(const Matrix<T>& other) const {
        if (this->cols_count != other.rows_count) {
            throw std::invalid_argument("Matrix dimension must be matching");
        }

        Matrix<T> result(this->rows_count, other.cols_count);

        for (int i = 0; i < this->rows_count; ++i) {
            for (int j = 0; j < other.cols_count; ++j) {
                T sum = T{};
                for (int k = 0; k < this->cols_count; ++k) {
                    sum += (*this)(i, k) * other(k, j);
                }
                result(i, j) = sum;
            }
        }
        return result;
    }
};