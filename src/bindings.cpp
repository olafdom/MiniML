#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 
#include "knn.hpp"
#include "matrix.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "MiniML: A high-performance KNN library written in C++ with Python bindings";

    py::class_<Matrix<double>>(m, "Matrix")
        .def(py::init<int, int>()) // Pierwszy konstruktor
        .def(py::init<int, int, const std::vector<double>&>()) // Drugi konstruktor z danymi
        .def("rows", &Matrix<double>::rows)
        .def("cols", &Matrix<double>::cols)
        .def("get_elements", &Matrix<double>::get_elements)
        .def("transpose", &Matrix<double>::transpose)
        .def("dot", &Matrix<double>::dot)
        .def("__add__", &Matrix<double>::operator+);

    py::enum_<DistanceMetric>(m, "DistanceMetric")
        .value("EUCLIDEAN", DistanceMetric::EUCLIDEAN)
        .value("MANHATTAN", DistanceMetric::MANHATTAN)
        .export_values();

    py::class_<KNN>(m, "KNN")
        .def(py::init([](std::vector<std::vector<double>> input_list, 
                         std::vector<int> labels, 
                         DistanceMetric metric) {
            
            if (input_list.empty()) {
                throw std::invalid_argument("Input data cannot be empty.");
            }

            size_t rows = input_list.size();
            size_t cols = input_list[0].size();
            
            if (labels.size() != rows) {
                throw std::invalid_argument("Number of labels must match the number of input vectors.");
            }

            std::vector<double> flat_data;
            flat_data.reserve(rows * cols);

            for (const auto& row : input_list) {
                if (row.size() != cols) {
                    throw std::invalid_argument("All input vectors must have the same dimension.");
                }
                flat_data.insert(flat_data.end(), row.begin(), row.end());
            }

            return new KNN(static_cast<int>(cols), std::move(flat_data), std::move(labels), metric);
        }))
        
        .def("predict", &KNN::predict, 
             py::arg("sample"), 
             py::arg("k"),
             "Predicts the label for a given sample point using K-Nearest Neighbors");
}