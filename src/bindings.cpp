#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 
#include <pybind11/numpy.h>
#include "knn.hpp"
#include "matrix.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "MiniML: A high-performance KNN library written in C++ with Python bindings";

    py::class_<Matrix<double>>(m, "_Matrix")
        .def(py::init([](py::array_t<double> input_array) {
            py::buffer_info buf = input_array.request();
            
            int rows = buf.shape[0];
            int cols = buf.shape[1];
            
            double* ptr = static_cast<double*>(buf.ptr);

            std::vector<double> flat_data(ptr, ptr + (rows * cols));
            
            return new Matrix<double>(rows, cols, flat_data);
        }))
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

    py::class_<KNN>(m, "_KNN")
        .def(py::init([](py::array_t<double> input_list, 
                         py::array_t<int> labels, 
                         DistanceMetric metric) {
            
            py::buffer_info data_buf = input_list.request();
            py::buffer_info labels_buf = labels.request();

            int rows = data_buf.shape[0];
            int cols = data_buf.shape[1];

            double* data_ptr = static_cast<double*>(data_buf.ptr);
            int* labels_ptr = static_cast<int*>(labels_buf.ptr);

            std::vector<double> flat_data(data_ptr, data_ptr + (rows * cols));
            std::vector<int> labels_vec(labels_ptr, labels_ptr + rows);

            return new KNN(cols, std::move(flat_data), std::move(labels_vec), metric);
        }))
        .def("predict", &KNN::predict, 
             py::arg("sample"), 
             py::arg("k"),
             "Predicts the label for a given sample point using K-Nearest Neighbors");
}