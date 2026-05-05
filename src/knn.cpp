#include "knn.hpp"
#include <utility>
#include <stdexcept>

KNN::KNN(int dim, std::vector<double> vectors, std::vector<int> labels, DistanceMetric distanceMetric) : 
    dim(dim), 
    vectors(std::move(vectors)), 
    labels(std::move(labels)), 
    distanceMetric(distanceMetric) 
    { 
        if (dim <= 0) {
            throw std::invalid_argument("Dimension must be greater than 0.");
        }
        if (this->vectors.size() % dim != 0) {
            throw std::invalid_argument("The total number of elements in 'vectors' must be divisible by 'dim'.");
        }
        if (this->vectors.size() / dim != this->labels.size()) {
            throw std::invalid_argument("The number of vectors does not match the number of labels.");
        }
    }
