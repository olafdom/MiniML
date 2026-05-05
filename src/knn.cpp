#include "knn.hpp"
#include <utility>
#include <stdexcept>

double KNN::getSquaredEuclideanDist(const std::vector<double>& p1, const std::vector<double>& p2) {
    double res = 0;
    for (int i = 0; i < dim; ++i) {
        res += (p1[i] - p2[i]) * (p1[i] - p2[i]);
    }
    return sqrt(res);
}

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

    int KNN::predict(const std::vector<double> &sample, int k) {
        if (sample.size() != dim) {
            throw std::invalid_argument("Given vector must have the correct dimension");
        }
        if (k >= vectors_count()) {
            throw std::invalid_argument("k must be less than the number of vectors in dataset");
        }

        return 0;
    }
