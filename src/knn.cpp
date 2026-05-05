#include "knn.hpp"
#include <utility>
#include <stdexcept>
#include <algorithm>

double KNN::getSquaredEuclideanDist(const std::vector<double>& p1, const std::vector<double>& p2) {
    double res = 0;
    for (int i = 0; i < dim; ++i) {
        res += (p1[i] - p2[i]) * (p1[i] - p2[i]);
    }
    return res;
}

double KNN::getSquaredEuclideanDist(int starting_idx, const std::vector<double> &p2) {
   double res = 0;
    for (int i = 0; i < dim; ++i) {
        res += (vectors[starting_idx + i] - p2[i]) * (vectors[starting_idx + i] - p2[i]);
    }
    return res;
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

        std::vector<std::pair<int, double>> dist; //index, distance

        for(int i = 0; i < vectors.size(); i += dim) {
            dist.push_back({i / dim, getSquaredEuclideanDist(i, sample)});
        }

        std::sort(dist.begin(), dist.end(), [](const std::pair<int, double> p1, const std::pair<int, double> p2) {
            return p1.second < p2.second; 
        });

        return 0;
    }
