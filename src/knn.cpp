#include "knn.hpp"

double KNN::getSquaredEuclideanDist(const std::vector<double>& p1, const std::vector<double>& p2) {
   return miniml::metrics::getSquaredEuclideanDist(p1, p2);
}

double KNN::getSquaredEuclideanDist(int starting_idx, const std::vector<double> &p) {
   return miniml::metrics::getSquaredEuclideanDist(vectors, starting_idx, p, dim);
}

double KNN::getManhattanDist(int starting_idx, const std::vector<double> &p) {
    return miniml::metrics::getManhattanDist(vectors, starting_idx, p, dim);
}

KNN::KNN(int dim, std::vector<double> vectors, std::vector<int> labels, DistanceMetric distanceMetric) : 
    dim(dim), 
    vectors(std::move(vectors)), 
    labels(std::move(labels)), 
    distanceMetric(distanceMetric) { 

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
        dist.reserve(vectors.size() / dim);

        if(distanceMetric == DistanceMetric::EUCLIDEAN) {
            for(int i = 0; i < vectors.size(); i += dim) {
                dist.push_back({i / dim, getSquaredEuclideanDist(i, sample)});
            }
        }
        else if(distanceMetric == DistanceMetric::MANHATTAN) {
            for(int i = 0; i < vectors.size(); i += dim) {
                dist.push_back({i / dim, getManhattanDist(i, sample)});
            }
        }
        
        std::sort(dist.begin(), dist.end(), [](const std::pair<int, double> p1, const std::pair<int, double> p2) {
            return p1.second < p2.second; 
        });

        std::unordered_map<int, int> occ;

        for(int i = 0; i < k; ++i) {
            occ[labels[dist[i].first]]++;
        }

        auto it = std::max_element(occ.begin(), occ.end(), 
        [](const auto& a, const auto& b) {
            return a.second < b.second;
        });

        int nearest_neighbour = -1;
        if (it != occ.end()) nearest_neighbour = it->first;

        return nearest_neighbour;
    }
