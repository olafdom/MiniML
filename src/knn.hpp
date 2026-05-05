#include<vector>

enum class DistanceMetric {
    EUCLIDEAN,
    MANHATTAN
};

class KNN {
    int dim; //dimension of vectors
    std::vector<double> vectors; //vectors stored in flattened vector (next to each other)
    std::vector<int> labels; //i-th vector label is labels[i]
    DistanceMetric distanceMetric = DistanceMetric::EUCLIDEAN; //distance metrics
    int vectors_count() { return vectors.size() / dim; } 
    double getSquaredEuclideanDist(const std::vector<double>& p1, const std::vector<double>& p2);
public:
    KNN(int dim, std::vector<double> vectors, std::vector<int> labels, DistanceMetric distanceMetric = DistanceMetric::EUCLIDEAN);
    int predict(const std::vector<double>& sample, int k);
};