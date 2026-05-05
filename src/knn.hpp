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
public:
    KNN(int dim, std::vector<double> vectors, std::vector<int> labels, DistanceMetric distanceMetric = DistanceMetric::EUCLIDEAN);
};