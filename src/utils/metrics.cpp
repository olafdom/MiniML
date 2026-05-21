#include "metrics.hpp"


namespace miniml::metrics {
    //trzeba przerobic, zeby dzialalo
    
    double getSquaredEuclideanDist(const std::vector<double>& p1, const std::vector<double>& p2) {

        double res = 0;
        for (int i = 0; i < p1.size(); ++i) {
            res += (p1[i] - p2[i]) * (p1[i] - p2[i]);
        }
        return res;
    }
    
    double getSquaredEuclideanDist(const std::vector<double>& vectors, int starting_idx, const std::vector<double>& p, int dim) {

        double res = 0;
        for (int i = 0; i < dim; ++i) {
            res += (vectors[starting_idx + i] - p[i]) * (vectors[starting_idx + i] - p[i]);
        }
        return res;
    }
    
    double getManhattanDist(const std::vector<double>& vectors, int starting_idx, const std::vector<double> &p, int dim) {
        double res = 0;
        for(int i = 0; i < dim; ++i) {
            res += std::abs(vectors[starting_idx + i] - p[i]);
        }
        return res;
    }
}