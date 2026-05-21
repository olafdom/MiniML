#pragma once
#include<vector>
#include<cmath>

namespace miniml::metrics {
    double getSquaredEuclideanDist(const std::vector<double>& p1, const std::vector<double>& p2);
    double getSquaredEuclideanDist(const std::vector<double>& vectors, int starting_idx, const std::vector<double>& p, int dim);
    double getManhattanDist(const std::vector<double>& vectors, int starting_idx, const std::vector<double> &p, int dim);
}