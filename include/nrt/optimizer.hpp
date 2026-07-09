#pragma once

#include <vector>

#include "nrt/parameter.hpp"
#include "nrt/tensor.hpp"

namespace nrt {

// Stochastic Gradient Descent optimizer
class SGD {
public:
    // Constructor: takes parameters similar to PyTorch
    explicit SGD(const std::vector<Parameter>& params, double learning_rate);

    // Classic zero_grad to be called before each backward pass
    void zero_grad();

    // Updates all parameters in-place using their accumulated gradients
    void step();

    // Getters
    double learning_rate() const;
    size_t num_params() const;

private:
    std::vector<Parameter> params_;
    double learning_rate_;
};

}  // namespace nrt
