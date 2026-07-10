#pragma once

#include <memory>

#include "nrt/tensor.hpp"

namespace nrt {

// Free-function forms of the loss (no autograd graph).
double mse(const Tensor& y_hat, const Tensor& y);
Tensor mse_derivative(const Tensor& y_hat, const Tensor& y);

class MSELoss {
public:
    // Returned as a {1,1} tensor carrying a computation node.
    // The target y is treated as a constant (no gradient flows into it), matching PyTorch
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> y_hat, std::shared_ptr<Tensor> y);
};

}  // namespace nrt
