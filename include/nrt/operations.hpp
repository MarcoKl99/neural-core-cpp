#pragma once

#include <memory>

#include "nrt/tensor.hpp"

namespace nrt {

// Wrapped operations that track computational nodes for autograd
std::shared_ptr<Tensor> matmul_autodiff(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> add_autodiff(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> subtract_autodiff(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
std::shared_ptr<Tensor> scalar_mult_autodiff(std::shared_ptr<Tensor> a, double scalar);
std::shared_ptr<Tensor> reshape_autodiff(std::shared_ptr<Tensor> a,
                                         const std::vector<size_t>& new_shape);
}  // namespace nrt
