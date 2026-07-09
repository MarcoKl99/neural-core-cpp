#pragma once

#include "nrt/tensor.hpp"

namespace nrt {

// Wrapped operations that track computational nodes for autograd
Tensor matmul_autodiff(Tensor& a, Tensor& b);
Tensor add_autodiff(Tensor& a, Tensor& b);
Tensor subtract_autodiff(Tensor& a, Tensor& b);
Tensor scalar_mult_autodiff(Tensor& a, double scalar);

}  // namespace nrt
