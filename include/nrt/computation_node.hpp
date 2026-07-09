#pragma once

#include <functional>

namespace nrt {

// Forward declaration, is defined in nrt/tensor.hpp
// but cannot be included due to circular dependencies
class Tensor;

// Represents a node in the computation graph
struct ComputationNode {
    // Backward function: computes gradients for inputs
    // Signature: void(Tensor& output, const Tensor& grad_output)
    std::function<void(Tensor&, const Tensor&)> backward_fn;
};

}  // namespace nrt
