#pragma once

#include <functional>

namespace nrt {

// Forward declaration, is defined in nrt/tensor.hpp
// but cannot be included due to circular dependencies
class Tensor;

// Represents a node in the computation graph
struct ComputationNode {
    // Store pointers to input tensors (so they don't dangle when going out of scope)
    std::vector<Tensor*> inputs;

    // Backward function now receives inputs as parameters instead of capturing by reference
    // Signature: void(output_tensor, gradient_wrt_output, input_pointers)
    std::function<void(Tensor&, const Tensor&, const std::vector<Tensor*>&)> backward_fn;
};

}  // namespace nrt
