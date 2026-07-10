#pragma once

#include <functional>
#include <memory>

namespace nrt {

// Forward declaration, is defined in nrt/tensor.hpp
// but cannot be included due to circular dependencies
class Tensor;

// Represents a node in the computation graph
struct ComputationNode {
    std::vector<std::shared_ptr<Tensor>> inputs;
    std::function<void(Tensor&, const Tensor&, const std::vector<std::shared_ptr<Tensor>>&)>
        backward_fn;
};

}  // namespace nrt
