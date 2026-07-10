#pragma once

#include <memory>
#include <vector>

#include "nrt/parameter.hpp"
#include "nrt/tensor.hpp"

namespace nrt {

// Base class for all neural network modules
// Inspired by torch.nn.Module
class Module {
public:
    // Explicitly state the exception specification to avoid mismatches with child classes
    // -> Child classes also mark the destructor as noexcept
    virtual ~Module() noexcept = default;

    // Forward pass: applies the module to input tensor
    virtual std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x) = 0;

    // Returns all learnable parameters (weights, biases, potentially others like scale and shift
    // for batchnorm later) - Empty vector if the module has no parameters
    virtual std::vector<Parameter> parameters() = 0;
};

}  // namespace nrt
