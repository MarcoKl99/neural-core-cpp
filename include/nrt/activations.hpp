#pragma once

#include "nrt/module.hpp"
#include "nrt/parameter.hpp"
#include "nrt/tensor.hpp"

namespace nrt {

// Standard function definitions
Tensor relu(const Tensor& x);
Tensor sigmoid(const Tensor& x);

// Local derivatives of the functions
Tensor relu_derivative(const Tensor& x);
Tensor sigmoid_derivative(const Tensor& x);

// Chained derivatives of the functions, taking
Tensor relu_backward(const Tensor& grad_output, const Tensor& x);
Tensor sigmoid_backward(const Tensor& grad_output, const Tensor& x);

class ReLU : public Module {
public:
    Tensor forward(const Tensor& x) override;
    std::vector<Parameter> parameters() override;
};

class Sigmoid : public Module {
public:
    Tensor forward(const Tensor& x) override;
    std::vector<Parameter> parameters() override;
};

}  // namespace nrt
