#pragma once

#include <memory>

#include "nrt/module.hpp"
#include "nrt/parameter.hpp"
#include "nrt/tensor.hpp"

namespace nrt {

// Standard function definitions
Tensor relu(const Tensor& x);
Tensor sigmoid(const Tensor& x);
Tensor softmax(const Tensor& x);

// Local derivatives of the functions
Tensor relu_derivative(const Tensor& x);
Tensor sigmoid_derivative(const Tensor& x);

class ReLU : public Module {
public:
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x) override;
    std::vector<Parameter> parameters() override;
};

class Sigmoid : public Module {
public:
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x) override;
    std::vector<Parameter> parameters() override;
};

class Softmax : public Module {
public:
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x) override;
    std::vector<Parameter> parameters() override;
};

}  // namespace nrt
