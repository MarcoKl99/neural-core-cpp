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

// Cross-entropy over RAW LOGITS (softmax applied internally - do not pre-apply Softmax
// yourself). target is a class index in [0, num_classes), not one-hot - mirrors
// PyTorch's nn.CrossEntropyLoss.
double cross_entropy(const Tensor& logits, size_t target);
Tensor cross_entropy_grad(const Tensor& logits,
                          size_t target);  // = softmax(logits) - one_hot(target)

class CrossEntropyLoss {
public:
    // Batch version: logits {batch, num_classes}, targets {batch, 1} with class indices
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> logits,
                                    std::shared_ptr<Tensor> targets);
};

}  // namespace nrt
