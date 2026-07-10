#include "nrt/linear.hpp"

#include <random>
#include <stdexcept>

#include "nrt/operations.hpp"

namespace nrt {

Linear::Linear(size_t in_features, size_t out_features)
    : in_features_(in_features),
      out_features_(out_features),
      weights_(std::make_shared<Tensor>(std::vector<size_t>{out_features, in_features})),
      bias_(std::make_shared<Tensor>(std::vector<size_t>{out_features, 1})),
      grad_weights_({out_features, in_features}),
      grad_bias_({out_features, 1}) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 0.1);

    for (size_t i = 0; i < out_features_; ++i) {
        for (size_t j = 0; j < in_features_; ++j) {
            (*weights_)(i, j) = dist(gen);  // dereference to reach operator()
        }
    }
    // bias_ stays zero
}

void Linear::set_weights(const Tensor& w, const Tensor& b) {
    // Check the shapes
    if (w.shape() != std::vector<size_t>{out_features_, in_features_}) {
        throw std::invalid_argument("Linear::set_weights: weight shape mismatch");
    }
    if (b.shape() != std::vector<size_t>{out_features_, 1}) {
        throw std::invalid_argument("Linear::set_weights: bias shape mismatch");
    }

    // Set the values
    *weights_ = w;
    *bias_ = b;
}

std::shared_ptr<Tensor> Linear::forward(std::shared_ptr<Tensor> x) {
    if (x->shape() != std::vector<size_t>{in_features_, 1}) {
        throw std::invalid_argument("Linear::forward: input shape mismatch");
    }
    auto z = matmul_autodiff(weights_, x);  // passes the shared param straight in
    auto y = add_autodiff(z, bias_);
    return y;
}

Tensor Linear::backward(const Tensor& grad_output) {
    // Check that forward was called before
    if (!last_input_.has_value()) {
        throw std::logic_error("Linear::backward: forward() must be called before backward()");
    }

    // Check that the incoming gradient has the correct shape
    if (grad_output.shape() != std::vector<size_t>{out_features_, 1}) {
        throw std::invalid_argument("Linear::backward: grad_output shape mismatch");
    }

    const Tensor& x = last_input_.value();

    // dL/dW = grad_output * x^T
    grad_weights_ += grad_output.matmul(x.transpose());

    // dL/db = grad_output
    grad_bias_ += grad_output;

    accumulation_count_ += 1;

    // dL/dx = W^T * grad_output
    return weights_->transpose().matmul(grad_output);
}

void Linear::zero_grad() {
    // Reset the gradients
    grad_weights_ = Tensor({out_features_, in_features_});
    grad_bias_ = Tensor({out_features_, 1});
    accumulation_count_ = 0;
}

Tensor Linear::average_grad_weights() const {
    if (accumulation_count_ == 0) {
        throw std::logic_error("Linear::average_grad_weights: no gradients accumulated");
    }
    return grad_weights_ * (1.0 / static_cast<double>(accumulation_count_));
}

Tensor Linear::average_grad_bias() const {
    if (accumulation_count_ == 0) {
        throw std::logic_error("Linear::average_grad_bias: no gradients accumulated");
    }
    return grad_bias_ * (1.0 / static_cast<double>(accumulation_count_));
}

std::vector<nrt::Parameter> Linear::parameters() {
    return {
        {weights_.get()},
        {bias_.get()},
    };
}

// Getter
size_t Linear::in_features() const {
    return in_features_;
}

size_t Linear::out_features() const {
    return out_features_;
}

Tensor& Linear::weights() {
    return *weights_;
}

Tensor& Linear::bias() {
    return *bias_;
}

}  // namespace nrt
