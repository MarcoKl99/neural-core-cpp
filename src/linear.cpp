#include "nrt/linear.hpp"

#include <optional>
#include <random>

#include "nrt/operations.hpp"

namespace nrt {

Linear::Linear(size_t in_features, size_t out_features, WeightInit init,
               std::optional<unsigned int> seed)
    : in_features_(in_features),
      out_features_(out_features),
      weights_(std::make_shared<Tensor>(std::vector<size_t>{out_features, in_features})),
      bias_(std::make_shared<Tensor>(std::vector<size_t>{out_features, 1})) {
    std::mt19937 gen(seed.has_value() ? seed.value() : std::random_device{}());

    // He (Kaiming): for layers followed by ReLU. Xavier (Glorot): for Sigmoid/Tanh.
    double std_dev = (init == WeightInit::He)
                         ? std::sqrt(2.0 / static_cast<double>(in_features_))
                         : std::sqrt(2.0 / static_cast<double>(in_features_ + out_features_));
    std::normal_distribution<double> dist(0.0, std_dev);

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
    if (x->rank() != 2 || x->shape()[1] != in_features_) {
        throw std::invalid_argument("Linear::forward: input shape must be {batch, in_features}");
    }

    // Forward: y = x @ W.T + b
    auto w_t = transpose_autodiff(weights_);  // Use autodiff transpose!
    auto z = matmul_autodiff(x, w_t);         // {batch, out_features}

    // Reshape bias from {out_features, 1} to {1, out_features}
    auto bias_reshaped = reshape_autodiff(bias_, {1, out_features_});
    auto y = add_autodiff(z, bias_reshaped);

    return y;
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
