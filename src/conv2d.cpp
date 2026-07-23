#include "nrt/conv2d.hpp"

#include "nrt/operations.hpp"

namespace nrt {

Conv2D::Conv2D(size_t in_channels, size_t out_channels, size_t kernel_size, WeightInit init,
               std::optional<unsigned int> seed)
    : in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size) {
    // Create weight tensor {out_channels, in_channels, kernel_size, kernel_size}
    weights_ = std::make_shared<Tensor>(
        std::vector<size_t>{out_channels, in_channels, kernel_size, kernel_size});

    // Create bias tensor {out_channels, 1}
    bias_ = std::make_shared<Tensor>(std::vector<size_t>{out_channels, 1});

    // Conv2D fan-in = in_channels * kernel_height * kernel_width
    double fan_in = static_cast<double>(in_channels * kernel_size * kernel_size);
    double scale = 0.0;

    // Determine initialization scale
    if (init == WeightInit::He) {
        scale = std::sqrt(2.0 / fan_in);
    } else if (init == WeightInit::Xavier) {
        scale = std::sqrt(1.0 / fan_in);
    }

    // Initialize weights with random values scaled appropriately
    std::mt19937 rng(seed.has_value() ? seed.value() : std::random_device{}());
    std::normal_distribution<double> dist(0.0, 1.0);

    for (size_t oc = 0; oc < out_channels; ++oc) {
        for (size_t ic = 0; ic < in_channels; ++ic) {
            for (size_t kh = 0; kh < kernel_size; ++kh) {
                for (size_t kw = 0; kw < kernel_size; ++kw) {
                    (*weights_)(oc, ic, kh, kw) = dist(rng) * scale;
                }
            }
        }
    }

    // Initialize bias to zero (already done by Tensor constructor)
}

std::shared_ptr<Tensor> Conv2D::forward(std::shared_ptr<Tensor> x) {
    // Validate input shape: {batch, in_channels, height, width}
    if (x->rank() != 4 || x->shape()[1] != in_channels_) {
        throw std::invalid_argument("Conv2D::forward: input shape must be {batch, " +
                                    std::to_string(in_channels_) + ", height, width}");
    }

    // Use conv2d_autodiff to compute forward pass and attach computation node
    return conv2d_autodiff(x, weights_, bias_, kernel_size_);
}

std::vector<Parameter> Conv2D::parameters() {
    return {Parameter{weights_.get()}, Parameter{bias_.get()}};
}

void Conv2D::set_weights(const Tensor& w, const Tensor& b) {
    // Validate shapes
    if (w.shape() != weights_->shape()) {
        throw std::invalid_argument("Conv2D::set_weights: weight shape mismatch");
    }
    if (b.shape() != bias_->shape()) {
        throw std::invalid_argument("Conv2D::set_weights: bias shape mismatch");
    }

    // Copy weights element-by-element
    // Weights shape: {out_channels, in_channels, kernel_size, kernel_size}
    for (size_t oc = 0; oc < out_channels_; ++oc) {
        for (size_t ic = 0; ic < in_channels_; ++ic) {
            for (size_t kh = 0; kh < kernel_size_; ++kh) {
                for (size_t kw = 0; kw < kernel_size_; ++kw) {
                    (*weights_)(oc, ic, kh, kw) = w(oc, ic, kh, kw);
                }
            }
        }
    }

    // Copy bias element-by-element
    // Bias shape: {out_channels, 1}
    for (size_t oc = 0; oc < out_channels_; ++oc) {
        (*bias_)(oc, 0) = b(oc, 0);
    }
}

}  // namespace nrt
