#include "nrt/activations.hpp"

#include <cmath>
#include <functional>

#include "nrt/parameter.hpp"

namespace nrt {

namespace {

// Apply the function fn to every element of the passed Tensor (regardless of its rank)
Tensor apply_elementwise(const Tensor& x, const std::function<double(double)>& fn) {
    Tensor result(x.shape());

    if (x.rank() == 1) {
        for (size_t i = 0; i < x.shape()[0]; ++i) {
            result(i) = fn(x(i));
        }
    } else {
        for (size_t i = 0; i < x.shape()[0]; ++i) {
            for (size_t j = 0; j < x.shape()[1]; ++j) {
                result(i, j) = fn(x(i, j));
            }
        }
    }

    return result;
}

}  // namespace

Tensor relu(const Tensor& x) {
    return apply_elementwise(x, [](double v) { return v > 0.0 ? v : 0.0; });
}

Tensor relu_derivative(const Tensor& x) {
    // Note that the dervative of ReLU is not defined at x = 0
    // Both conventions d = 0 and d = 1 exist, here d = 1 is chosen
    // Makes to difference in practice anyway, as it is only precisely at 0.0
    return apply_elementwise(x, [](double v) { return v > 0.0 ? 1.0 : 0.0; });
}

Tensor sigmoid(const Tensor& x) {
    return apply_elementwise(x, [](double v) { return 1.0 / (1.0 + std::exp(-v)); });
}

Tensor sigmoid_derivative(const Tensor& x) {
    return apply_elementwise(x, [](double v) {
        double s = 1.0 / (1.0 + std::exp(-v));
        return s * (1.0 - s);
    });
}

Tensor relu_backward(const Tensor& grad_output, const Tensor& x) {
    // Chain rule elementwise -> Hadamard product
    return grad_output.hadamard(relu_derivative(x));
}

Tensor sigmoid_backward(const Tensor& grad_output, const Tensor& x) {
    // Chain rule elementwise -> Hadamard product
    return grad_output.hadamard(sigmoid_derivative(x));
}

// Implement the class methods for ReLU and Sigmoid
Tensor ReLU::forward(const Tensor& x) {
    return relu(x);
}

std::vector<Parameter> ReLU::parameters() {
    return {};  // No learnable parameters
}

Tensor Sigmoid::forward(const Tensor& x) {
    return sigmoid(x);
}

std::vector<Parameter> Sigmoid::parameters() {
    return {};  // No learnable parameters
}

}  // namespace nrt
