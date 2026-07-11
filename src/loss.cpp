#include "nrt/loss.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>

#include "nrt/activations.hpp"
#include "nrt/computation_node.hpp"

namespace nrt {

double mse(const Tensor& y_hat, const Tensor& y) {
    // Check the shapes
    if (y_hat.shape() != y.shape()) {
        throw std::invalid_argument("mse: shape mismatch");
    }

    // MSE calculation using the hadamard product with itself
    Tensor diff = y_hat - y;
    Tensor squared = diff.hadamard(diff);

    return squared.sum() / static_cast<double>(y_hat.size());
}

Tensor mse_derivative(const Tensor& y_hat, const Tensor& y) {
    // Check the shapes
    if (y_hat.shape() != y.shape()) {
        throw std::invalid_argument("mse_derivative: shape mismatch");
    }

    // Calculate the derivative for both Tensor dimensions (y_hat and y)
    Tensor diff = y_hat - y;
    double scale = 2.0 / static_cast<double>(y_hat.size());

    return diff * scale;
}

std::shared_ptr<Tensor> MSELoss::forward(std::shared_ptr<Tensor> y_hat, std::shared_ptr<Tensor> y) {
    double loss_value = mse(*y_hat, *y);
    auto result = std::make_shared<Tensor>(std::vector<size_t>{1, 1});
    (*result)(0, 0) = loss_value;

    // Attach computation node for backward
    result->creator_node_ =
        ComputationNode{.inputs = {y_hat, y},
                        .backward_fn = [](Tensor& output, const Tensor& grad_output,
                                          const std::vector<std::shared_ptr<Tensor>>& inputs) {
                            auto& y_hat = inputs[0];
                            auto& y = inputs[1];

                            // Upstream gradient seeded to 1.0 when this is the graph root.
                            double upstream = grad_output(0, 0);

                            // Chain rule: dL/dy_hat = upstream * d(mse)/dy_hat
                            //                       = upstream * 2/N * (y_hat - y)
                            Tensor grad_y_hat = mse_derivative(*y_hat, *y) * upstream;

                            // Accumulate gradient into the prediction and recurse.
                            y_hat->accumulate_gradient(grad_y_hat);
                            if (y_hat->creator_node_) y_hat->backward_impl(grad_y_hat);

                            // The target y is a constant
                        }};

    return result;
}

double cross_entropy(const Tensor& logits, size_t target) {
    if (logits.rank() != 2 || logits.shape()[1] != 1) {
        throw std::invalid_argument("cross_entropy: expected column vector shape {n,1}");
    }
    if (target >= logits.shape()[0]) {
        throw std::out_of_range("cross_entropy: target index out of range");
    }

    Tensor probs = softmax(logits);
    constexpr double eps = 1e-12;  // guards log(0) if a probability underflows
    return -std::log(probs(target, 0) + eps);
}

Tensor cross_entropy_grad(const Tensor& logits, size_t target) {
    Tensor grad = softmax(logits);
    grad(target, 0) -= 1.0;  // probs - one_hot(target)
    return grad;
}

std::shared_ptr<Tensor> CrossEntropyLoss::forward(std::shared_ptr<Tensor> y_hat, size_t target) {
    double loss_value = cross_entropy(*y_hat, target);
    auto result = std::make_shared<Tensor>(std::vector<size_t>{1, 1});
    (*result)(0, 0) = loss_value;

    result->creator_node_ = ComputationNode{
        .inputs = {y_hat},
        .backward_fn = [target](Tensor& output, const Tensor& grad_output,
                                const std::vector<std::shared_ptr<Tensor>>& inputs) {
            auto& y_hat = inputs[0];

            double upstream = grad_output(0, 0);
            Tensor grad_logits = cross_entropy_grad(*y_hat, target) * upstream;

            y_hat->accumulate_gradient(grad_logits);
            if (y_hat->creator_node_) y_hat->backward_impl(grad_logits);
        }};

    return result;
}

}  // namespace nrt
