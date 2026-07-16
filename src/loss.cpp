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

double cross_entropy_batch(const Tensor& logits, const Tensor& targets) {
    // logits: {batch, num_classes}
    // targets: {batch, 1} with class indices
    if (logits.rank() != 2 || targets.rank() != 2) {
        throw std::invalid_argument("cross_entropy_batch: both must be rank 2");
    }
    if (logits.shape()[0] != targets.shape()[0]) {
        throw std::invalid_argument("cross_entropy_batch: batch size mismatch");
    }
    if (targets.shape()[1] != 1) {
        throw std::invalid_argument("cross_entropy_batch: targets must be {batch, 1}");
    }

    size_t batch_size = logits.shape()[0];
    double total_loss = 0.0;

    // Compute loss for each sample in the batch
    for (size_t b = 0; b < batch_size; ++b) {
        size_t target_class = static_cast<size_t>(targets(b, 0));
        if (target_class >= logits.shape()[1]) {
            throw std::out_of_range("cross_entropy_batch: target class index out of range");
        }

        // Compute softmax over logits[b]
        Tensor logits_b({logits.shape()[1], 1});
        for (size_t c = 0; c < logits.shape()[1]; ++c) {
            logits_b(c, 0) = logits(b, c);
        }
        Tensor probs = softmax(logits_b);

        // Cross-entropy for this sample
        constexpr double eps = 1e-12;
        total_loss += -std::log(probs(target_class, 0) + eps);
    }

    return total_loss / static_cast<double>(batch_size);
}

Tensor cross_entropy_batch_grad(const Tensor& logits, const Tensor& targets) {
    // Gradient: softmax(logits) - one_hot(targets), per sample, then averaged
    if (logits.rank() != 2 || targets.rank() != 2) {
        throw std::invalid_argument("cross_entropy_batch_grad: both must be rank 2");
    }
    if (logits.shape()[0] != targets.shape()[0]) {
        throw std::invalid_argument("cross_entropy_batch_grad: batch size mismatch");
    }

    size_t batch_size = logits.shape()[0];
    size_t num_classes = logits.shape()[1];
    Tensor grad({batch_size, num_classes});

    // For each sample: grad[b] = (softmax(logits[b]) - one_hot(targets[b])) / batch_size
    for (size_t b = 0; b < batch_size; ++b) {
        size_t target_class = static_cast<size_t>(targets(b, 0));

        // Extract logits for this sample and compute softmax
        Tensor logits_b({num_classes, 1});
        for (size_t c = 0; c < num_classes; ++c) {
            logits_b(c, 0) = logits(b, c);
        }
        Tensor probs = softmax(logits_b);

        // grad[b] = (probs - one_hot) / batch_size
        for (size_t c = 0; c < num_classes; ++c) {
            double one_hot_val = (c == target_class) ? 1.0 : 0.0;
            grad(b, c) = (probs(c, 0) - one_hot_val) / static_cast<double>(batch_size);
        }
    }

    return grad;
}

std::shared_ptr<Tensor> CrossEntropyLoss::forward(std::shared_ptr<Tensor> logits,
                                                  std::shared_ptr<Tensor> targets) {
    double loss_value = cross_entropy_batch(*logits, *targets);
    auto result = std::make_shared<Tensor>(std::vector<size_t>{1, 1});
    (*result)(0, 0) = loss_value;

    result->creator_node_ = ComputationNode{
        .inputs = {logits},
        .backward_fn = [targets](Tensor& output, const Tensor& grad_output,
                                 const std::vector<std::shared_ptr<Tensor>>& inputs) {
            auto& logits = inputs[0];

            double upstream = grad_output(0, 0);
            Tensor grad_logits = cross_entropy_batch_grad(*logits, *targets) * upstream;

            logits->accumulate_gradient(grad_logits);
            if (logits->creator_node_) logits->backward_impl(grad_logits);
        }};

    return result;
}

}  // namespace nrt
