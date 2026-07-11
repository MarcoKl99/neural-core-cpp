#include "nrt/activations.hpp"

#include <cmath>
#include <functional>
#include <memory>

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

/*****************/
/*      ReLU     */
/*****************/

Tensor relu(const Tensor& x) {
    return apply_elementwise(x, [](double v) { return v > 0.0 ? v : 0.0; });
}

Tensor relu_derivative(const Tensor& x) {
    // Note that the dervative of ReLU is not defined at x = 0
    // Both conventions d = 0 and d = 1 exist, here d = 1 is chosen
    // Makes to difference in practice anyway, as it is only precisely at 0.0
    return apply_elementwise(x, [](double v) { return v > 0.0 ? 1.0 : 0.0; });
}

std::shared_ptr<Tensor> ReLU::forward(std::shared_ptr<Tensor> x) {
    // Forward: y = relu(x)
    auto result = std::make_shared<Tensor>(relu(*x));

    // Attach computation node for backward
    result->creator_node_ =
        ComputationNode{.inputs = {x},
                        .backward_fn = [](Tensor& output, const Tensor& grad_output,
                                          const std::vector<std::shared_ptr<Tensor>>& inputs) {
                            auto& x = inputs[0];

                            // Chain rule: dL/dx = dL/dy * dy/dx
                            // where dy/dx = relu_derivative(x)

                            Tensor grad_x = grad_output.hadamard(relu_derivative(*x));

                            // Accumulate gradient
                            x->accumulate_gradient(grad_x);

                            // Recurse on input
                            if (x->creator_node_) x->backward_impl(grad_x);
                        }};

    return result;
}

std::vector<Parameter> ReLU::parameters() {
    return {};  // No learnable parameters
}

/*****************/
/*    Sigmoid    */
/*****************/

Tensor sigmoid(const Tensor& x) {
    return apply_elementwise(x, [](double v) { return 1.0 / (1.0 + std::exp(-v)); });
}

Tensor sigmoid_derivative(const Tensor& x) {
    return apply_elementwise(x, [](double v) {
        double s = 1.0 / (1.0 + std::exp(-v));
        return s * (1.0 - s);
    });
}

std::shared_ptr<Tensor> Sigmoid::forward(std::shared_ptr<Tensor> x) {
    // Forward: y = sigmoid(x)
    auto result = std::make_shared<Tensor>(sigmoid(*x));

    // Attach computation node for backward
    result->creator_node_ =
        ComputationNode{.inputs = {x},
                        .backward_fn = [](Tensor& output, const Tensor& grad_output,
                                          const std::vector<std::shared_ptr<Tensor>>& inputs) {
                            auto& x = inputs[0];

                            // Chain rule: dL/dx = dL/dy * dy/dx
                            // where dy/dx = sigmoid_derivative(x)

                            Tensor grad_x = grad_output.hadamard(sigmoid_derivative(*x));

                            // Accumulate gradient
                            x->accumulate_gradient(grad_x);

                            // Recurse on input
                            if (x->creator_node_) x->backward_impl(grad_x);
                        }};

    return result;
}

std::vector<Parameter> Sigmoid::parameters() {
    return {};  // No learnable parameters
}

/*****************/
/*    Softmax    */
/*****************/

Tensor softmax(const Tensor& x) {
    if (x.rank() != 2 || x.shape()[1] != 1) {
        throw std::invalid_argument("softmax: expected column vector shape {n,1}");
    }
    size_t n = x.shape()[0];

    // Numerically stable: subtract the max before exponentiating
    double max_val = x(0, 0);
    for (size_t i = 1; i < n; ++i) {
        if (x(i, 0) > max_val) max_val = x(i, 0);
    }

    Tensor result({n, 1});
    double sum_exp = 0.0;
    for (size_t i = 0; i < n; ++i) {
        result(i, 0) = std::exp(x(i, 0) - max_val);
        sum_exp += result(i, 0);
    }
    for (size_t i = 0; i < n; ++i) result(i, 0) /= sum_exp;

    return result;
}

std::shared_ptr<Tensor> Softmax::forward(std::shared_ptr<Tensor> x) {
    auto result = std::make_shared<Tensor>(softmax(*x));

    result->creator_node_ =
        ComputationNode{.inputs = {x},
                        .backward_fn = [](Tensor& output, const Tensor& grad_output,
                                          const std::vector<std::shared_ptr<Tensor>>& inputs) {
                            auto& x = inputs[0];
                            size_t n = output.shape()[0];

                            // output is y = softmax(x) already - reuse it instead of recomputing.
                            // dL/dx_i = y_i * (dL/dy_i - sum_j(dL/dy_j * y_j))
                            double dot = 0.0;
                            for (size_t i = 0; i < n; ++i) dot += grad_output(i, 0) * output(i, 0);

                            Tensor grad_x(x->shape());
                            for (size_t i = 0; i < n; ++i) {
                                grad_x(i, 0) = output(i, 0) * (grad_output(i, 0) - dot);
                            }

                            x->accumulate_gradient(grad_x);
                            if (x->creator_node_) x->backward_impl(grad_x);
                        }};

    return result;
}

std::vector<Parameter> Softmax::parameters() {
    return {};  // No learnable parameters
}

}  // namespace nrt
