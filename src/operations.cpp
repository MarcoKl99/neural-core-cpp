#include "nrt/operations.hpp"

#include <memory>

#include "nrt/computation_node.hpp"
#include "nrt/tensor.hpp"

namespace nrt {

std::shared_ptr<Tensor> matmul_autodiff(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    // Forward: heap-allocate the result so the graph can co-own it
    auto result = std::make_shared<Tensor>(a->matmul(*b));

    result->creator_node_ =
        ComputationNode{.inputs = {a, b},  // <-- node now OWNS a and b (refcount++)
                        .backward_fn = [](Tensor& result_output, const Tensor& grad_result,
                                          const std::vector<std::shared_ptr<Tensor>>& inputs) {
                            auto& a = inputs[0];
                            auto& b = inputs[1];

                            Tensor grad_a = grad_result.matmul(b->transpose());
                            Tensor grad_b = a->transpose().matmul(grad_result);

                            a->accumulate_gradient(grad_a);
                            b->accumulate_gradient(grad_b);

                            if (a->creator_node_) a->backward_impl(grad_a);
                            if (b->creator_node_) b->backward_impl(grad_b);
                        }};

    return result;
}

std::shared_ptr<Tensor> scalar_mult_autodiff(std::shared_ptr<Tensor> a, double scalar) {
    // Forward: z = a * scalar
    auto result = std::make_shared<Tensor>(*a * scalar);

    // Attach the computational node
    result->creator_node_ = ComputationNode{
        .inputs = {a},
        .backward_fn = [scalar](Tensor& result_output, const Tensor& grad_result,
                                const std::vector<std::shared_ptr<Tensor>>& inputs) {
            auto& a = inputs[0];

            // Chain rule: dL/da = dL/dz * scalar
            Tensor grad_a = grad_result * scalar;

            // Accumulate the gradients
            a->accumulate_gradient(grad_a);

            // Recursion
            if (a->creator_node_) a->backward_impl(grad_a);
        }};

    return result;
}

std::shared_ptr<Tensor> add_autodiff(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    // Forward: z = a + b
    auto result = std::make_shared<Tensor>(*a + *b);

    // Attach computation node for backward
    result->creator_node_ =
        ComputationNode{.inputs = {a, b},
                        .backward_fn = [](Tensor& result_output, const Tensor& grad_result,
                                          const std::vector<std::shared_ptr<Tensor>>& inputs) {
                            auto& a = inputs[0];
                            auto& b = inputs[1];

                            // Chain rule: both inputs get the same gradient
                            // dL/da = dL/dz
                            // dL/db = dL/dz

                            a->accumulate_gradient(grad_result);
                            b->accumulate_gradient(grad_result);

                            // Recurse on inputs
                            if (a->creator_node_) a->backward_impl(grad_result);
                            if (b->creator_node_) b->backward_impl(grad_result);
                        }};

    return result;
}

std::shared_ptr<Tensor> subtract_autodiff(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    // Forward: z = a - b
    auto result = std::make_shared<Tensor>(*a - *b);

    // Attach computation node for backward
    result->creator_node_ =
        ComputationNode{.inputs = {a, b},
                        .backward_fn = [](Tensor& result_output, const Tensor& grad_result,
                                          const std::vector<std::shared_ptr<Tensor>>& inputs) {
                            auto& a = inputs[0];
                            auto& b = inputs[1];

                            // Chain rule:
                            // dL/da = dL/dz
                            // dL/db = -dL/dz (negated)

                            a->accumulate_gradient(grad_result);
                            b->accumulate_gradient(grad_result * -1.0);  // Negate for b

                            // Recurse on inputs
                            if (a->creator_node_) a->backward_impl(grad_result);
                            if (b->creator_node_) b->backward_impl(grad_result * -1.0);
                        }};

    return result;
}

}  // namespace nrt
