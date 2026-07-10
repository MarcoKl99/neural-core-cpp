#include "nrt/operations.hpp"

#include "nrt/computation_node.hpp"

namespace nrt {

Tensor matmul_autodiff(Tensor& a, Tensor& b) {
    // Regular forward computation
    Tensor result = a.matmul(b);

    // Attach the computational node for backward
    result.creator_node_ =
        ComputationNode{.inputs = {&a, &b},
                        .backward_fn = [](Tensor& result_output, const Tensor& grad_result,
                                          const std::vector<Tensor*>& inputs) {
                            // Chain rule for matmul:
                            //  z = matmul(x, W)
                            //  dL/dx = dL/dz * W^T
                            //  dL/dW = x^T * dL/dz
                            Tensor& a = *inputs[0];
                            Tensor& b = *inputs[1];

                            Tensor grad_a = grad_result.matmul(b.transpose());
                            Tensor grad_b = a.transpose().matmul(grad_result);

                            // Accumulate gradients
                            a.accumulate_gradient(grad_a);
                            b.accumulate_gradient(grad_b);

                            // Recurse on inputs
                            if (a.creator_node_) a.backward_impl(grad_a);
                            if (b.creator_node_) b.backward_impl(grad_b);
                        }};

    return result;
}

Tensor scalar_mult_autodiff(Tensor& a, double scalar) {
    // Forward: z = a * scalar
    Tensor result = a * scalar;

    // Attach the computational node
    result.creator_node_ =
        ComputationNode{.inputs = {&a},
                        .backward_fn = [scalar](Tensor& result_output, const Tensor& grad_result,
                                                const std::vector<Tensor*>& inputs) {
                            Tensor& a = *inputs[0];

                            // Chain rule: dL/da = dL/dz * scalar
                            Tensor grad_a = grad_result * scalar;

                            // Accumulate the gradients
                            a.accumulate_gradient(grad_a);

                            // Recursion
                            if (a.creator_node_) a.backward_impl(grad_a);
                        }};

    return result;
}

Tensor add_autodiff(Tensor& a, Tensor& b) {
    // Forward: z = a + b
    Tensor result = a + b;

    // Attach computation node for backward
    result.creator_node_ =
        ComputationNode{.inputs = {&a, &b},
                        .backward_fn = [](Tensor& result_output, const Tensor& grad_result,
                                          const std::vector<Tensor*>& inputs) {
                            Tensor& a = *inputs[0];
                            Tensor& b = *inputs[1];

                            // Chain rule: both inputs get the same gradient
                            // dL/da = dL/dz
                            // dL/db = dL/dz

                            a.accumulate_gradient(grad_result);
                            b.accumulate_gradient(grad_result);

                            // Recurse on inputs
                            if (a.creator_node_) a.backward_impl(grad_result);
                            if (b.creator_node_) b.backward_impl(grad_result);
                        }};

    return result;
}

Tensor subtract_autodiff(Tensor& a, Tensor& b) {
    // Forward: z = a - b
    Tensor result = a - b;

    // Attach computation node for backward
    result.creator_node_ =
        ComputationNode{.inputs = {&a, &b},
                        .backward_fn = [](Tensor& result_output, const Tensor& grad_result,
                                          const std::vector<Tensor*>& inputs) {
                            Tensor& a = *inputs[0];
                            Tensor& b = *inputs[1];

                            // Chain rule:
                            // dL/da = dL/dz
                            // dL/db = -dL/dz (negated)

                            a.accumulate_gradient(grad_result);
                            b.accumulate_gradient(grad_result * -1.0);  // Negate for b

                            // Recurse on inputs
                            if (a.creator_node_) a.backward_impl(grad_result);
                            if (b.creator_node_) b.backward_impl(grad_result * -1.0);
                        }};

    return result;
}

}  // namespace nrt
