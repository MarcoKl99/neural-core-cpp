#include "nrt/operations.hpp"

#include "nrt/computation_node.hpp"

namespace nrt {

Tensor matmul_autodiff(Tensor& a, Tensor& b) {
    // Regular forward computation
    Tensor result = a.matmul(b);

    // Attach the computational node for backward
    result.creator_node_ =
        ComputationNode{.backward_fn = [&a, &b](Tensor& result_output, const Tensor& grad_result) {
            // Chain rule for matmul:
            //  z = matmul(x, W)
            //  dL/dx = dL/dz * W^T
            //  dL/dW = x^T * dL/dz

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

}  // namespace nrt
