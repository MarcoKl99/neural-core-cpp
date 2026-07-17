#include "nrt/operations.hpp"

#include <iostream>
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

                            // Un-broadcast gradients to match original input shapes
                            Tensor grad_a = Tensor::unbroadcast_gradient(grad_result, a->shape());
                            Tensor grad_b = Tensor::unbroadcast_gradient(grad_result, b->shape());

                            a->accumulate_gradient(grad_a);
                            b->accumulate_gradient(grad_b);

                            // Recurse on inputs
                            if (a->creator_node_) a->backward_impl(grad_a);
                            if (b->creator_node_) b->backward_impl(grad_b);
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

                            // Un-broadcast gradients to match original input shapes
                            Tensor grad_a = Tensor::unbroadcast_gradient(grad_result, a->shape());
                            Tensor grad_b = Tensor::unbroadcast_gradient(grad_result, b->shape());

                            a->accumulate_gradient(grad_a);
                            b->accumulate_gradient(grad_b * -1.0);  // Negate for b

                            // Recurse on inputs
                            if (a->creator_node_) a->backward_impl(grad_a);
                            if (b->creator_node_) b->backward_impl(grad_b * -1.0);
                        }};

    return result;
}

std::shared_ptr<Tensor> reshape_autodiff(std::shared_ptr<Tensor> a,
                                         const std::vector<size_t>& new_shape) {
    // Forward: z = a.reshape(new_shape)
    auto result = std::make_shared<Tensor>(a->reshape(new_shape));

    // Attach computation node for backward
    result->creator_node_ =
        ComputationNode{.inputs = {a},
                        .backward_fn = [](Tensor& result_output, const Tensor& grad_result,
                                          const std::vector<std::shared_ptr<Tensor>>& inputs) {
                            auto& a = inputs[0];

                            // Reshape's local derivative is the identity - it's a pure relabeling
                            // of the same values, so the gradient just needs to be reshaped back
                            // to a's original shape before it flows further upstream.
                            Tensor grad_a = grad_result.reshape(a->shape());

                            a->accumulate_gradient(grad_a);
                            if (a->creator_node_) a->backward_impl(grad_a);
                        }};

    return result;
}

std::shared_ptr<Tensor> transpose_autodiff(std::shared_ptr<Tensor> a) {
    // Forward: transpose the tensor
    auto result = std::make_shared<Tensor>(a->transpose());

    // Attach computation node for backward
    result->creator_node_ =
        ComputationNode{.inputs = {a},
                        .backward_fn = [](Tensor& output, const Tensor& grad_output,
                                          const std::vector<std::shared_ptr<Tensor>>& inputs) {
                            auto& a = inputs[0];

                            // Transpose is its own inverse: transposing the gradient back gives us
                            // grad_a
                            Tensor grad_a = grad_output.transpose();

                            a->accumulate_gradient(grad_a);
                            if (a->creator_node_) a->backward_impl(grad_a);
                        }};

    return result;
}

// Conv2D-Helper 1: Compute gradient w.r.t. bias (simplest!)
static Tensor compute_grad_bias(const Tensor& grad_output) {
    // grad_output shape: {batch, out_channels, out_height, out_width}
    // grad_bias shape: {out_channels}
    // grad_bias(oc) = mean over b of: (sum over h_out, w_out of: grad_output(b, oc, h_out, w_out))

    size_t batch_size = grad_output.shape()[0];
    size_t out_channels = grad_output.shape()[1];
    size_t out_height = grad_output.shape()[2];
    size_t out_width = grad_output.shape()[3];

    Tensor grad_bias({out_channels, 1});

    for (size_t oc = 0; oc < out_channels; ++oc) {
        double sum = 0.0;
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t h_out = 0; h_out < out_height; ++h_out) {
                for (size_t w_out = 0; w_out < out_width; ++w_out) {
                    sum += grad_output(b, oc, h_out, w_out);
                }
            }
        }
        grad_bias(oc, 0) = sum / static_cast<double>(batch_size);  // Average over batch!
    }

    return grad_bias;
}

// Conv2D-Helper 2: Compute gradient w.r.t. weights
static Tensor compute_grad_weights(const Tensor& input, const Tensor& grad_output,
                                   size_t kernel_size) {
    // input: {batch, in_channels, height, width}
    // grad_output: {batch, out_channels, out_height, out_width}
    // returns: {out_channels, in_channels, kernel_size, kernel_size}

    size_t batch = input.shape()[0];
    size_t in_channels = input.shape()[1];
    size_t out_channels = grad_output.shape()[1];
    size_t out_height = grad_output.shape()[2];
    size_t out_width = grad_output.shape()[3];

    Tensor grad_weights({out_channels, in_channels, kernel_size, kernel_size});

    // For each weight element
    for (size_t oc = 0; oc < out_channels; ++oc) {
        for (size_t ic = 0; ic < in_channels; ++ic) {
            for (size_t kh = 0; kh < kernel_size; ++kh) {
                for (size_t kw = 0; kw < kernel_size; ++kw) {
                    double sum = 0.0;

                    // Sum over all batch samples and output positions
                    for (size_t b = 0; b < batch; ++b) {
                        for (size_t h_out = 0; h_out < out_height; ++h_out) {
                            for (size_t w_out = 0; w_out < out_width; ++w_out) {
                                size_t input_h = h_out + kh;
                                size_t input_w = w_out + kw;

                                sum += input(b, ic, input_h, input_w) *
                                       grad_output(b, oc, h_out, w_out);
                            }
                        }
                    }

                    grad_weights(oc, ic, kh, kw) = sum / static_cast<double>(batch);
                }
            }
        }
    }

    return grad_weights;
}

// Conv2D-Helper 3: Compute gradient w.r.t. input
static Tensor compute_grad_input(const Tensor& grad_output, const Tensor& weights,
                                 const std::vector<size_t>& input_shape, size_t kernel_size) {
    // Transposed convolution
    // grad_input(b, ic, h, w) = sum over oc, kh, kw of:
    // grad_output(b, oc, h - kh, w - kw) * weights(oc, ic, kh, kw)
    // (where h - kh and w - kw are valid output positions)

    size_t batch = grad_output.shape()[0];
    size_t in_channels = input_shape[1];
    size_t height = input_shape[2];
    size_t width = input_shape[3];
    size_t out_channels = grad_output.shape()[1];
    size_t out_height = grad_output.shape()[2];
    size_t out_width = grad_output.shape()[3];

    Tensor grad_input(input_shape);

    // For each input position
    // ...did someone see my for-loop?
    for (size_t b = 0; b < batch; ++b) {
        for (size_t ic = 0; ic < in_channels; ++ic) {
            for (size_t h = 0; h < height; ++h) {
                for (size_t w = 0; w < width; ++w) {
                    // So this is now kinda through all input positions
                    double sum = 0.0;

                    // Sum over all output channels and kernel positions
                    for (size_t oc = 0; oc < out_channels; ++oc) {
                        for (size_t kh = 0; kh < kernel_size; ++kh) {
                            for (size_t kw = 0; kw < kernel_size; ++kw) {
                                // Which output position used this input position?
                                int h_out = static_cast<int>(h) - static_cast<int>(kh);
                                int w_out = static_cast<int>(w) - static_cast<int>(kw);

                                // Check if output position is valid
                                if (h_out >= 0 && h_out < static_cast<int>(out_height) &&
                                    w_out >= 0 && w_out < static_cast<int>(out_width)) {
                                    sum +=
                                        grad_output(b, oc, h_out, w_out) * weights(oc, ic, kh, kw);
                                }
                            }
                        }
                    }

                    // Store the grad of this input position in the result
                    grad_input(b, ic, h, w) = sum;
                }
            }
        }
    }

    return grad_input;
}

std::shared_ptr<Tensor> conv2d_autodiff(std::shared_ptr<Tensor> input,
                                        std::shared_ptr<Tensor> weights,
                                        std::shared_ptr<Tensor> bias, size_t kernel_size) {
    // Extract dimensions
    size_t batch = input->shape()[0];
    size_t in_channels = input->shape()[1];
    size_t height = input->shape()[2];
    size_t width = input->shape()[3];
    size_t out_channels = weights->shape()[0];
    size_t out_height = height - kernel_size + 1;
    size_t out_width = width - kernel_size + 1;

    // Forward pass: compute output
    auto output =
        std::make_shared<Tensor>(std::vector<size_t>{batch, out_channels, out_height, out_width});

    // Convolution computation
    for (size_t b = 0; b < batch; ++b) {
        for (size_t oc = 0; oc < out_channels; ++oc) {
            for (size_t h_out = 0; h_out < out_height; ++h_out) {
                for (size_t w_out = 0; w_out < out_width; ++w_out) {
                    // Do this for every output position
                    // -> Calculate the resulting output value
                    double sum = 0.0;

                    for (size_t ic = 0; ic < in_channels; ++ic) {
                        for (size_t kh = 0; kh < kernel_size; ++kh) {
                            for (size_t kw = 0; kw < kernel_size; ++kw) {
                                sum += (*input)(b, ic, h_out + kh, w_out + kw) *
                                       (*weights)(oc, ic, kh, kw);
                            }
                        }
                    }

                    (*output)(b, oc, h_out, w_out) = sum + (*bias)(oc, 0);
                }
            }
        }
    }

    // Create backward function
    auto backward_fn = [input, weights, bias, kernel_size, batch, in_channels](
                           const Tensor& output, const Tensor& grad_output,
                           const std::vector<std::shared_ptr<Tensor>>& inputs) {
        // Compute gradients using helper functions
        Tensor grad_bias = compute_grad_bias(grad_output);
        Tensor grad_weights = compute_grad_weights(*input, grad_output, kernel_size);
        Tensor grad_input = compute_grad_input(grad_output, *weights, input->shape(), kernel_size);

        // Accumulate gradients into input tensors
        inputs[0]->accumulate_gradient(grad_input);    // input gradient
        inputs[1]->accumulate_gradient(grad_weights);  // weights gradient
        inputs[2]->accumulate_gradient(grad_bias);     // bias gradient

        // Recursively backpropagate
        inputs[0]->backward_impl(grad_input);
        inputs[1]->backward_impl(grad_weights);
        inputs[2]->backward_impl(grad_bias);
    };

    // Attach computation node
    output->creator_node_ =
        ComputationNode{.inputs = {input, weights, bias}, .backward_fn = backward_fn};

    return output;
}

}  // namespace nrt
