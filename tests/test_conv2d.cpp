#include <catch2/catch_test_macros.hpp>
#include <memory>

#include "nrt/activations.hpp"
#include "nrt/conv2d.hpp"
#include "nrt/flatten.hpp"
#include "nrt/loss.hpp"
#include "nrt/sequential.hpp"
#include "nrt/tensor.hpp"

namespace {

// Helper: Check that gradient tensor is not all zeros
bool gradients_not_all_zero(const nrt::Tensor& grad, double threshold = 1e-10) {
    double grad_abs_sum = 0.0;

    for (size_t i = 0; i < grad.size(); ++i) {
        // Convert flat index to multi-dimensional coordinates
        std::vector<size_t> indices(grad.rank());
        size_t idx = i;
        for (int d = grad.rank() - 1; d >= 0; --d) {
            indices[d] = idx % grad.shape()[d];
            idx /= grad.shape()[d];
        }

        grad_abs_sum += std::abs(grad.at(indices));
    }

    return grad_abs_sum > threshold;
}

}  // namespace

TEST_CASE("Conv2D output shape", "[conv2d][forward]") {
    // Create a simple conv layer: 1 input channel, 1 output channel, 3x3 kernel
    nrt::Conv2D conv(1, 1, 3);

    // Input: {batch=2, in_channels=1, height=5, width=5}
    auto input = std::make_shared<nrt::Tensor>(std::vector<size_t>{2, 1, 5, 5});

    auto output = conv.forward(input);

    // Expected output shape: {2, 1, 3, 3} (5 - 3 + 1 = 3)
    REQUIRE(output->shape() == std::vector<size_t>{2, 1, 3, 3});
}

TEST_CASE("Conv2D forward with concrete 2x2 kernel on 4x4 input", "[conv2d][forward][concrete]") {
    // Create layer: 1 in_channel, 1 out_channel, 2x2 kernel
    nrt::Conv2D conv(1, 1, 2);

    // Set specific weights
    nrt::Tensor w({1, 1, 2, 2});
    w(0, 0, 0, 0) = 1.0;  // top-left
    w(0, 0, 0, 1) = 0.0;  // top-right
    w(0, 0, 1, 0) = 0.0;  // bottom-left
    w(0, 0, 1, 1) = 1.0;  // bottom-right

    nrt::Tensor b({1, 1});
    b(0, 0) = 0.0;

    conv.set_weights(w, b);

    // Create input {1, 1, 4, 4}
    // 1  2  3  4
    // 5  6  7  8
    // 9  10 11 12
    // 13 14 15 16
    auto input = std::make_shared<nrt::Tensor>(std::vector<size_t>{1, 1, 4, 4});
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            (*input)(0, 0, i, j) = static_cast<double>(i * 4 + j + 1);
        }
    }

    // Forward pass
    auto output = conv.forward(input);

    // Verify output shape
    REQUIRE(output->shape() == std::vector<size_t>{1, 1, 3, 3});

    // Verify specific output values
    REQUIRE((*output)(0, 0, 0, 0) == 7.0);
    REQUIRE((*output)(0, 0, 0, 1) == 9.0);
    REQUIRE((*output)(0, 0, 0, 2) == 11.0);

    REQUIRE((*output)(0, 0, 1, 0) == 15.0);
    REQUIRE((*output)(0, 0, 1, 1) == 17.0);
    REQUIRE((*output)(0, 0, 1, 2) == 19.0);

    REQUIRE((*output)(0, 0, 2, 0) == 23.0);
    REQUIRE((*output)(0, 0, 2, 1) == 25.0);
    REQUIRE((*output)(0, 0, 2, 2) == 27.0);
}

TEST_CASE("Conv2D forward with concrete 2x2 kernel on 4x4 input - 2 output channels",
          "[conv2d][forward][concrete]") {
    // Create layer: 1 in_channel, 2 out_channels, 2x2 kernel
    nrt::Conv2D conv(1, 2, 2);

    // Set specific weights
    nrt::Tensor w({2, 1, 2, 2});
    w(0, 0, 0, 0) = 1.0;
    w(0, 0, 0, 1) = 0.0;
    w(0, 0, 1, 0) = 0.0;
    w(0, 0, 1, 1) = 1.0;

    w(1, 0, 0, 0) = 1.0;
    w(1, 0, 0, 1) = 1.0;
    w(1, 0, 1, 0) = 1.0;
    w(1, 0, 1, 1) = 1.0;

    nrt::Tensor b({2, 1});
    b(0, 0) = 0.0;
    b(1, 0) = 0.0;

    conv.set_weights(w, b);

    // Create input {1, 1, 4, 4}
    // 1  2  3  4
    // 5  6  7  8
    // 9  10 11 12
    // 13 14 15 16
    auto input = std::make_shared<nrt::Tensor>(std::vector<size_t>{1, 1, 4, 4});
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            (*input)(0, 0, i, j) = static_cast<double>(i * 4 + j + 1);
        }
    }

    // Forward pass
    auto output = conv.forward(input);

    // Verify output shape
    REQUIRE(output->shape() == std::vector<size_t>{1, 2, 3, 3});

    // Output channel 1
    REQUIRE((*output)(0, 0, 0, 0) == 7.0);
    REQUIRE((*output)(0, 0, 0, 1) == 9.0);
    REQUIRE((*output)(0, 0, 0, 2) == 11.0);

    REQUIRE((*output)(0, 0, 1, 0) == 15.0);
    REQUIRE((*output)(0, 0, 1, 1) == 17.0);
    REQUIRE((*output)(0, 0, 1, 2) == 19.0);

    REQUIRE((*output)(0, 0, 2, 0) == 23.0);
    REQUIRE((*output)(0, 0, 2, 1) == 25.0);
    REQUIRE((*output)(0, 0, 2, 2) == 27.0);

    // Output channel 2
    REQUIRE((*output)(0, 1, 0, 0) == 14.0);
    REQUIRE((*output)(0, 1, 0, 1) == 18.0);
    REQUIRE((*output)(0, 1, 0, 2) == 22.0);

    REQUIRE((*output)(0, 1, 1, 0) == 30.0);
    REQUIRE((*output)(0, 1, 1, 1) == 34.0);
    REQUIRE((*output)(0, 1, 1, 2) == 38.0);

    REQUIRE((*output)(0, 1, 2, 0) == 46.0);
    REQUIRE((*output)(0, 1, 2, 1) == 50.0);
    REQUIRE((*output)(0, 1, 2, 2) == 54.0);
}

TEST_CASE("Conv2D gradient flow (backward pass)", "[conv2d][backward]") {
    // Create simple conv: 1 in, 1 out, 2x2 kernel
    nrt::Conv2D conv(1, 1, 2);

    // Set weights: identity kernel
    nrt::Tensor w({1, 1, 2, 2});
    w(0, 0, 0, 0) = 1.0;
    w(0, 0, 0, 1) = 0.0;
    w(0, 0, 1, 0) = 0.0;
    w(0, 0, 1, 1) = 1.0;

    nrt::Tensor b({1, 1});
    b(0, 0) = 0.0;

    conv.set_weights(w, b);

    // Create input {1, 1, 4, 4}
    // 1  2  3  4
    // 5  6  7  8
    // 9  10 11 12
    // 13 14 15 16
    auto input = std::make_shared<nrt::Tensor>(std::vector<size_t>{1, 1, 4, 4});
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            (*input)(0, 0, i, j) = static_cast<double>(i * 4 + j + 1);
        }
    }

    // Forward pass
    auto output = conv.forward(input);

    // Create gradient tensor {1, 1, 3, 3} with all ones
    nrt::Tensor grad_output({1, 1, 3, 3});
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            grad_output(0, 0, i, j) = 1.0;
        }
    }

    // Backward pass
    output->backward_impl(grad_output);

    // Verify bias gradient: should be sum of all grad_output values = 9.0
    auto grad_bias = conv.bias().gradient();
    REQUIRE(grad_bias.shape() == std::vector<size_t>{1, 1});
    REQUIRE(grad_bias(0, 0) == 9.0);

    // Verify weights gradient
    auto grad_weights = conv.weights().gradient();
    REQUIRE(grad_weights.shape() == std::vector<size_t>{1, 1, 2, 2});
    // Note that all positions would be multiplied with the incoming gradient positions here,
    // but all are 1.0 in this simple example
    // grad_weights(0, 0, 0, 0) = sum of top-left 3x3 patch = 1+2+3+5+6+7+9+10+11 = 54.0
    REQUIRE(grad_weights(0, 0, 0, 0) == 54.0);
    // grad_weights(0, 0, 1, 1) = sum of bottom-right 3x3 patch = 6+7+8+10+11+12+14+15+16 = 99.0
    REQUIRE(grad_weights(0, 0, 1, 1) == 99.0);

    // Verify input gradient
    auto grad_input = input->gradient();
    REQUIRE(grad_input.shape() == std::vector<size_t>{1, 1, 4, 4});
    // grad_input(0, 0, 1, 1) receives from outputs at (0,0) and (1,1)
    // = grad_output(0,0,0,0)*weights(0,0,1,1) + grad_output(0,0,1,1)*weights(0,0,0,0)
    // = 1.0*1.0 + 1.0*1.0 = 2.0
    REQUIRE(grad_input(0, 0, 1, 1) == 2.0);
}

TEST_CASE("Integration - Conv2D + ReLU forward pass", "[integration][conv2d][relu]") {
    nrt::Conv2D conv(1, 8, 3);
    nrt::ReLU relu;

    // Input: {1, 1, 5, 5}
    auto input = std::make_shared<nrt::Tensor>(std::vector<size_t>{1, 1, 5, 5});
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            (*input)(0, 0, i, j) =
                static_cast<double>(i * 5 + j) - 10.0;  // Mix of positive/negative
        }
    }

    // Forward through Conv2D + ReLU
    auto conv_out = conv.forward(input);
    REQUIRE(conv_out->shape() == std::vector<size_t>{1, 8, 3, 3});

    auto relu_out = relu.forward(conv_out);
    REQUIRE(relu_out->shape() == std::vector<size_t>{1, 8, 3, 3});

    // Verify ReLU applied (all values >= 0)
    bool all_non_negative = true;
    for (size_t i = 0; i < 1; ++i) {
        for (size_t c = 0; c < 8; ++c) {
            for (size_t h = 0; h < 3; ++h) {
                for (size_t w = 0; w < 3; ++w) {
                    if ((*relu_out)(i, c, h, w) < 0.0) {
                        all_non_negative = false;
                    }
                }
            }
        }
    }
    REQUIRE(all_non_negative);
}

TEST_CASE("Integration - Conv2D + ReLU + Flatten + Linear",
          "[integration][conv2d][full_pipeline]") {
    // Small network: Conv2D(1,4,3) -> ReLU -> Flatten -> Linear(4*3*3, 2)
    std::vector<std::unique_ptr<nrt::Module>> modules;
    modules.push_back(std::make_unique<nrt::Conv2D>(1, 4, 3));
    modules.push_back(std::make_unique<nrt::ReLU>());
    modules.push_back(std::make_unique<nrt::Flatten>());
    modules.push_back(std::make_unique<nrt::Linear>(4 * 3 * 3, 2));
    nrt::Sequential model(std::move(modules));

    // Input: {2, 1, 5, 5}  (batch of 2)
    auto input = std::make_shared<nrt::Tensor>(std::vector<size_t>{2, 1, 5, 5});
    for (size_t b = 0; b < 2; ++b) {
        for (size_t i = 0; i < 5; ++i) {
            for (size_t j = 0; j < 5; ++j) {
                (*input)(b, 0, i, j) = static_cast<double>(b * 25 + i * 5 + j);
            }
        }
    }

    // Forward pass
    auto output = model.forward(input);
    REQUIRE(output->shape() == std::vector<size_t>{2, 2});
}

TEST_CASE("Integration - Conv2D gradient flow through ReLU", "[integration][conv2d][backward]") {
    unsigned int seed = 123;

    std::vector<std::unique_ptr<nrt::Module>> modules;
    modules.push_back(std::make_unique<nrt::Conv2D>(1, 4, 3, nrt::WeightInit::He, seed));
    modules.push_back(std::make_unique<nrt::ReLU>());
    modules.push_back(std::make_unique<nrt::Flatten>());
    modules.push_back(std::make_unique<nrt::Linear>(4 * 3 * 3, 2, nrt::WeightInit::Xavier, seed));
    nrt::Sequential model(std::move(modules));

    // Input
    auto input = std::make_shared<nrt::Tensor>(std::vector<size_t>{1, 1, 5, 5});
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            (*input)(0, 0, i, j) = static_cast<double>(i * 5 + j);
        }
    }

    // Labels
    auto labels = std::make_shared<nrt::Tensor>(std::vector<size_t>{1, 1});
    (*labels)(0, 0) = 1.0;

    // Forward + backward
    auto logits = model.forward(input);
    nrt::CrossEntropyLoss loss_fn;
    auto loss = loss_fn.forward(logits, labels);
    loss->backward();

    // Verify input gradient
    auto grad_input = input->gradient();
    REQUIRE(grad_input.shape() == input->shape());
    REQUIRE(gradients_not_all_zero(grad_input));

    // Verify Conv2D weights gradient
    auto conv = dynamic_cast<nrt::Conv2D*>(model.get(0));
    REQUIRE(gradients_not_all_zero(conv->weights().gradient()));

    // Verify Linear weights gradient
    auto linear = dynamic_cast<nrt::Linear*>(model.get(3));
    REQUIRE(gradients_not_all_zero(linear->weights().gradient()));
}
