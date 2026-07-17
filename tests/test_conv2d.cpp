#include <catch2/catch_test_macros.hpp>
#include <memory>

#include "nrt/conv2d.hpp"
#include "nrt/tensor.hpp"

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
