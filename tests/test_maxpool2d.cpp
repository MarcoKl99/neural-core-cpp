#include <catch2/catch_test_macros.hpp>
#include <iostream>

#include "nrt/activations.hpp"
#include "nrt/conv2d.hpp"
#include "nrt/flatten.hpp"
#include "nrt/linear.hpp"
#include "nrt/loss.hpp"
#include "nrt/maxpool2d.hpp"
#include "nrt/sequential.hpp"

TEST_CASE("Integration - MaxPool2D forward pass with known values",
          "[integration][maxpool2d][forward]") {
    nrt::MaxPool2D pool;

    // Input: {1, 1, 4, 4} with values 0 - 15
    //  0  1  2  3
    //  4  5  6  7
    //  8  9 10 11
    // 12 13 14 15
    auto input = std::make_shared<nrt::Tensor>(std::vector<size_t>{1, 1, 4, 4});
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            (*input)(0, 0, i, j) = static_cast<double>(i * 4 + j);
        }
    }

    // Forward: 2×2 pooling with stride 2 -> {1, 1, 2, 2}
    auto output = pool.forward(input);
    REQUIRE(output->shape() == std::vector<size_t>{1, 1, 2, 2});

    // Verify output values (max of each 2×2 patch)
    REQUIRE((*output)(0, 0, 0, 0) == 5.0);
    REQUIRE((*output)(0, 0, 0, 1) == 7.0);
    REQUIRE((*output)(0, 0, 1, 0) == 13.0);
    REQUIRE((*output)(0, 0, 1, 1) == 15.0);
}

TEST_CASE("Integration - MaxPool2D with batch dimension", "[integration][maxpool2d][batch]") {
    nrt::MaxPool2D pool;

    // Input: {2, 2, 4, 4} (batch of 2, 2 channels)
    auto input = std::make_shared<nrt::Tensor>(std::vector<size_t>{2, 2, 4, 4});
    for (size_t b = 0; b < 2; ++b) {
        for (size_t c = 0; c < 2; ++c) {
            for (size_t i = 0; i < 4; ++i) {
                for (size_t j = 0; j < 4; ++j) {
                    (*input)(b, c, i, j) = static_cast<double>(b * 100 + c * 10 + i * 4 + j);
                }
            }
        }
    }

    auto output = pool.forward(input);
    REQUIRE(output->shape() == std::vector<size_t>{2, 2, 2, 2});

    // Check batch 0, channel 0, position (0,0): max(0,1,4,5) = 5
    REQUIRE((*output)(0, 0, 0, 0) == 5.0);

    // Check batch 0, channel 1, position (0,0): max(10,11,14,15) = 15
    REQUIRE((*output)(0, 1, 0, 0) == 15.0);

    // Check batch 1, channel 0, position (0,0): max(100,101,104,105) = 105
    REQUIRE((*output)(1, 0, 0, 0) == 105.0);
}

TEST_CASE("Integration - MaxPool2D gradient flow", "[integration][maxpool2d][backward]") {
    nrt::MaxPool2D pool;

    // Small input: {1, 1, 4, 4}
    //  0  1  2  3
    //  4  5  6  7
    //  8  9 10 11
    // 12 13 14 15
    auto input = std::make_shared<nrt::Tensor>(std::vector<size_t>{1, 1, 4, 4});
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            (*input)(0, 0, i, j) = static_cast<double>(i * 4 + j);
        }
    }

    auto output = pool.forward(input);

    // Initializes the gradients to 1.0
    std::cout << "Calling backward..." << '\n';
    output->backward();

    // Check gradients exist
    auto grad_input = input->gradient();
    REQUIRE(grad_input.shape() == input->shape());

    // Only max positions should have gradients (value 1.0)
    // All others should be 0
    REQUIRE(grad_input(0, 0, 1, 1) == 1.0);
    REQUIRE(grad_input(0, 0, 1, 3) == 1.0);
    REQUIRE(grad_input(0, 0, 3, 1) == 1.0);
    REQUIRE(grad_input(0, 0, 3, 3) == 1.0);

    // Non-max positions should be 0
    REQUIRE(grad_input(0, 0, 0, 0) == 0.0);
    REQUIRE(grad_input(0, 0, 2, 2) == 0.0);
}
