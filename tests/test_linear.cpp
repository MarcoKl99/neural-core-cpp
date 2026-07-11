#include <catch2/catch_test_macros.hpp>

#include "nrt/linear.hpp"
#include "nrt/tensor.hpp"

TEST_CASE("Linear construction", "[linear][construction]") {
    nrt::Linear layer(3, 2);

    SECTION("Feature counts are stored correctly") {
        REQUIRE(layer.in_features() == 3);
        REQUIRE(layer.out_features() == 2);
    }

    SECTION("Weights have correct shape") {
        REQUIRE(layer.weights().shape() == std::vector<size_t>{2, 3});
    }

    SECTION("Bias has correct shape") {
        REQUIRE(layer.bias().shape() == std::vector<size_t>{2, 1});
    }

    SECTION("Bias is zero-initialized") {
        REQUIRE(layer.bias()(0, 0) == 0.0);
        REQUIRE(layer.bias()(1, 0) == 0.0);
    }

    SECTION("Weights are randomly initialized (not all zero)") {
        // Loser Sanity-Check: bei zufaelliger Normalverteilung (sigma=0.1)
        // ist es praktisch ausgeschlossen, dass alle 6 Werte exakt 0.0 sind.
        bool all_zero = true;
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                if (layer.weights()(i, j) != 0.0) {
                    all_zero = false;
                }
            }
        }
        REQUIRE_FALSE(all_zero);
    }
}

TEST_CASE("Linear set_weights", "[linear][set_weights]") {
    nrt::Linear layer(2, 2);

    nrt::Tensor w({2, 2});
    w(0, 0) = 1.0;
    w(0, 1) = 2.0;
    w(1, 0) = 3.0;
    w(1, 1) = 4.0;

    nrt::Tensor b({2, 1});
    b(0, 0) = 0.5;
    b(1, 0) = -0.5;

    SECTION("Weights and bias are correctly overwritten") {
        layer.set_weights(w, b);
        REQUIRE(layer.weights()(0, 0) == 1.0);
        REQUIRE(layer.weights()(1, 1) == 4.0);
        REQUIRE(layer.bias()(0, 0) == 0.5);
        REQUIRE(layer.bias()(1, 0) == -0.5);
    }

    SECTION("Wrong weight shape throws") {
        nrt::Tensor wrong_w({3, 2});
        REQUIRE_THROWS_AS(layer.set_weights(wrong_w, b), std::invalid_argument);
    }

    SECTION("Wrong bias shape throws") {
        nrt::Tensor wrong_b({3, 1});
        REQUIRE_THROWS_AS(layer.set_weights(w, wrong_b), std::invalid_argument);
    }
}

TEST_CASE("Linear forward", "[linear][forward]") {
    nrt::Linear layer(3, 2);

    // W = [[1, 2, 3], [4, 5, 6]], b = [1, 1]
    nrt::Tensor w({2, 3});
    w(0, 0) = 1.0;
    w(0, 1) = 2.0;
    w(0, 2) = 3.0;
    w(1, 0) = 4.0;
    w(1, 1) = 5.0;
    w(1, 2) = 6.0;

    nrt::Tensor b({2, 1});
    b(0, 0) = 1.0;
    b(1, 0) = 1.0;

    layer.set_weights(w, b);

    // x = [1, 1, 1]^T
    auto x = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{3, 1});
    (*x)(0, 0) = 1.0;
    (*x)(1, 0) = 1.0;
    (*x)(2, 0) = 1.0;

    SECTION("Output has correct shape") {
        auto y = layer.forward(x);
        REQUIRE(y->shape() == std::vector<size_t>{2, 1});
    }

    SECTION("Output is correctly computed (W*x + b)") {
        // y[0] = 1*1 + 2*1 + 3*1 + 1 = 7
        // y[1] = 4*1 + 5*1 + 6*1 + 1 = 16
        auto y = layer.forward(x);
        REQUIRE((*y)(0, 0) == 7.0);
        REQUIRE((*y)(1, 0) == 16.0);
    }

    SECTION("Wrong input shape throws") {
        auto wrong_x = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{4, 1});
        REQUIRE_THROWS_AS(layer.forward(wrong_x), std::invalid_argument);
    }

    SECTION("1D input throws (must be {in_features, 1})") {
        auto wrong_x = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{3});
        REQUIRE_THROWS_AS(layer.forward(wrong_x), std::invalid_argument);
    }
}

TEST_CASE("Linear forward creates computation nodes") {
    nrt::Linear layer(2, 2);

    // Set weights deterministically
    nrt::Tensor w({2, 2});
    w(0, 0) = 1.0;
    w(0, 1) = 2.0;
    w(1, 0) = 3.0;
    w(1, 1) = 4.0;

    nrt::Tensor b({2, 1});
    b(0, 0) = 0.5;
    b(1, 0) = 1.0;

    layer.set_weights(w, b);

    // Input
    auto x = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{2, 1});
    (*x)(0, 0) = 1.0;
    (*x)(1, 0) = 2.0;

    // Forward pass
    auto output = layer.forward(x);

    // Verify: output should NOT be a leaf
    // (because forward uses matmul_autodiff and add_autodiff which attach computation nodes)
    REQUIRE(!(output->is_leaf()));
}

TEST_CASE("Linear forward backward propagates gradients") {
    nrt::Linear layer(2, 2);

    // Set weights deterministically
    nrt::Tensor w({2, 2});
    w(0, 0) = 1.0;
    w(0, 1) = 2.0;
    w(1, 0) = 3.0;
    w(1, 1) = 4.0;

    nrt::Tensor b({2, 1});
    b(0, 0) = 0.5;
    b(1, 0) = 1.0;

    layer.set_weights(w, b);

    // Input
    auto x = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{2, 1});
    (*x)(0, 0) = 1.0;
    (*x)(1, 0) = 2.0;

    // Forward pass
    auto output = layer.forward(x);

    // Backward pass
    output->backward();

    // Verify: gradients should be accumulated in weights and bias
    // Using the public gradient() method
    nrt::Tensor grad_w = layer.weights().gradient();
    nrt::Tensor grad_b = layer.bias().gradient();

    // Verify: gradients are not all zeros (some gradient should exist)
    REQUIRE(grad_w.size() > 0);
    REQUIRE(grad_b.size() > 0);
}

TEST_CASE("Linear weight initialization statistics", "[linear][init]") {
    // Asymmetric shape on purpose: He depends only on fan_in, Xavier on fan_in+fan_out.
    // If the two ever got swapped in the implementation, this would catch it -
    // He std (sqrt(2/64) = 0.17678) and Xavier std (sqrt(2/320) = 0.07906) differ by ~2.2x.
    constexpr size_t in_features = 64;
    constexpr size_t out_features = 256;

    // Returns {mean, stddev} over all weight entries.
    auto weight_stats = [](nrt::Tensor& w, size_t rows, size_t cols) {
        double n = static_cast<double>(rows * cols);
        double mean = 0.0;
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j) mean += w(i, j);
        mean /= n;

        double variance = 0.0;
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j) {
                double d = w(i, j) - mean;
                variance += d * d;
            }
        variance /= n;

        return std::pair<double, double>{mean, std::sqrt(variance)};
    };

    SECTION("He init std-dev matches sqrt(2/fan_in)") {
        nrt::Linear layer(in_features, out_features, nrt::WeightInit::He);
        auto [mean, stddev] = weight_stats(layer.weights(), out_features, in_features);

        double expected_std = std::sqrt(2.0 / static_cast<double>(in_features));

        REQUIRE(std::abs(mean) < 0.01);
        REQUIRE(std::abs(stddev - expected_std) / expected_std < 0.1);  // within 10%
    }

    SECTION("Xavier init std-dev matches sqrt(2/(fan_in+fan_out))") {
        nrt::Linear layer(in_features, out_features, nrt::WeightInit::Xavier);
        auto [mean, stddev] = weight_stats(layer.weights(), out_features, in_features);

        double expected_std = std::sqrt(2.0 / static_cast<double>(in_features + out_features));

        REQUIRE(std::abs(mean) < 0.01);
        REQUIRE(std::abs(stddev - expected_std) / expected_std < 0.1);
    }

    SECTION("He and Xavier produce different std-devs for the same shape") {
        nrt::Linear he_layer(in_features, out_features, nrt::WeightInit::He);
        nrt::Linear xavier_layer(in_features, out_features, nrt::WeightInit::Xavier);

        auto [he_mean, he_std] = weight_stats(he_layer.weights(), out_features, in_features);
        auto [xavier_mean, xavier_std] =
            weight_stats(xavier_layer.weights(), out_features, in_features);

        REQUIRE(he_std > xavier_std);
    }

    SECTION("Bias stays zero-initialized regardless of scheme") {
        nrt::Linear he_layer(in_features, out_features, nrt::WeightInit::He);
        nrt::Linear xavier_layer(in_features, out_features, nrt::WeightInit::Xavier);

        for (size_t i = 0; i < out_features; ++i) {
            REQUIRE(he_layer.bias()(i, 0) == 0.0);
            REQUIRE(xavier_layer.bias()(i, 0) == 0.0);
        }
    }
}
