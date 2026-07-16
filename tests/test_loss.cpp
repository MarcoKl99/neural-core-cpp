#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "nrt/activations.hpp"
#include "nrt/loss.hpp"
#include "nrt/tensor.hpp"

TEST_CASE("mse", "[loss][mse]") {
    SECTION("Identical tensors give zero loss") {
        nrt::Tensor y_hat({3});
        nrt::Tensor y({3});
        y_hat(0) = 1.0;
        y_hat(1) = 2.0;
        y_hat(2) = 3.0;
        y(0) = 1.0;
        y(1) = 2.0;
        y(2) = 3.0;

        REQUIRE(nrt::mse(y_hat, y) == 0.0);
    }

    SECTION("Known value, 1D") {
        // y_hat = [1, 2], y = [3, 4]
        // diff = [-2, -2], squared = [4, 4], sum = 8, mse = 8/2 = 4
        nrt::Tensor y_hat({2});
        nrt::Tensor y({2});
        y_hat(0) = 1.0;
        y_hat(1) = 2.0;
        y(0) = 3.0;
        y(1) = 4.0;

        REQUIRE(nrt::mse(y_hat, y) == 4.0);
    }

    SECTION("Known value, 2D") {
        // y_hat = [[0, 0], [0, 0]], y = [[1, 1], [1, 1]]
        // diff = -1 everywhere, squared = 1 everywhere, sum = 4, mse = 4/4 = 1
        nrt::Tensor y_hat({2, 2});
        nrt::Tensor y({2, 2});
        y(0, 0) = 1.0;
        y(0, 1) = 1.0;
        y(1, 0) = 1.0;
        y(1, 1) = 1.0;

        REQUIRE(nrt::mse(y_hat, y) == 1.0);
    }

    SECTION("Loss is symmetric in sign: mse(a, b) == mse(b, a)") {
        nrt::Tensor a({2});
        nrt::Tensor b({2});
        a(0) = 1.0;
        a(1) = 5.0;
        b(0) = 3.0;
        b(1) = 2.0;

        REQUIRE(nrt::mse(a, b) == nrt::mse(b, a));
    }

    SECTION("Shape mismatch throws") {
        nrt::Tensor y_hat({2});
        nrt::Tensor y({3});
        REQUIRE_THROWS_AS(nrt::mse(y_hat, y), std::invalid_argument);
    }

    SECTION("Rank mismatch throws") {
        nrt::Tensor y_hat({2, 2});
        nrt::Tensor y({4});
        REQUIRE_THROWS_AS(nrt::mse(y_hat, y), std::invalid_argument);
    }
}

TEST_CASE("mse_derivative", "[loss][mse][backward]") {
    SECTION("Identical tensors give zero gradient") {
        nrt::Tensor y_hat({3});
        nrt::Tensor y({3});
        y_hat(0) = 1.0;
        y_hat(1) = 2.0;
        y_hat(2) = 3.0;
        y(0) = 1.0;
        y(1) = 2.0;
        y(2) = 3.0;

        nrt::Tensor grad = nrt::mse_derivative(y_hat, y);
        REQUIRE(grad(0) == 0.0);
        REQUIRE(grad(1) == 0.0);
        REQUIRE(grad(2) == 0.0);
    }

    SECTION("Known value, 1D") {
        // y_hat = [1, 2], y = [3, 4], n = 2
        // grad = (2/2) * (y_hat - y) = [1*(1-3), 1*(2-4)] = [-2, -2]
        nrt::Tensor y_hat({2});
        nrt::Tensor y({2});
        y_hat(0) = 1.0;
        y_hat(1) = 2.0;
        y(0) = 3.0;
        y(1) = 4.0;

        nrt::Tensor grad = nrt::mse_derivative(y_hat, y);
        REQUIRE(grad(0) == -2.0);
        REQUIRE(grad(1) == -2.0);
    }

    SECTION("Known value, 2D") {
        // y_hat = [[0,0],[0,0]], y = [[1,1],[1,1]], n = 4
        // grad = (2/4) * (0 - 1) = -0.5 everywhere
        nrt::Tensor y_hat({2, 2});
        nrt::Tensor y({2, 2});
        y(0, 0) = 1.0;
        y(0, 1) = 1.0;
        y(1, 0) = 1.0;
        y(1, 1) = 1.0;

        nrt::Tensor grad = nrt::mse_derivative(y_hat, y);
        REQUIRE(grad(0, 0) == -0.5);
        REQUIRE(grad(0, 1) == -0.5);
        REQUIRE(grad(1, 0) == -0.5);
        REQUIRE(grad(1, 1) == -0.5);
    }

    SECTION("Gradient shape matches input shape") {
        nrt::Tensor y_hat({3, 1});
        nrt::Tensor y({3, 1});
        nrt::Tensor grad = nrt::mse_derivative(y_hat, y);
        REQUIRE(grad.shape() == std::vector<size_t>{3, 1});
    }

    SECTION("Shape mismatch throws") {
        nrt::Tensor y_hat({2});
        nrt::Tensor y({3});
        REQUIRE_THROWS_AS(nrt::mse_derivative(y_hat, y), std::invalid_argument);
    }
}

TEST_CASE("cross_entropy", "[loss][cross_entropy]") {
    SECTION("Known value") {
        // logits = [0, 1], target = 1
        // probs = softmax([0,1]) = [1/(1+e), e/(1+e)], loss = -log(probs[1])
        nrt::Tensor logits({2, 1});
        logits(0, 0) = 0.0;
        logits(1, 0) = 1.0;

        // This is the reformulated result of the cross-entropy applied to the softmax
        // Tests, if the softmax is applied internally correctly
        double expected = std::log(1.0 + std::exp(1.0)) - 1.0;
        REQUIRE(std::abs(nrt::cross_entropy(logits, 1) - expected) < 1e-9);
    }

    SECTION("Confident correct prediction gives near-zero loss") {
        nrt::Tensor logits({3, 1});
        logits(0, 0) = 10.0;
        logits(1, 0) = -10.0;
        logits(2, 0) = -10.0;

        REQUIRE(nrt::cross_entropy(logits, 0) < 1e-6);
    }

    SECTION("Confident wrong prediction gives large loss") {
        nrt::Tensor logits({3, 1});
        logits(0, 0) = 10.0;
        logits(1, 0) = -10.0;
        logits(2, 0) = -10.0;

        REQUIRE(nrt::cross_entropy(logits, 1) > 10.0);
    }

    SECTION("Uniform logits over n classes give loss = log(n)") {
        nrt::Tensor logits({4, 1});
        logits(0, 0) = logits(1, 0) = logits(2, 0) = logits(3, 0) = 0.0;

        REQUIRE(std::abs(nrt::cross_entropy(logits, 2) - std::log(4.0)) < 1e-9);
    }

    SECTION("Non-column-vector shape throws") {
        nrt::Tensor logits({2, 2});
        REQUIRE_THROWS_AS(nrt::cross_entropy(logits, 0), std::invalid_argument);
    }

    SECTION("Out-of-range target throws") {
        nrt::Tensor logits({3, 1});
        REQUIRE_THROWS_AS(nrt::cross_entropy(logits, 5), std::out_of_range);
    }
}

TEST_CASE("cross_entropy_grad", "[loss][cross_entropy][backward]") {
    SECTION("Equals probs - one_hot(target)") {
        nrt::Tensor logits({3, 1});
        logits(0, 0) = 1.0;
        logits(1, 0) = 2.0;
        logits(2, 0) = 0.5;
        const size_t target = 1;

        nrt::Tensor grad = nrt::cross_entropy_grad(logits, target);
        nrt::Tensor probs = nrt::softmax(logits);

        for (size_t i = 0; i < 3; ++i) {
            double expected = probs(i, 0) - (i == target ? 1.0 : 0.0);
            REQUIRE(std::abs(grad(i, 0) - expected) < 1e-9);
        }
    }

    SECTION("Gradient components sum to zero") {
        // probs sums to 1, one_hot sums to 1 -> (probs - one_hot) sums to 0.
        nrt::Tensor logits({4, 1});
        logits(0, 0) = -1.0;
        logits(1, 0) = 2.0;
        logits(2, 0) = 0.3;
        logits(3, 0) = 1.7;

        nrt::Tensor grad = nrt::cross_entropy_grad(logits, 2);
        REQUIRE(std::abs(grad.sum()) < 1e-9);
    }
}

TEST_CASE("CrossEntropyLoss with batches", "[loss][cross_entropy][batch]") {
    nrt::CrossEntropyLoss loss_fn;

    SECTION("Forward pass with batch_size=2, num_classes=3") {
        // Logits: {batch=2, num_classes=3}
        auto logits = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{2, 3});
        (*logits)(0, 0) = 1.0;
        (*logits)(0, 1) = 2.0;
        (*logits)(0, 2) = 0.5;
        (*logits)(1, 0) = 0.1;
        (*logits)(1, 1) = 0.2;
        (*logits)(1, 2) = 3.0;

        // Targets: {batch=2, 1} with class indices
        auto targets = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{2, 1});
        (*targets)(0, 0) = 1.0;  // Sample 0: class 1
        (*targets)(1, 0) = 2.0;  // Sample 1: class 2

        auto loss = loss_fn.forward(logits, targets);

        // Loss should be scalar {1, 1}
        REQUIRE(loss->shape() == std::vector<size_t>{1, 1});
        // Loss should be positive and finite
        REQUIRE((*loss)(0, 0) > 0.0);
        REQUIRE(std::isfinite((*loss)(0, 0)));
    }

    SECTION("Backward pass propagates gradients correctly") {
        auto logits = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{2, 3});
        (*logits)(0, 0) = 1.0;
        (*logits)(0, 1) = 2.0;
        (*logits)(0, 2) = 0.5;
        (*logits)(1, 0) = 0.1;
        (*logits)(1, 1) = 0.2;
        (*logits)(1, 2) = 3.0;

        auto targets = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{2, 1});
        (*targets)(0, 0) = 1.0;
        (*targets)(1, 0) = 2.0;

        logits->zero_grad();
        auto loss = loss_fn.forward(logits, targets);
        loss->backward();

        nrt::Tensor grad_logits = logits->gradient();

        // Gradient should match logits' shape
        REQUIRE(grad_logits.shape() == std::vector<size_t>{2, 3});
        // Gradients should be non-zero
        REQUIRE(grad_logits(0, 1) != 0.0);
        REQUIRE(grad_logits(1, 2) != 0.0);
    }

    SECTION("Batch size mismatch throws") {
        auto logits = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{2, 3});
        auto targets =
            std::make_shared<nrt::Tensor>(std::vector<std::size_t>{3, 1});  // Wrong batch size

        REQUIRE_THROWS_AS(loss_fn.forward(logits, targets), std::invalid_argument);
    }

    SECTION("Target class index out of range throws") {
        auto logits = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{2, 3});
        auto targets = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{2, 1});
        (*targets)(0, 0) = 5.0;  // Class 5 doesn't exist (only 0, 1, 2)

        REQUIRE_THROWS_AS(loss_fn.forward(logits, targets), std::out_of_range);
    }
}
