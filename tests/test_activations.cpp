#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "nrt/activations.hpp"
#include "nrt/tensor.hpp"

namespace {

// Helper function for comparing tensors with tolerance
bool tensors_approx_equal(const nrt::Tensor& a, const nrt::Tensor& b, double tol = 1e-9) {
    if (a.shape() != b.shape()) return false;

    for (size_t i = 0; i < a.size(); ++i) {
        double a_val = a(i / a.shape()[1], i % a.shape()[1]);
        double b_val = b(i / b.shape()[1], i % b.shape()[1]);
        if (std::abs(a_val - b_val) > tol) {
            return false;
        }
    }
    return true;
}

}  // namespace

TEST_CASE("relu", "[activations][relu]") {
    SECTION("1D: positive, negative and zero values") {
        nrt::Tensor x({4});
        x(0) = 3.0;
        x(1) = -2.0;
        x(2) = 0.0;
        x(3) = -0.001;

        nrt::Tensor y = nrt::relu(x);
        REQUIRE(y(0) == 3.0);
        REQUIRE(y(1) == 0.0);
        REQUIRE(y(2) == 0.0);
        REQUIRE(y(3) == 0.0);
    }

    SECTION("2D: shape is preserved") {
        nrt::Tensor x({2, 2});
        x(0, 0) = -1.0;
        x(0, 1) = 2.0;
        x(1, 0) = -3.0;
        x(1, 1) = 4.0;

        nrt::Tensor y = nrt::relu(x);
        REQUIRE(y.shape() == std::vector<size_t>{2, 2});
        REQUIRE(y(0, 0) == 0.0);
        REQUIRE(y(0, 1) == 2.0);
        REQUIRE(y(1, 0) == 0.0);
        REQUIRE(y(1, 1) == 4.0);
    }

    SECTION("Original tensor remains unchanged") {
        nrt::Tensor x({2});
        x(0) = -5.0;
        x(1) = 5.0;
        nrt::Tensor y = nrt::relu(x);
        REQUIRE(x(0) == -5.0);
        REQUIRE(x(1) == 5.0);
    }
}

TEST_CASE("relu_derivative", "[activations][relu]") {
    SECTION("Positive, negative and zero values") {
        nrt::Tensor x({3});
        x(0) = 2.0;
        x(1) = -2.0;
        x(2) = 0.0;

        nrt::Tensor d = nrt::relu_derivative(x);
        REQUIRE(d(0) == 1.0);
        REQUIRE(d(1) == 0.0);
        // Convention used here: Derivative at 0 is 0 (x <= 0 -> 0)
        REQUIRE(d(2) == 0.0);
    }
}

TEST_CASE("sigmoid", "[activations][sigmoid]") {
    SECTION("sigmoid(0) == 0.5") {
        nrt::Tensor x({1});
        x(0) = 0.0;
        nrt::Tensor y = nrt::sigmoid(x);
        REQUIRE(y(0) == 0.5);
    }

    SECTION("Output is always between 0 and 1") {
        nrt::Tensor x({5});
        x(0) = -100.0;
        x(1) = -50.0;
        x(2) = 0.0;
        x(3) = 50.0;
        x(4) = 100.0;

        nrt::Tensor y = nrt::sigmoid(x);
        for (size_t i = 0; i < 5; ++i) {
            REQUIRE(y(i) >= 0.0);  // >= due to rounding scenatios to 0.0
            REQUIRE(y(i) <= 1.0);  // <= due to rounding scenraios to 1.0
        }
    }

    SECTION("Known value: sigmoid(2) approx 0.8808") {
        nrt::Tensor x({1});
        x(0) = 2.0;
        nrt::Tensor y = nrt::sigmoid(x);
        REQUIRE(std::abs(y(0) - 0.8807970779778824) < 1e-9);
    }

    SECTION("2D: shape is preserved") {
        nrt::Tensor x({2, 2});
        nrt::Tensor y = nrt::sigmoid(x);
        REQUIRE(y.shape() == std::vector<size_t>{2, 2});
    }
}

TEST_CASE("sigmoid_derivative", "[activations][sigmoid]") {
    SECTION("At x=0: sigmoid(0)*(1-sigmoid(0)) = 0.25") {
        nrt::Tensor x({1});
        x(0) = 0.0;
        nrt::Tensor d = nrt::sigmoid_derivative(x);
        REQUIRE(std::abs(d(0) - 0.25) < 1e-12);
    }

    SECTION("Consistent with sigmoid(x) * (1 - sigmoid(x))") {
        nrt::Tensor x({5});
        x(0) = -50.5;
        x(1) = -1.32155;
        x(2) = 0.0;
        x(3) = 3.123456;
        x(4) = 72.4;

        nrt::Tensor s = nrt::sigmoid(x);
        nrt::Tensor d = nrt::sigmoid_derivative(x);

        for (size_t i = 0; i < 3; ++i) {
            double expected = s(i) * (1.0 - s(i));
            REQUIRE(std::abs(d(i) - expected) < 1e-12);
        }
    }
}

TEST_CASE("ReLU Autodiff - Forward pass") {
    auto x = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{2, 2});
    (*x)(0, 0) = -1.0;
    (*x)(0, 1) = 2.0;
    (*x)(1, 0) = -3.0;
    (*x)(1, 1) = 4.0;

    nrt::ReLU relu;
    auto y = relu.forward(x);
    nrt::Tensor expected({2, 2});
    expected(0, 0) = 0.0;
    expected(0, 1) = 2.0;
    expected(1, 0) = 0.0;
    expected(1, 1) = 4.0;

    REQUIRE(tensors_approx_equal(*y, expected));
}

TEST_CASE("ReLU Autodiff - Backward pass") {
    auto x = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{2, 2});
    (*x)(0, 0) = -1.0;
    (*x)(0, 1) = 2.0;
    (*x)(1, 0) = -3.0;
    (*x)(1, 1) = 4.0;

    nrt::ReLU relu;
    auto y = relu.forward(x);
    y->backward();

    // grad_x = grad_y * relu_derivative(x)
    // grad_y = [[1,1],[1,1]] (initialized in backward)
    // relu_derivative: 0 where x <= 0, 1 where x > 0
    // Expected grad_x = [[0,1],[0,1]]

    nrt::Tensor expected({2, 2});
    expected(0, 0) = 0.0;  // x <= 0
    expected(0, 1) = 1.0;  // x > 0
    expected(1, 0) = 0.0;  // x <= 0
    expected(1, 1) = 1.0;  // x > 0

    REQUIRE(tensors_approx_equal(x->gradient(), expected));
}

TEST_CASE("Sigmoid Autodiff - Forward pass") {
    auto x = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{2, 2});
    (*x)(0, 0) = 0.0;
    (*x)(0, 1) = 1.0;
    (*x)(1, 0) = -1.0;
    (*x)(1, 1) = 2.0;

    nrt::Sigmoid sigmoid;
    auto y = sigmoid.forward(x);
    nrt::Tensor expected({2, 2});
    // sigmoid(0) ≈ 0.5
    expected(0, 0) = 0.5;

    // sigmoid(1) ≈ 0.731
    expected(0, 1) = 0.731058578630;

    // sigmoid(-1) ≈ 0.268
    expected(1, 0) = 0.268941421369;

    // sigmoid(2) ≈ 0.880
    expected(1, 1) = 0.880797077973;

    REQUIRE(tensors_approx_equal(*y, expected));
}

TEST_CASE("Sigmoid Autodiff - Backward pass") {
    auto x = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{2, 2});
    (*x)(0, 0) = 0.0;
    (*x)(0, 1) = 1.0;
    (*x)(1, 0) = -1.0;
    (*x)(1, 1) = 2.0;

    nrt::Sigmoid sigmoid;
    auto y = sigmoid.forward(x);
    y->backward();

    // grad_x = grad_y * sigmoid_derivative(x)
    // grad_y = [[1,1],[1,1]]
    // sigmoid_derivative(x) = sigmoid(x) * (1 - sigmoid(x))
    nrt::Tensor expected({2, 2});
    expected(0, 0) = 0.25;
    expected(0, 1) = 0.196611933241;
    expected(1, 0) = 0.196611933241;
    expected(1, 1) = 0.104993585403;

    REQUIRE(tensors_approx_equal(x->gradient(), expected));
}

TEST_CASE("softmax", "[activations][softmax]") {
    SECTION("Output sums to 1") {
        nrt::Tensor x({4, 1});
        x(0, 0) = 1.0;
        x(1, 0) = 2.0;
        x(2, 0) = 3.0;
        x(3, 0) = 4.0;

        nrt::Tensor y = nrt::softmax(x);
        REQUIRE(std::abs(y.sum() - 1.0) < 1e-9);
    }

    SECTION("Uniform logits give a uniform distribution") {
        nrt::Tensor x({3, 1});
        x(0, 0) = x(1, 0) = x(2, 0) = 5.0;

        nrt::Tensor y = nrt::softmax(x);
        for (size_t i = 0; i < 3; ++i) {
            REQUIRE(std::abs(y(i, 0) - (1.0 / 3.0)) < 1e-9);
        }
    }

    SECTION("Known value: softmax([0, 1])") {
        nrt::Tensor x({2, 1});
        x(0, 0) = 0.0;
        x(1, 0) = 1.0;

        nrt::Tensor y = nrt::softmax(x);
        double e = std::exp(1.0);
        REQUIRE(std::abs(y(0, 0) - 1.0 / (1.0 + e)) < 1e-9);
        REQUIRE(std::abs(y(1, 0) - e / (1.0 + e)) < 1e-9);
    }

    SECTION("Shift invariance: softmax(x) == softmax(x + c)") {
        nrt::Tensor x({3, 1});
        x(0, 0) = 1.0;
        x(1, 0) = 2.0;
        x(2, 0) = 3.0;

        nrt::Tensor x_shifted({3, 1});
        for (size_t i = 0; i < 3; ++i) x_shifted(i, 0) = x(i, 0) + 1000.0;

        nrt::Tensor y = nrt::softmax(x);
        nrt::Tensor y_shifted = nrt::softmax(x_shifted);
        for (size_t i = 0; i < 3; ++i) {
            REQUIRE(std::abs(y(i, 0) - y_shifted(i, 0)) < 1e-9);
        }
    }

    SECTION("Non-column-vector shape throws") {
        nrt::Tensor x({2, 2});
        REQUIRE_THROWS_AS(nrt::softmax(x), std::invalid_argument);
    }
}

TEST_CASE("Softmax Autodiff - Forward pass") {
    auto x = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{3, 1});
    (*x)(0, 0) = 1.0;
    (*x)(1, 0) = 2.0;
    (*x)(2, 0) = 3.0;

    nrt::Softmax softmax;
    auto y = softmax.forward(x);

    nrt::Tensor expected = nrt::softmax(*x);
    REQUIRE(tensors_approx_equal(*y, expected));
    REQUIRE(!(y->is_leaf()));
    REQUIRE(x->is_leaf());
}

TEST_CASE("Softmax Autodiff - Backward pass with non-uniform seed") {
    auto x = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{3, 1});
    (*x)(0, 0) = 1.0;
    (*x)(1, 0) = 2.0;
    (*x)(2, 0) = 3.0;

    nrt::Softmax softmax;
    auto y = softmax.forward(x);

    // Seed [1,0,0]: dL/dx_i = y_i * (seed_i - dot(seed,y)), dot collapses to y(0,0)
    nrt::Tensor seed({3, 1});
    seed(0, 0) = 1.0;
    seed(1, 0) = 0.0;
    seed(2, 0) = 0.0;
    y->backward_impl(seed);

    nrt::Tensor probs = nrt::softmax(*x);
    double dot = probs(0, 0);
    nrt::Tensor expected({3, 1});
    for (size_t i = 0; i < 3; ++i) {
        double seed_i = (i == 0) ? 1.0 : 0.0;
        expected(i, 0) = probs(i, 0) * (seed(i, 0) - dot);
    }

    REQUIRE(tensors_approx_equal(x->gradient(), expected));
}

TEST_CASE("Softmax Autodiff - Uniform seed gives zero gradient") {
    // sum(softmax(x)) == 1 for any x, so d(sum)/dx must be exactly zero -
    auto x = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{4, 1});
    (*x)(0, 0) = -2.0;
    (*x)(1, 0) = 0.5;
    (*x)(2, 0) = 3.0;
    (*x)(3, 0) = 1.0;

    nrt::Softmax softmax;
    auto y = softmax.forward(x);
    y->backward();  // auto-seeds all ones

    for (size_t i = 0; i < 4; ++i) {
        REQUIRE(std::abs(x->gradient()(i, 0)) < 1e-9);
    }
}
