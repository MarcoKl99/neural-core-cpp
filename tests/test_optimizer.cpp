#include <catch2/catch_test_macros.hpp>
#include <iostream>

#include "nrt/linear.hpp"
#include "nrt/optimizer.hpp"
#include "nrt/tensor.hpp"

// Helper function for floating-point comparison
bool approx_equal(double a, double b, double tolerance = 1e-9) {
    return std::abs(a - b) < tolerance;
}

TEST_CASE("SGD Optimizer - Basic functionality") {
    // Create two simple tensors to optimize
    nrt::Tensor param1({2, 2});
    param1(0, 0) = 1.0;
    param1(0, 1) = 2.0;
    param1(1, 0) = 3.0;
    param1(1, 1) = 4.0;

    nrt::Tensor grad1({2, 2});
    grad1(0, 0) = 0.1;
    grad1(0, 1) = 0.2;
    grad1(1, 0) = 0.3;
    grad1(1, 1) = 0.4;

    // Create parameter and optimizer
    double lr = 0.1;
    std::vector<nrt::Parameter> params = {{&param1}};
    nrt::SGD optimizer(params, lr);

    // Test 1: Verify init
    REQUIRE(optimizer.learning_rate() == 0.1);
    REQUIRE(optimizer.num_params() == 1);

    // Test 2: zero_grad() clears gradients
    param1.accumulate_gradient(grad1);  // gradient is now non-zero
    optimizer.zero_grad();
    REQUIRE(param1.gradient()(0, 0) == 0.0);
    REQUIRE(param1.gradient()(0, 1) == 0.0);
    REQUIRE(param1.gradient()(1, 0) == 0.0);
    REQUIRE(param1.gradient()(1, 1) == 0.0);

    // Test 3: step() updates parameters correctly
    param1.accumulate_gradient(grad1);  // Set known gradient (adds onto the zeros)
    nrt::Tensor param1_before = param1;
    optimizer.step();

    // param -= lr * grad, so: param = param_before - 0.1 * grad
    REQUIRE(approx_equal(param1(0, 0), param1_before(0, 0) - 0.1 * 0.1));
    REQUIRE(approx_equal(param1(0, 1), param1_before(0, 1) - 0.1 * 0.2));
    REQUIRE(approx_equal(param1(1, 0), param1_before(1, 0) - 0.1 * 0.3));
    REQUIRE(approx_equal(param1(1, 1), param1_before(1, 1) - 0.1 * 0.4));
}

TEST_CASE("SGD Optimizer - Multiple parameters") {
    nrt::Tensor w({2, 2});
    w(0, 0) = 1.0;
    w(0, 1) = 2.0;
    w(1, 0) = 3.0;
    w(1, 1) = 4.0;

    nrt::Tensor grad_w({2, 2});
    grad_w(0, 0) = 0.1;
    grad_w(0, 1) = 0.2;
    grad_w(1, 0) = 0.3;
    grad_w(1, 1) = 0.4;

    nrt::Tensor b({2, 1});
    b(0, 0) = 0.5;
    b(1, 0) = 0.6;

    nrt::Tensor grad_b({2, 1});
    grad_b(0, 0) = 0.3;
    grad_b(1, 0) = 0.8;

    double lr = 0.1;
    std::vector<nrt::Parameter> params = {{&w}, {&b}};
    nrt::SGD optimizer(params, lr);

    REQUIRE(optimizer.num_params() == 2);

    // Push gradients into the parameter tensors
    w.accumulate_gradient(grad_w);
    b.accumulate_gradient(grad_b);

    // Store values before step
    nrt::Tensor w_before = w;
    nrt::Tensor b_before = b;

    optimizer.step();

    // Both should be updated
    REQUIRE(approx_equal(w(0, 0), w_before(0, 0) - lr * grad_w(0, 0)));
    REQUIRE(approx_equal(b(0, 0), b_before(0, 0) - lr * grad_b(0, 0)));
}

TEST_CASE("SGD Optimizer - Different learning rates") {
    nrt::Tensor param({1, 1});
    param(0, 0) = 1.0;
    const auto param_value_old = param(0, 0);

    nrt::Tensor grad({1, 1});
    grad(0, 0) = 0.3;

    // Test with lr = 0.1
    {
        const auto lr = 0.1;
        std::vector<nrt::Parameter> params = {{&param}};
        nrt::SGD opt(params, lr);
        param.zero_grad();
        param.accumulate_gradient(grad);
        opt.step();
        REQUIRE(approx_equal(param(0, 0), param_value_old - lr * grad(0, 0)));
    }

    // Reset and test with lr = 0.5
    param(0, 0) = 1.0;
    {
        const auto lr = 0.5;
        std::vector<nrt::Parameter> params = {{&param}};
        nrt::SGD opt(params, lr);
        param.zero_grad();
        param.accumulate_gradient(grad);
        opt.step();
        REQUIRE(approx_equal(param(0, 0), param_value_old - lr * grad(0, 0)));
    }
}

TEST_CASE("SGD Optimizer - Integration with Linear") {
    nrt::Linear layer(2, 3);
    auto input = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{2, 1});
    (*input)(0, 0) = 0.5;
    (*input)(1, 0) = 0.5;

    auto params = layer.parameters();
    const auto lr = 0.01;
    nrt::SGD optimizer(params, lr);

    // Simulate a training step
    optimizer.zero_grad();
    auto output = layer.forward(input);

    // Simulate backward: gradients now live INSIDE the parameter tensors, so we
    // build gradient tensors and accumulate them into the params, instead of
    // writing through a (now-removed) Parameter::gradient pointer.
    nrt::Tensor grad_w({3, 2});
    grad_w(0, 0) = 0.1;
    grad_w(0, 1) = 0.2;
    grad_w(1, 0) = 0.15;
    grad_w(1, 1) = 0.25;
    grad_w(2, 0) = 0.12;
    grad_w(2, 1) = 0.22;

    nrt::Tensor grad_b({3, 1});
    grad_b(0, 0) = 0.05;
    grad_b(1, 0) = 0.04;
    grad_b(2, 0) = 0.03;

    params[0].value->accumulate_gradient(grad_w);
    params[1].value->accumulate_gradient(grad_b);

    // Store weights before
    nrt::Tensor w_before = layer.weights();
    nrt::Tensor b_before = layer.bias();

    optimizer.step();

    // Verify update happened
    REQUIRE(approx_equal(layer.weights()(0, 0), w_before(0, 0) - lr * grad_w(0, 0)));
    REQUIRE(approx_equal(layer.bias()(0, 0), b_before(0, 0) - lr * grad_b(0, 0)));
}
