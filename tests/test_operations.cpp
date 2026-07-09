#include <catch2/catch_test_macros.hpp>

#include "nrt/operations.hpp"
#include "nrt/tensor.hpp"

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

TEST_CASE("MatMul Autodiff - Simple forward pass") {
    nrt::Tensor a({2, 2});
    a(0, 0) = 1.0;
    a(0, 1) = 2.0;
    a(1, 0) = 3.0;
    a(1, 1) = 4.0;

    nrt::Tensor b({2, 2});
    b(0, 0) = 1.0;
    b(0, 1) = 1.0;
    b(1, 0) = 1.0;
    b(1, 1) = 1.0;

    nrt::Tensor c = nrt::matmul_autodiff(a, b);

    // Define the expected result
    nrt::Tensor expected({2, 2});
    expected(0, 0) = 3.0;
    expected(0, 1) = 3.0;
    expected(1, 0) = 7.0;
    expected(1, 1) = 7.0;

    REQUIRE(tensors_approx_equal(c, expected));
}

TEST_CASE("MatMul Autodiff - Backward pass gradients") {
    nrt::Tensor a({2, 2});
    a(0, 0) = 1.0;
    a(0, 1) = 2.0;
    a(1, 0) = 3.0;
    a(1, 1) = 4.0;

    nrt::Tensor b({2, 2});
    b(0, 0) = 1.0;
    b(0, 1) = 1.0;
    b(1, 0) = 1.0;
    b(1, 1) = 1.0;

    nrt::Tensor c = nrt::matmul_autodiff(a, b);
    c.backward();

    // Expected gradients: grad_a = grad_c @ b.T, grad_b = a.T @ grad_c
    // grad_c = [[1, 1], [1, 1]] (initialized in backward)
    // b.T = [[1, 1], [1, 1]]
    // grad_a = [[1, 1], [1, 1]] @ [[1, 1], [1, 1]] = [[2, 2], [2, 2]]

    nrt::Tensor expected_grad_a({2, 2});
    expected_grad_a(0, 0) = 2.0;
    expected_grad_a(0, 1) = 2.0;
    expected_grad_a(1, 0) = 2.0;
    expected_grad_a(1, 1) = 2.0;

    REQUIRE(tensors_approx_equal(a.gradient(), expected_grad_a));

    // a.T = [[1, 3], [2, 4]]
    // grad_b = [[1, 3], [2, 4]] @ [[1, 1], [1, 1]] = [[4, 4], [6, 6]]

    nrt::Tensor expected_grad_b({2, 2});
    expected_grad_b(0, 0) = 4.0;
    expected_grad_b(0, 1) = 4.0;
    expected_grad_b(1, 0) = 6.0;
    expected_grad_b(1, 1) = 6.0;

    REQUIRE(tensors_approx_equal(b.gradient(), expected_grad_b));
}

TEST_CASE("MatMul Autodiff - Leaf tensor identification") {
    nrt::Tensor a({2, 2});
    a(0, 0) = 1.0;
    a(0, 1) = 2.0;
    a(1, 0) = 3.0;
    a(1, 1) = 4.0;

    nrt::Tensor b({2, 2});
    b(0, 0) = 1.0;
    b(0, 1) = 1.0;
    b(1, 0) = 1.0;
    b(1, 1) = 1.0;

    // Tensors a and b are initially leafs
    REQUIRE(a.is_leaf() == true);
    REQUIRE(b.is_leaf() == true);

    nrt::Tensor c = nrt::matmul_autodiff(a, b);

    // Check all tensors (a and b again, c now as well)
    REQUIRE(a.is_leaf() == true);
    REQUIRE(b.is_leaf() == true);
    REQUIRE(c.is_leaf() == false);
}
