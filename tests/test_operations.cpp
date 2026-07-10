#include <catch2/catch_test_macros.hpp>

#include "nrt/operations.hpp"
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

TEST_CASE("MatMul Autodiff - Simple forward pass") {
    auto a = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{2, 2});
    (*a)(0, 0) = 1.0;
    (*a)(0, 1) = 2.0;
    (*a)(1, 0) = 3.0;
    (*a)(1, 1) = 4.0;

    auto b = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{2, 2});
    (*b)(0, 0) = 1.0;
    (*b)(0, 1) = 1.0;
    (*b)(1, 0) = 1.0;
    (*b)(1, 1) = 1.0;

    auto c = nrt::matmul_autodiff(a, b);

    // Define the expected result
    nrt::Tensor expected({2, 2});
    expected(0, 0) = 3.0;
    expected(0, 1) = 3.0;
    expected(1, 0) = 7.0;
    expected(1, 1) = 7.0;

    REQUIRE(tensors_approx_equal(*c, expected));
}

TEST_CASE("MatMul Autodiff - Backward pass gradients") {
    auto a = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{2, 2});
    (*a)(0, 0) = 1.0;
    (*a)(0, 1) = 2.0;
    (*a)(1, 0) = 3.0;
    (*a)(1, 1) = 4.0;

    auto b = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{2, 2});
    (*b)(0, 0) = 1.0;
    (*b)(0, 1) = 1.0;
    (*b)(1, 0) = 1.0;
    (*b)(1, 1) = 1.0;

    auto c = nrt::matmul_autodiff(a, b);
    c->backward();

    // Expected gradients: grad_a = grad_c @ b.T, grad_b = a.T @ grad_c
    // grad_c = [[1, 1], [1, 1]] (initialized in backward)
    // b.T = [[1, 1], [1, 1]]
    // grad_a = [[1, 1], [1, 1]] @ [[1, 1], [1, 1]] = [[2, 2], [2, 2]]

    nrt::Tensor expected_grad_a({2, 2});
    expected_grad_a(0, 0) = 2.0;
    expected_grad_a(0, 1) = 2.0;
    expected_grad_a(1, 0) = 2.0;
    expected_grad_a(1, 1) = 2.0;

    REQUIRE(tensors_approx_equal(a->gradient(), expected_grad_a));

    // a.T = [[1, 3], [2, 4]]
    // grad_b = [[1, 3], [2, 4]] @ [[1, 1], [1, 1]] = [[4, 4], [6, 6]]

    nrt::Tensor expected_grad_b({2, 2});
    expected_grad_b(0, 0) = 4.0;
    expected_grad_b(0, 1) = 4.0;
    expected_grad_b(1, 0) = 6.0;
    expected_grad_b(1, 1) = 6.0;

    REQUIRE(tensors_approx_equal(b->gradient(), expected_grad_b));
}

TEST_CASE("MatMul Autodiff - Leaf tensor identification") {
    auto a = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{2, 2});
    (*a)(0, 0) = 1.0;
    (*a)(0, 1) = 2.0;
    (*a)(1, 0) = 3.0;
    (*a)(1, 1) = 4.0;

    auto b = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{2, 2});
    (*b)(0, 0) = 1.0;
    (*b)(0, 1) = 1.0;
    (*b)(1, 0) = 1.0;
    (*b)(1, 1) = 1.0;

    // Tensors a and b are initially leafs
    REQUIRE(a->is_leaf());
    REQUIRE(b->is_leaf());

    auto c = nrt::matmul_autodiff(a, b);

    // Check all tensors (a and b again, c now as well)
    REQUIRE(a->is_leaf());
    REQUIRE(b->is_leaf());
    REQUIRE(!(c->is_leaf()));
}

TEST_CASE("Scalar Mult Autodiff - Forward pass") {
    auto a = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{2, 2});
    (*a)(0, 0) = 1.0;
    (*a)(0, 1) = 2.0;
    (*a)(1, 0) = 3.0;
    (*a)(1, 1) = 4.0;

    double scalar = 2.0;
    auto b = nrt::scalar_mult_autodiff(a, scalar);

    nrt::Tensor expected({2, 2});
    expected(0, 0) = 2.0;
    expected(0, 1) = 4.0;
    expected(1, 0) = 6.0;
    expected(1, 1) = 8.0;

    REQUIRE(tensors_approx_equal(*b, expected));
}

TEST_CASE("Scalar Mult Autodiff - Backward pass") {
    auto a = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{2, 2});
    (*a)(0, 0) = 1.0;
    (*a)(0, 1) = 2.0;
    (*a)(1, 0) = 3.0;
    (*a)(1, 1) = 4.0;

    double scalar = 2.0;
    auto b = nrt::scalar_mult_autodiff(a, scalar);
    b->backward();

    // grad_a = grad_b * scalar = [[1,1],[1,1]] * 2.0 = [[2,2],[2,2]]
    nrt::Tensor expected({2, 2});
    expected(0, 0) = 2.0;
    expected(0, 1) = 2.0;
    expected(1, 0) = 2.0;
    expected(1, 1) = 2.0;

    REQUIRE(tensors_approx_equal(a->gradient(), expected));
}

TEST_CASE("Add Autodiff - Forward pass") {
    auto a = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{2, 2});
    (*a)(0, 0) = 1.0;
    (*a)(0, 1) = 2.0;
    (*a)(1, 0) = 3.0;
    (*a)(1, 1) = 4.0;

    auto b = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{2, 2});
    (*b)(0, 0) = 5.0;
    (*b)(0, 1) = 6.0;
    (*b)(1, 0) = 7.0;
    (*b)(1, 1) = 8.0;

    auto c = nrt::add_autodiff(a, b);
    nrt::Tensor expected({2, 2});
    expected(0, 0) = 6.0;
    expected(0, 1) = 8.0;
    expected(1, 0) = 10.0;
    expected(1, 1) = 12.0;

    REQUIRE(tensors_approx_equal(*c, expected));
}

TEST_CASE("Add Autodiff - Backward pass") {
    auto a = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{2, 2});
    (*a)(0, 0) = 1.0;
    (*a)(0, 1) = 2.0;
    (*a)(1, 0) = 3.0;
    (*a)(1, 1) = 4.0;

    auto b = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{2, 2});
    (*b)(0, 0) = 5.0;
    (*b)(0, 1) = 6.0;
    (*b)(1, 0) = 7.0;
    (*b)(1, 1) = 8.0;

    auto c = nrt::add_autodiff(a, b);
    c->backward();

    // Both gradients should be [[1,1],[1,1]] (same as grad_c)
    nrt::Tensor expected({2, 2});
    expected(0, 0) = 1.0;
    expected(0, 1) = 1.0;
    expected(1, 0) = 1.0;
    expected(1, 1) = 1.0;

    REQUIRE(tensors_approx_equal(a->gradient(), expected));
    REQUIRE(tensors_approx_equal(b->gradient(), expected));
}

TEST_CASE("Subtract Autodiff - Forward pass") {
    auto a = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{2, 2});
    (*a)(0, 0) = 5.0;
    (*a)(0, 1) = 6.0;
    (*a)(1, 0) = 7.0;
    (*a)(1, 1) = 8.0;

    auto b = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{2, 2});
    (*b)(0, 0) = 1.0;
    (*b)(0, 1) = 2.0;
    (*b)(1, 0) = 3.0;
    (*b)(1, 1) = 4.0;

    auto c = nrt::subtract_autodiff(a, b);
    nrt::Tensor expected({2, 2});
    expected(0, 0) = 4.0;
    expected(0, 1) = 4.0;
    expected(1, 0) = 4.0;
    expected(1, 1) = 4.0;

    REQUIRE(tensors_approx_equal(*c, expected));
}

TEST_CASE("Subtract Autodiff - Backward pass") {
    auto a = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{2, 2});
    (*a)(0, 0) = 5.0;
    (*a)(0, 1) = 6.0;
    (*a)(1, 0) = 7.0;
    (*a)(1, 1) = 8.0;

    auto b = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{2, 2});
    (*b)(0, 0) = 1.0;
    (*b)(0, 1) = 2.0;
    (*b)(1, 0) = 3.0;
    (*b)(1, 1) = 4.0;

    auto c = nrt::subtract_autodiff(a, b);
    c->backward();

    // grad_a should be [[1,1],[1,1]]
    nrt::Tensor expected_a({2, 2});
    expected_a(0, 0) = 1.0;
    expected_a(0, 1) = 1.0;
    expected_a(1, 0) = 1.0;
    expected_a(1, 1) = 1.0;

    // grad_b should be [[-1,-1],[-1,-1]] (negated)
    nrt::Tensor expected_b({2, 2});
    expected_b(0, 0) = -1.0;
    expected_b(0, 1) = -1.0;
    expected_b(1, 0) = -1.0;
    expected_b(1, 1) = -1.0;

    REQUIRE(tensors_approx_equal(a->gradient(), expected_a));
    REQUIRE(tensors_approx_equal(b->gradient(), expected_b));
}

// ==================================================
// Comprehensive tests: chaining multiple operations
// ==================================================

TEST_CASE("Chain: MatMul -> Add -> Backward") {
    // Test gradient flow through: (w @ x) + b
    auto w = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{2, 2});
    (*w)(0, 0) = 1.0;
    (*w)(0, 1) = 2.0;
    (*w)(1, 0) = 3.0;
    (*w)(1, 1) = 4.0;

    auto x = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{2, 1});
    (*x)(0, 0) = 1.0;
    (*x)(1, 0) = 2.0;

    auto b = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{2, 1});
    (*b)(0, 0) = 0.5;
    (*b)(1, 0) = 1.0;

    // Forward: z = w @ x, y = z + b
    auto z = nrt::matmul_autodiff(w, x);
    auto y = nrt::add_autodiff(z, b);

    // Expected forward:
    // z = [[1*1 + 2*2], [3*1 + 4*2]] = [[5], [11]]
    // y = [[5.5], [12]]
    nrt::Tensor expected_y({2, 1});
    expected_y(0, 0) = 5.5;
    expected_y(1, 0) = 12.0;
    REQUIRE(tensors_approx_equal(*y, expected_y));

    // Backward: initialize gradient of y as [[1], [1]]
    y->backward();

    // Expected gradients:
    // grad_y = [[1], [1]]
    // grad_z = grad_y = [[1], [1]] (passed through add)
    // grad_b = grad_y = [[1], [1]] (passed through add)
    // grad_w = grad_z @ x.T = [[1], [1]] @ [[1, 2]] = [[1, 2], [1, 2]]
    // grad_x = w.T @ grad_z = [[1, 3], [2, 4]] @ [[1], [1]] = [[4], [6]]

    nrt::Tensor expected_grad_w({2, 2});
    expected_grad_w(0, 0) = 1.0;
    expected_grad_w(0, 1) = 2.0;
    expected_grad_w(1, 0) = 1.0;
    expected_grad_w(1, 1) = 2.0;

    nrt::Tensor expected_grad_x({2, 1});
    expected_grad_x(0, 0) = 4.0;
    expected_grad_x(1, 0) = 6.0;

    nrt::Tensor expected_grad_b({2, 1});
    expected_grad_b(0, 0) = 1.0;
    expected_grad_b(1, 0) = 1.0;

    REQUIRE(tensors_approx_equal(w->gradient(), expected_grad_w));
    REQUIRE(tensors_approx_equal(x->gradient(), expected_grad_x));
    REQUIRE(tensors_approx_equal(b->gradient(), expected_grad_b));
}

TEST_CASE("Chain: MatMul -> Scalar Mult -> Backward") {
    // Test gradient flow through: (w @ x) * 2
    auto w = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{2, 2});
    (*w)(0, 0) = 1.0;
    (*w)(0, 1) = 2.0;
    (*w)(1, 0) = 3.0;
    (*w)(1, 1) = 4.0;

    auto x = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{2, 1});
    (*x)(0, 0) = 1.0;
    (*x)(1, 0) = 2.0;

    // Forward: z = w @ x, y = z * 2
    auto z = nrt::matmul_autodiff(w, x);
    auto y = nrt::scalar_mult_autodiff(z, 2.0);

    // Expected forward:
    // z = [[5], [11]]
    // y = [[10], [22]]
    nrt::Tensor expected_y({2, 1});
    expected_y(0, 0) = 10.0;
    expected_y(1, 0) = 22.0;
    REQUIRE(tensors_approx_equal(*y, expected_y));

    // Backward:
    y->backward();

    // grad_y = [[1], [1]]
    // grad_z = grad_y * 2 = [[2], [2]]
    // grad_w = grad_z @ x.T = [[2], [2]] @ [[1, 2]] = [[2, 4], [2, 4]]
    // grad_x = w.T @ grad_z = [[1, 3], [2, 4]] @ [[2], [2]] = [[8], [12]]

    nrt::Tensor expected_grad_w({2, 2});
    expected_grad_w(0, 0) = 2.0;
    expected_grad_w(0, 1) = 4.0;
    expected_grad_w(1, 0) = 2.0;
    expected_grad_w(1, 1) = 4.0;

    nrt::Tensor expected_grad_x({2, 1});
    expected_grad_x(0, 0) = 8.0;
    expected_grad_x(1, 0) = 12.0;

    REQUIRE(tensors_approx_equal(w->gradient(), expected_grad_w));
    REQUIRE(tensors_approx_equal(x->gradient(), expected_grad_x));
}

TEST_CASE("Chain: MatMul -> Add -> Subtract -> Backward") {
    // Test 3-operation chain: ((w @ x) + b) - penalty
    // This tests deeper recursion through the computation graph
    auto w = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{2, 2});
    (*w)(0, 0) = 1.0;
    (*w)(0, 1) = 2.0;
    (*w)(1, 0) = 3.0;
    (*w)(1, 1) = 4.0;

    auto x = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{2, 1});
    (*x)(0, 0) = 1.0;
    (*x)(1, 0) = 2.0;

    auto b = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{2, 1});
    (*b)(0, 0) = 0.5;
    (*b)(1, 0) = 1.0;

    auto penalty = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{2, 1});
    (*penalty)(0, 0) = 0.1;
    (*penalty)(1, 0) = 0.2;

    // Forward: z = w @ x, y = z + b, loss = y - penalty
    auto z = nrt::matmul_autodiff(w, x);
    auto y = nrt::add_autodiff(z, b);
    auto loss = nrt::subtract_autodiff(y, penalty);

    // Expected forward:
    // z = [[5], [11]]
    // y = [[5.5], [12]]
    // loss = [[5.4], [11.8]]
    nrt::Tensor expected_loss({2, 1});
    expected_loss(0, 0) = 5.4;
    expected_loss(1, 0) = 11.8;
    REQUIRE(tensors_approx_equal(*loss, expected_loss));

    // Backward:
    loss->backward();

    // grad_loss = [[1], [1]]
    // grad_y = grad_loss = [[1], [1]] (from subtract)
    // grad_penalty = -grad_loss = [[-1], [-1]]
    // grad_z = grad_y = [[1], [1]] (from add)
    // grad_b = grad_y = [[1], [1]] (from add)
    // grad_w = grad_z @ x.T = [[1], [1]] @ [[1, 2]] = [[1, 2], [1, 2]]
    // grad_x = w.T @ grad_z = [[1, 3], [2, 4]] @ [[1], [1]] = [[4], [6]]

    nrt::Tensor expected_grad_w({2, 2});
    expected_grad_w(0, 0) = 1.0;
    expected_grad_w(0, 1) = 2.0;
    expected_grad_w(1, 0) = 1.0;
    expected_grad_w(1, 1) = 2.0;

    nrt::Tensor expected_grad_x({2, 1});
    expected_grad_x(0, 0) = 4.0;
    expected_grad_x(1, 0) = 6.0;

    nrt::Tensor expected_grad_b({2, 1});
    expected_grad_b(0, 0) = 1.0;
    expected_grad_b(1, 0) = 1.0;

    nrt::Tensor expected_grad_penalty({2, 1});
    expected_grad_penalty(0, 0) = -1.0;
    expected_grad_penalty(1, 0) = -1.0;

    REQUIRE(tensors_approx_equal(w->gradient(), expected_grad_w));
    REQUIRE(tensors_approx_equal(x->gradient(), expected_grad_x));
    REQUIRE(tensors_approx_equal(b->gradient(), expected_grad_b));
    REQUIRE(tensors_approx_equal(penalty->gradient(), expected_grad_penalty));
}
