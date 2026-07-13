#include <catch2/catch_test_macros.hpp>

#include "nrt/tensor.hpp"

TEST_CASE("Tensor construction with valid shapes", "[tensor][construction]") {
    SECTION("1D shape initializes correctly") {
        nrt::Tensor t({5});
        REQUIRE(t.rank() == 1);
        REQUIRE(t.shape() == std::vector<size_t>{5});
        REQUIRE(t.size() == 5);
    }

    SECTION("2D shape initializes correctly") {
        nrt::Tensor t({2, 3});
        REQUIRE(t.rank() == 2);
        REQUIRE(t.shape() == std::vector<size_t>{2, 3});
        REQUIRE(t.size() == 6);
    }

    SECTION("all elements are zero-initialized") {
        nrt::Tensor t({2, 3});
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                REQUIRE(t(i, j) == 0.0);
            }
        }
    }

    SECTION("Multi-dimensional construction") {
        nrt::Tensor t({3, 4, 3});
        REQUIRE(t.rank() == 3);
        REQUIRE(t.shape() == std::vector<size_t>{3, 4, 3});
        REQUIRE(t.size() == 36);
    }
}

TEST_CASE("Tensor construction with invalid shapes throws", "[tensor][construction][errors]") {
    SECTION("rank 0 (empty shape) throws") {
        REQUIRE_THROWS_AS(nrt::Tensor({}), std::invalid_argument);
    }

    SECTION("zero dimension throws") {
        REQUIRE_THROWS_AS(nrt::Tensor({0, 3}), std::invalid_argument);
    }
}

TEST_CASE("Tensor 1D element access", "[tensor][access]") {
    nrt::Tensor t({3});

    SECTION("write then read returns the same value") {
        t(1) = 42.0;
        REQUIRE(t(1) == 42.0);
    }

    SECTION("out of range index throws") {
        REQUIRE_THROWS_AS(t(3), std::out_of_range);
    }

    SECTION("out of range index throws (const version)") {
        const nrt::Tensor t_const({2});
        REQUIRE_THROWS_AS(t_const(10), std::out_of_range);
    }

    SECTION("2-argument access on 1D tensor throws") {
        REQUIRE_THROWS_AS(t(0, 0), std::invalid_argument);
    }
}

TEST_CASE("Tensor 2D element access", "[tensor][access]") {
    nrt::Tensor t({2, 3});

    SECTION("write then read returns the same value") {
        t(1, 2) = 7.5;
        REQUIRE(t(1, 2) == 7.5);
    }

    SECTION("out of range row throws") {
        REQUIRE_THROWS_AS(t(2, 0), std::out_of_range);
    }

    SECTION("out of range column throws") {
        REQUIRE_THROWS_AS(t(0, 3), std::out_of_range);
    }

    SECTION("out of range row throws (const version)") {
        const nrt::Tensor t_const({2, 3});
        REQUIRE_THROWS_AS(t_const(2, 0), std::out_of_range);
    }

    SECTION("out of range column throws (const version)") {
        const nrt::Tensor t_const({2, 3});
        REQUIRE_THROWS_AS(t_const(0, 3), std::out_of_range);
    }

    SECTION("1-argument access on 2D tensor throws") {
        REQUIRE_THROWS_AS(t(0), std::invalid_argument);
    }

    SECTION("1-argument access on 2D tensor throws (const version)") {
        const nrt::Tensor t_const({2, 3});
        REQUIRE_THROWS_AS(t_const(0), std::invalid_argument);
    }

    SECTION("1-argument access on 2D tensor throws (const version)") {
        const nrt::Tensor t_const({2, 3});
        REQUIRE_THROWS_AS(t_const(0), std::invalid_argument);
    }
}

TEST_CASE("Tensor rank-3 element access bounds", "[tensor][access]") {
    nrt::Tensor t({2, 3, 4});

    SECTION("out of range on first dimension throws") {
        REQUIRE_THROWS_AS(t(2, 0, 0), std::out_of_range);
    }

    SECTION("out of range on second dimension throws") {
        REQUIRE_THROWS_AS(t(0, 3, 0), std::out_of_range);
    }

    SECTION("out of range on third dimension throws") {
        REQUIRE_THROWS_AS(t(0, 0, 4), std::out_of_range);
    }

    SECTION("out of range throws (const version)") {
        const nrt::Tensor t_const({2, 3, 4});
        REQUIRE_THROWS_AS(t_const(2, 0, 0), std::out_of_range);
    }

    SECTION("too few indices throws") {
        REQUIRE_THROWS_AS(t(0, 0), std::invalid_argument);
    }

    SECTION("too many indices throws") {
        REQUIRE_THROWS_AS(t(0, 0, 0, 0), std::invalid_argument);
    }

    SECTION("too few indices throws (const version)") {
        const nrt::Tensor t_const({2, 3, 4});
        REQUIRE_THROWS_AS(t_const(0, 0), std::invalid_argument);
    }
}

TEST_CASE("Multi-dimensional tensor access", "[tensor][access]") {
    SECTION("6D tensor access") {
        auto t_multi = nrt::Tensor({2, 3, 4, 2, 3, 4});
        t_multi(1, 2, 1, 0, 0, 0) = 1.0;

        REQUIRE(t_multi(1, 2, 1, 0, 0, 0) == 1.0);
    }

    SECTION("Distinct values at every position map to correct, non-colliding offsets") {
        nrt::Tensor t({2, 3, 4});

        // Phase 1: write a value to every valid index. Nothing is checked yet.
        int counter = 0;
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                for (size_t k = 0; k < 4; ++k) {
                    t(i, j, k) = static_cast<double>(counter++);
                }
            }
        }

        // Phase 2: re-read every index and check it still holds the exact value
        // that was written to THAT tuple.
        counter = 0;
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                for (size_t k = 0; k < 4; ++k) {
                    REQUIRE(t(i, j, k) == static_cast<double>(counter++));
                }
            }
        }
    }
}

// Mathematical Operations

TEST_CASE("Tensor elementwise operations on rank-3 tensors", "[tensor][elementwise]") {
    nrt::Tensor t1({2, 2, 2});
    nrt::Tensor t2({2, 2, 2});

    // Fill values with a running counter
    int counter1 = 1;
    int counter2 = 10;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            for (size_t k = 0; k < 2; ++k) {
                t1(i, j, k) = static_cast<double>(counter1++);
                t2(i, j, k) = static_cast<double>(counter2++);
            }
        }
    }
    // t1 ranges 1..8 in row-major order, t2 ranges 10..17

    SECTION("operator+ adds element-wise") {
        nrt::Tensor t3 = t1 + t2;
        REQUIRE(t3(0, 0, 0) == 11.0);  // 1 + 10
        REQUIRE(t3(1, 1, 1) == 25.0);  // 8 + 17
    }

    SECTION("operator+= adds in place") {
        t1 += t2;
        REQUIRE(t1(0, 0, 0) == 11.0);
        REQUIRE(t1(1, 1, 1) == 25.0);
    }

    SECTION("operator- subtracts element-wise") {
        nrt::Tensor t3 = t2 - t1;
        REQUIRE(t3(0, 0, 0) == 9.0);
        REQUIRE(t3(1, 1, 1) == 9.0);
    }

    SECTION("operator-= subtracts in place") {
        t1 -= t2;
        REQUIRE(t1(0, 0, 0) == -9.0);
        REQUIRE(t1(1, 1, 1) == -9.0);
    }

    SECTION("hadamard multiplies element-wise") {
        nrt::Tensor t3 = t1.hadamard(t2);
        REQUIRE(t3(0, 0, 0) == 10.0);   // 1 * 10
        REQUIRE(t3(1, 1, 1) == 136.0);  // 8 * 17
    }

    SECTION("scalar multiplication scales every element") {
        nrt::Tensor result = t1 * 2.0;
        REQUIRE(result(0, 0, 0) == 2.0);
        REQUIRE(result(1, 1, 1) == 16.0);
    }

    SECTION("shape mismatch (same rank, different shape) throws") {
        nrt::Tensor t_other_shape({2, 2, 3});
        REQUIRE_THROWS_AS(t1 + t_other_shape, std::invalid_argument);
        REQUIRE_THROWS_AS(t1 - t_other_shape, std::invalid_argument);
        REQUIRE_THROWS_AS(t1.hadamard(t_other_shape), std::invalid_argument);
    }

    SECTION("shape mismatch (different rank) throws") {
        nrt::Tensor t_other_rank({2, 2});
        REQUIRE_THROWS_AS(t1 + t_other_rank, std::invalid_argument);
        REQUIRE_THROWS_AS(t1 - t_other_rank, std::invalid_argument);
        REQUIRE_THROWS_AS(t1.hadamard(t_other_rank), std::invalid_argument);
    }
}

// Addition
TEST_CASE("Tensor 2D addition", "[tensor][addition]") {
    nrt::Tensor t1({2, 2});
    nrt::Tensor t2({2, 2});

    // Set some values
    t1(0, 0) = 1.0;
    t1(0, 1) = 2.0;
    t1(1, 0) = 3.0;
    t1(1, 1) = 4.0;

    t2(0, 0) = 5.0;
    t2(0, 1) = 6.0;
    t2(1, 0) = 7.0;
    t2(1, 1) = 8.0;

    SECTION("Correct sum element-wise + operator") {
        nrt::Tensor t3 = t1 + t2;

        REQUIRE(t3(0, 0) == 6.0);
        REQUIRE(t3(0, 1) == 8.0);
        REQUIRE(t3(1, 0) == 10.0);
        REQUIRE(t3(1, 1) == 12.0);
    }

    SECTION("Correct sum element-wise += operator") {
        t1 += t2;

        REQUIRE(t1(0, 0) == 6.0);
        REQUIRE(t1(0, 1) == 8.0);
        REQUIRE(t1(1, 0) == 10.0);
        REQUIRE(t1(1, 1) == 12.0);
    }

    SECTION("Tensors must be same shape") {
        nrt::Tensor t_other_shape({2, 3});
        REQUIRE_THROWS_AS(t1 + t_other_shape, std::invalid_argument);

        nrt::Tensor t_other_rank({2});
        REQUIRE_THROWS_AS(t1 + t_other_rank, std::invalid_argument);
    }
}

TEST_CASE("Tensor 1D addition", "[tensor][addition]") {
    nrt::Tensor t1({3});
    nrt::Tensor t2({3});

    // Set some values
    t1(0) = 1.0;
    t1(1) = 2.0;
    t1(2) = 3.0;

    t2(0) = 4.0;
    t2(1) = 5.0;
    t2(2) = 6.0;

    SECTION("Correct sum element-wise + operator") {
        nrt::Tensor t3 = t1 + t2;

        REQUIRE(t3(0) == 5.0);
        REQUIRE(t3(1) == 7.0);
        REQUIRE(t3(2) == 9.0);
    }

    SECTION("Correct sum element-wise += operator") {
        t1 += t2;

        REQUIRE(t1(0) == 5.0);
        REQUIRE(t1(1) == 7.0);
        REQUIRE(t1(2) == 9.0);
    }

    SECTION("Tensors must be same shape") {
        nrt::Tensor t_other_shape({4});
        REQUIRE_THROWS_AS(t1 + t_other_shape, std::invalid_argument);

        nrt::Tensor t_other_rank({3, 3});
        REQUIRE_THROWS_AS(t1 + t_other_rank, std::invalid_argument);
    }
}

// Addition
TEST_CASE("Tensor 2D subtraction", "[tensor][addition]") {
    nrt::Tensor t1({2, 2});
    nrt::Tensor t2({2, 2});

    // Set some values
    t1(0, 0) = 1.0;
    t1(0, 1) = 2.0;
    t1(1, 0) = 3.0;
    t1(1, 1) = 4.0;

    t2(0, 0) = 5.0;
    t2(0, 1) = 6.0;
    t2(1, 0) = 7.0;
    t2(1, 1) = 8.0;

    SECTION("Correct sum element-wise - operator") {
        nrt::Tensor t3 = t2 - t1;

        REQUIRE(t3(0, 0) == 4.0);
        REQUIRE(t3(0, 1) == 4.0);
        REQUIRE(t3(1, 0) == 4.0);
        REQUIRE(t3(1, 1) == 4.0);
    }

    SECTION("Correct sum element-wise -= operator") {
        t2 -= t1;

        REQUIRE(t2(0, 0) == 4.0);
        REQUIRE(t2(0, 1) == 4.0);
        REQUIRE(t2(1, 0) == 4.0);
        REQUIRE(t2(1, 1) == 4.0);
    }

    SECTION("Tensors must be same shape") {
        nrt::Tensor t_other_shape({2, 3});
        REQUIRE_THROWS_AS(t1 - t_other_shape, std::invalid_argument);

        nrt::Tensor t_other_rank({2});
        REQUIRE_THROWS_AS(t1 - t_other_rank, std::invalid_argument);
    }
}

TEST_CASE("Tensor 1D subtraction", "[tensor][addition]") {
    nrt::Tensor t1({3});
    nrt::Tensor t2({3});

    // Set some values
    t1(0) = 1.0;
    t1(1) = 2.0;
    t1(2) = 3.0;

    t2(0) = 4.0;
    t2(1) = 5.0;
    t2(2) = 6.0;

    SECTION("Correct sum element-wise - operator") {
        nrt::Tensor t3 = t2 - t1;

        REQUIRE(t3(0) == 3.0);
        REQUIRE(t3(1) == 3.0);
        REQUIRE(t3(2) == 3.0);
    }

    SECTION("Correct sum element-wise -= operator") {
        t2 -= t1;

        REQUIRE(t2(0) == 3.0);
        REQUIRE(t2(1) == 3.0);
        REQUIRE(t2(2) == 3.0);
    }

    SECTION("Tensors must be same shape") {
        nrt::Tensor t_other_shape({4});
        REQUIRE_THROWS_AS(t1 - t_other_shape, std::invalid_argument);

        nrt::Tensor t_other_rank({3, 3});
        REQUIRE_THROWS_AS(t1 - t_other_rank, std::invalid_argument);
    }
}

// Scalar multiplication
TEST_CASE("Tensor scalar multiplication on 2D", "[tensor][scalar_mul]") {
    nrt::Tensor t({2, 2});
    t(0, 0) = 1.0;
    t(0, 1) = -2.0;
    t(1, 0) = 3.0;
    t(1, 1) = 4.0;

    SECTION("Correct result with operator*") {
        nrt::Tensor result = t * 2.0;
        REQUIRE(result(0, 0) == 2.0);
        REQUIRE(result(0, 1) == -4.0);
        REQUIRE(result(1, 0) == 6.0);
        REQUIRE(result(1, 1) == 8.0);
    }

    SECTION("Correct result with operator*=") {
        t *= 2.0;
        REQUIRE(t(0, 0) == 2.0);
        REQUIRE(t(0, 1) == -4.0);
        REQUIRE(t(1, 0) == 6.0);
        REQUIRE(t(1, 1) == 8.0);
    }

    SECTION("Original tensor remains unchanged with operator*") {
        // Note that the outer scope gets executed before the section here again -> t unchanged
        nrt::Tensor result = t * 3.0;
        REQUIRE(t(0, 0) == 1.0);
        REQUIRE(t(1, 1) == 4.0);
    }

    SECTION("Multiplication by zero sets all elements to zero") {
        nrt::Tensor result = t * 0.0;
        REQUIRE(result(0, 0) == 0.0);
        REQUIRE(result(0, 1) == 0.0);
        REQUIRE(result(1, 0) == 0.0);
        REQUIRE(result(1, 1) == 0.0);
    }

    SECTION("Multiplication by negative scalar flips sign") {
        nrt::Tensor result = t * -1.0;
        REQUIRE(result(0, 0) == -1.0);
        REQUIRE(result(0, 1) == 2.0);
        REQUIRE(result(1, 0) == -3.0);
        REQUIRE(result(1, 1) == -4.0);
    }

    SECTION("Multiplication by one is identity") {
        nrt::Tensor result = t * 1.0;
        REQUIRE(result(0, 0) == 1.0);
        REQUIRE(result(0, 1) == -2.0);
        REQUIRE(result(1, 0) == 3.0);
        REQUIRE(result(1, 1) == 4.0);
    }
}

TEST_CASE("Tensor scalar multiplication on 1D", "[tensor][scalar_mul]") {
    nrt::Tensor t({3});
    t(0) = 1.0;
    t(1) = 2.0;
    t(2) = 3.0;

    SECTION("Correct result") {
        nrt::Tensor result = t * 2.0;
        REQUIRE(result(0) == 2.0);
        REQUIRE(result(1) == 4.0);
        REQUIRE(result(2) == 6.0);
    }
}

// Hadamard Product
TEST_CASE("Tensor 2D hadamard product", "[tensor][hadamard]") {
    nrt::Tensor t1({2, 2});
    nrt::Tensor t2({2, 2});
    t1(0, 0) = 1.0;
    t1(0, 1) = 2.0;
    t1(1, 0) = 3.0;
    t1(1, 1) = 4.0;

    t2(0, 0) = 5.0;
    t2(0, 1) = 6.0;
    t2(1, 0) = 7.0;
    t2(1, 1) = 8.0;

    SECTION("Correct elementwise product") {
        nrt::Tensor t3 = t1.hadamard(t2);
        REQUIRE(t3(0, 0) == 5.0);
        REQUIRE(t3(0, 1) == 12.0);
        REQUIRE(t3(1, 0) == 21.0);
        REQUIRE(t3(1, 1) == 32.0);
    }

    SECTION("Original tensors remain unchanged") {
        nrt::Tensor t3 = t1.hadamard(t2);
        REQUIRE(t1(0, 0) == 1.0);
        REQUIRE(t2(0, 0) == 5.0);
    }

    SECTION("Tensors must be same shape") {
        nrt::Tensor t_other_shape({2, 3});
        REQUIRE_THROWS_AS(t1.hadamard(t_other_shape), std::invalid_argument);
    }

    SECTION("Tensors must be same rank") {
        nrt::Tensor t_other_rank({2});
        REQUIRE_THROWS_AS(t1.hadamard(t_other_rank), std::invalid_argument);
    }
}

TEST_CASE("Tensor 1D hadamard product", "[tensor][hadamard]") {
    nrt::Tensor t1({3});
    nrt::Tensor t2({3});
    t1(0) = 1.0;
    t1(1) = 2.0;
    t1(2) = 3.0;

    t2(0) = 4.0;
    t2(1) = 5.0;
    t2(2) = 6.0;

    SECTION("Correct elementwise product") {
        nrt::Tensor t3 = t1.hadamard(t2);
        REQUIRE(t3(0) == 4.0);
        REQUIRE(t3(1) == 10.0);
        REQUIRE(t3(2) == 18.0);
    }

    SECTION("Tensors must be same shape") {
        nrt::Tensor t_other_shape({4});
        REQUIRE_THROWS_AS(t1.hadamard(t_other_shape), std::invalid_argument);
    }

    SECTION("Tensors must be same rank") {
        nrt::Tensor t_other_rank({3, 3});
        REQUIRE_THROWS_AS(t1.hadamard(t_other_rank), std::invalid_argument);
    }
}

// Matrix multiplication
TEST_CASE("Tensor matmul with square matrices", "[tensor][matmul]") {
    // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
    nrt::Tensor a({2, 2});
    a(0, 0) = 1.0;
    a(0, 1) = 2.0;
    a(1, 0) = 3.0;
    a(1, 1) = 4.0;

    nrt::Tensor b({2, 2});
    b(0, 0) = 5.0;
    b(0, 1) = 6.0;
    b(1, 0) = 7.0;
    b(1, 1) = 8.0;

    SECTION("Correct result") {
        // Expected: [[19, 22], [43, 50]]
        nrt::Tensor c = a.matmul(b);
        REQUIRE(c.shape() == std::vector<size_t>{2, 2});
        REQUIRE(c(0, 0) == 19.0);
        REQUIRE(c(0, 1) == 22.0);
        REQUIRE(c(1, 0) == 43.0);
        REQUIRE(c(1, 1) == 50.0);
    }

    SECTION("Original tensors remain unchanged") {
        nrt::Tensor c = a.matmul(b);
        REQUIRE(a(0, 0) == 1.0);
        REQUIRE(b(0, 0) == 5.0);
    }
}

TEST_CASE("Tensor matmul with non-square matrices", "[tensor][matmul]") {
    // A is 2x3, B is 3x2 -> result must be 2x2
    // A = [[1, 2, 3], [4, 5, 6]]
    nrt::Tensor a({2, 3});
    a(0, 0) = 1.0;
    a(0, 1) = 2.0;
    a(0, 2) = 3.0;
    a(1, 0) = 4.0;
    a(1, 1) = 5.0;
    a(1, 2) = 6.0;

    // B = [[7, 8], [9, 10], [11, 12]]
    nrt::Tensor b({3, 2});
    b(0, 0) = 7.0;
    b(0, 1) = 8.0;
    b(1, 0) = 9.0;
    b(1, 1) = 10.0;
    b(2, 0) = 11.0;
    b(2, 1) = 12.0;

    SECTION("Result has correct shape (outer dimensions)") {
        nrt::Tensor c = a.matmul(b);
        REQUIRE(c.shape() == std::vector<size_t>{2, 2});
    }

    SECTION("Correct values") {
        // Expected:
        // c(0,0) = 1*7 + 2*9  + 3*11 = 58
        // c(0,1) = 1*8 + 2*10 + 3*12 = 64
        // c(1,0) = 4*7 + 5*9  + 6*11 = 139
        // c(1,1) = 4*8 + 5*10 + 6*12 = 154
        nrt::Tensor c = a.matmul(b);
        REQUIRE(c(0, 0) == 58.0);
        REQUIRE(c(0, 1) == 64.0);
        REQUIRE(c(1, 0) == 139.0);
        REQUIRE(c(1, 1) == 154.0);
    }
}

TEST_CASE("Tensor matmul shape validation", "[tensor][matmul][errors]") {
    nrt::Tensor a({2, 3});
    nrt::Tensor b({3, 2});

    SECTION("Mismatched inner dimensions throw") {
        nrt::Tensor wrong_inner({4, 2});
        REQUIRE_THROWS_AS(a.matmul(wrong_inner), std::invalid_argument);
    }

    SECTION("Rank-1 left operand throws") {
        nrt::Tensor rank1({3});
        REQUIRE_THROWS_AS(rank1.matmul(b), std::invalid_argument);
    }

    SECTION("Rank-1 right operand throws") {
        nrt::Tensor rank1({3});
        REQUIRE_THROWS_AS(a.matmul(rank1), std::invalid_argument);
    }

    SECTION("Rank-3 left operand throws") {
        nrt::Tensor rank3({2, 3, 4});
        REQUIRE_THROWS_AS(rank3.matmul(b), std::invalid_argument);
    }

    SECTION("Rank-3 right operand throws") {
        nrt::Tensor rank3({2, 3, 4});
        REQUIRE_THROWS_AS(a.matmul(rank3), std::invalid_argument);
    }

    SECTION("Rank-3 on both tensors throws") {
        nrt::Tensor rank3_1({2, 2, 2});
        nrt::Tensor rank3_2({2, 2, 2});
        REQUIRE_THROWS_AS(rank3_1.matmul(rank3_2), std::invalid_argument);
    }
}

// Transpose
TEST_CASE("Tensor transpose", "[tensor][transpose]") {
    SECTION("Square matrix") {
        // A = [[1, 2], [3, 4]] -> A^T = [[1, 3], [2, 4]]
        nrt::Tensor a({2, 2});
        a(0, 0) = 1.0;
        a(0, 1) = 2.0;
        a(1, 0) = 3.0;
        a(1, 1) = 4.0;

        nrt::Tensor t = a.transpose();
        REQUIRE(t.shape() == std::vector<size_t>{2, 2});
        REQUIRE(t(0, 0) == 1.0);
        REQUIRE(t(0, 1) == 3.0);
        REQUIRE(t(1, 0) == 2.0);
        REQUIRE(t(1, 1) == 4.0);
    }

    SECTION("Non-square matrix: shape is swapped") {
        // A is 2x3 -> A^T is 3x2
        nrt::Tensor a({2, 3});
        a(0, 0) = 1.0;
        a(0, 1) = 2.0;
        a(0, 2) = 3.0;
        a(1, 0) = 4.0;
        a(1, 1) = 5.0;
        a(1, 2) = 6.0;

        nrt::Tensor t = a.transpose();
        REQUIRE(t.shape() == std::vector<size_t>{3, 2});
        REQUIRE(t(0, 0) == 1.0);
        REQUIRE(t(1, 0) == 2.0);
        REQUIRE(t(2, 0) == 3.0);
        REQUIRE(t(0, 1) == 4.0);
        REQUIRE(t(1, 1) == 5.0);
        REQUIRE(t(2, 1) == 6.0);
    }

    SECTION("Double transpose returns the original matrix") {
        nrt::Tensor a({2, 3});
        a(0, 0) = 1.0;
        a(0, 1) = 2.0;
        a(0, 2) = 3.0;
        a(1, 0) = 4.0;
        a(1, 1) = 5.0;
        a(1, 2) = 6.0;

        nrt::Tensor t = a.transpose().transpose();
        REQUIRE(t.shape() == a.shape());
        for (size_t i = 0; i < a.shape()[0]; ++i) {
            for (size_t j = 0; j < a.shape()[1]; ++j) {
                REQUIRE(t(i, j) == a(i, j));
            }
        }
    }

    SECTION("Original tensor remains unchanged") {
        nrt::Tensor a({2, 2});
        a(0, 0) = 1.0;
        a(0, 1) = 2.0;
        a(1, 0) = 3.0;
        a(1, 1) = 4.0;

        nrt::Tensor t = a.transpose();
        REQUIRE(a(0, 1) == 2.0);
        REQUIRE(a(1, 0) == 3.0);
    }

    SECTION("Rank-1 tensor throws") {
        nrt::Tensor v({3});
        REQUIRE_THROWS_AS(v.transpose(), std::invalid_argument);
    }

    SECTION("Rank-3 tensor throws") {
        nrt::Tensor t3({2, 3, 4});
        REQUIRE_THROWS_AS(t3.transpose(), std::invalid_argument);
    }
}

TEST_CASE("Tensor reshape", "[tensor][reshape]") {
    nrt::Tensor t({2, 3});
    t(0, 0) = 1.0;
    t(0, 1) = 2.0;
    t(0, 2) = 3.0;
    t(1, 0) = 4.0;
    t(1, 1) = 5.0;
    t(1, 2) = 6.0;

    SECTION("reshape to a different 2D shape preserves row-major order") {
        nrt::Tensor r = t.reshape({3, 2});
        REQUIRE(r.shape() == std::vector<size_t>{3, 2});
        REQUIRE(r(0, 0) == 1.0);
        REQUIRE(r(0, 1) == 2.0);
        REQUIRE(r(1, 0) == 3.0);
        REQUIRE(r(1, 1) == 4.0);
        REQUIRE(r(2, 0) == 5.0);
        REQUIRE(r(2, 1) == 6.0);
    }

    SECTION("reshape to 1D flattens in row-major order") {
        nrt::Tensor r = t.reshape({6});
        REQUIRE(r.rank() == 1);
        for (size_t i = 0; i < 6; ++i) {
            REQUIRE(r(i) == static_cast<double>(i + 1));
        }
    }

    SECTION("reshape to rank-3 shape") {
        nrt::Tensor r = t.reshape({2, 3, 1});
        REQUIRE(r.shape() == std::vector<size_t>{2, 3, 1});
        REQUIRE(r(0, 0, 0) == 1.0);
        REQUIRE(r(1, 2, 0) == 6.0);
    }

    SECTION("reshape to the same shape leaves values unchanged") {
        nrt::Tensor r = t.reshape({2, 3});
        REQUIRE(r.shape() == t.shape());
        REQUIRE(r(0, 0) == 1.0);
        REQUIRE(r(1, 2) == 6.0);
    }

    SECTION("mismatched element count throws") {
        REQUIRE_THROWS_AS(t.reshape({4, 2}), std::invalid_argument);
    }

    SECTION("result is an independent copy, not a view") {
        nrt::Tensor r = t.reshape({3, 2});

        r(0, 0) = 99.0;
        REQUIRE(t(0, 0) == 1.0);  // mutating r must not affect t

        t(0, 1) = -1.0;
        REQUIRE(r(0, 1) == 2.0);  // mutating t after the fact must not affect r
    }
}
