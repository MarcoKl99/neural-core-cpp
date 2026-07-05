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
}

TEST_CASE("Tensor construction with invalid shapes throws", "[tensor][construction][errors]") {
    SECTION("rank 0 (empty shape) throws") {
        REQUIRE_THROWS_AS(nrt::Tensor({}), std::invalid_argument);
    }

    SECTION("rank 3 throws") {
        REQUIRE_THROWS_AS(nrt::Tensor({2, 3, 4}), std::invalid_argument);
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

    SECTION("1-argument access on 2D tensor throws") {
        REQUIRE_THROWS_AS(t(0), std::invalid_argument);
    }
}
