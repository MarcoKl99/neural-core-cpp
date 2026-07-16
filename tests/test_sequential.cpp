#include <catch2/catch_test_macros.hpp>

#include "nrt/activations.hpp"
#include "nrt/linear.hpp"
#include "nrt/sequential.hpp"
#include "nrt/tensor.hpp"

TEST_CASE("Sequential forward chains modules in order", "[sequential][forward]") {
    // Layer 1: identity weights, bias [1,1]  -> y1 = x + 1
    nrt::Tensor w1({2, 2});
    w1(0, 0) = 1.0;
    w1(0, 1) = 0.0;
    w1(1, 0) = 0.0;
    w1(1, 1) = 1.0;
    nrt::Tensor b1({2, 1});
    b1(0, 0) = 1.0;
    b1(1, 0) = 1.0;

    auto layer1 = std::make_unique<nrt::Linear>(2, 2);
    layer1->set_weights(w1, b1);

    // Layer 2: 2*identity weights, bias 0  -> y2 = 2 * y1
    nrt::Tensor w2({2, 2});
    w2(0, 0) = 2.0;
    w2(0, 1) = 0.0;
    w2(1, 0) = 0.0;
    w2(1, 1) = 2.0;
    nrt::Tensor b2({2, 1});
    b2(0, 0) = 0.0;
    b2(1, 0) = 0.0;

    auto layer2 = std::make_unique<nrt::Linear>(2, 2);
    layer2->set_weights(w2, b2);

    std::vector<std::unique_ptr<nrt::Module>> modules;
    modules.push_back(std::move(layer1));
    modules.push_back(std::move(layer2));
    nrt::Sequential model(std::move(modules));

    auto x = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{1, 2});
    (*x)(0, 0) = 3.0;
    (*x)(0, 1) = 5.0;

    // y1 = x + 1 = [4, 6]; y2 = 2 * y1 = [8, 12]
    auto y = model.forward(x);
    REQUIRE(y->shape() == std::vector<size_t>{1, 2});
    REQUIRE((*y)(0, 0) == 8.0);
    REQUIRE((*y)(0, 1) == 12.0);
}

TEST_CASE("Sequential parameters aggregates all modules' parameters in order",
          "[sequential][parameters]") {
    auto layer1 = std::make_unique<nrt::Linear>(3, 2);
    auto layer2 = std::make_unique<nrt::Linear>(2, 1);

    // Capture raw pointers BEFORE moving ownership into Sequential, so we can
    // still compare against the real underlying tensors afterward.
    nrt::Linear* layer1_ptr = layer1.get();
    nrt::Linear* layer2_ptr = layer2.get();

    std::vector<std::unique_ptr<nrt::Module>> modules;
    modules.push_back(std::move(layer1));
    modules.push_back(std::move(layer2));
    nrt::Sequential model(std::move(modules));

    auto params = model.parameters();

    // 2 params (structs) per Linear (weights, bias) x 2 layers = 4, in layer order.
    // Note that the checks happen on the references -> required as the optimzer's .step() method
    // modifies the parameters inplace
    // -> Not accidentially copies!
    REQUIRE(params.size() == 4);
    REQUIRE(params[0].value == &layer1_ptr->weights());
    REQUIRE(params[1].value == &layer1_ptr->bias());
    REQUIRE(params[2].value == &layer2_ptr->weights());
    REQUIRE(params[3].value == &layer2_ptr->bias());
}

TEST_CASE("Sequential parameters skips modules with no learnable parameters",
          "[sequential][parameters]") {
    std::vector<std::unique_ptr<nrt::Module>> modules;
    modules.push_back(std::make_unique<nrt::Linear>(2, 2));
    modules.push_back(std::make_unique<nrt::ReLU>());

    nrt::Sequential model(std::move(modules));

    // ReLU::parameters() returns {} - only the Linear's weights + bias should show up.
    REQUIRE(model.parameters().size() == 2);
}

TEST_CASE("Sequential parameter_count sums sizes across all modules", "[sequential][parameters]") {
    std::vector<std::unique_ptr<nrt::Module>> modules;
    modules.push_back(std::make_unique<nrt::Linear>(3, 2));  // weights 2*3=6, bias 2 -> 8
    modules.push_back(std::make_unique<nrt::Linear>(2, 4));  // weights 4*2=8, bias 4 -> 12

    nrt::Sequential model(std::move(modules));

    REQUIRE(model.parameter_count() == 20);
}

TEST_CASE("Sequential add appends a module to the sequence", "[sequential][add]") {
    nrt::Sequential model(std::vector<std::unique_ptr<nrt::Module>>{});

    auto layer = std::make_unique<nrt::Linear>(2, 2);
    nrt::Tensor w({2, 2});
    w(0, 0) = 1.0;
    w(0, 1) = 0.0;
    w(1, 0) = 0.0;
    w(1, 1) = 1.0;
    nrt::Tensor b({2, 1});
    b(0, 0) = 0.0;
    b(1, 0) = 0.0;
    layer->set_weights(w, b);

    model.add(std::move(layer));

    auto x = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{1, 2});
    (*x)(0, 0) = 3.0;
    (*x)(0, 1) = 5.0;

    // Identity weights, zero bias -> output should equal input.
    auto y = model.forward(x);
    REQUIRE((*y)(0, 0) == 3.0);
    REQUIRE((*y)(0, 1) == 5.0);
    REQUIRE(model.parameters().size() == 2);
}

TEST_CASE("Sequential get returns the module at a given index", "[sequential][get]") {
    std::vector<std::unique_ptr<nrt::Module>> modules;
    modules.push_back(std::make_unique<nrt::Linear>(3, 2));
    modules.push_back(std::make_unique<nrt::Linear>(2, 5));

    nrt::Sequential model(std::move(modules));

    SECTION("Valid indices return the correct module") {
        auto* m0 = dynamic_cast<nrt::Linear*>(model.get(0));
        auto* m1 = dynamic_cast<nrt::Linear*>(model.get(1));

        REQUIRE(m0 != nullptr);
        REQUIRE(m1 != nullptr);
        REQUIRE(m0->in_features() == 3);
        REQUIRE(m0->out_features() == 2);
        REQUIRE(m1->in_features() == 2);
        REQUIRE(m1->out_features() == 5);
    }

    SECTION("Out-of-range index returns nullptr") {
        REQUIRE(model.get(2) == nullptr);
    }
}

TEST_CASE("Sequential with no modules acts as identity and has no parameters",
          "[sequential][edge]") {
    nrt::Sequential model(std::vector<std::unique_ptr<nrt::Module>>{});

    auto x = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{2, 1});
    (*x)(0, 0) = 42.0;
    (*x)(1, 0) = -7.0;

    auto y = model.forward(x);

    REQUIRE(y == x);  // same shared_ptr - the for-loop in forward() never ran
    REQUIRE(model.parameters().empty());
    REQUIRE(model.parameter_count() == 0);
}
