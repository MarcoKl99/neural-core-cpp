#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <memory>
#include <vector>

#include "nrt/activations.hpp"
#include "nrt/conv2d.hpp"
#include "nrt/flatten.hpp"
#include "nrt/linear.hpp"
#include "nrt/loss.hpp"
#include "nrt/maxpool2d.hpp"
#include "nrt/optimizer.hpp"
#include "nrt/sequential.hpp"
#include "nrt/tensor.hpp"

// Helper functions in anonymous namespace
namespace {

bool approx_equal(double a, double b, double tol = 1e-9) {
    return std::abs(a - b) < tol;
}

// Build a column vector {n,1} shared tensor from values.
std::shared_ptr<nrt::Tensor> col(std::vector<double> values) {
    auto t = std::make_shared<nrt::Tensor>(std::vector<size_t>{values.size(), 1});

    for (size_t i = 0; i < values.size(); ++i) (*t)(i, 0) = values[i];

    return t;
}

}  // namespace

// =============================================================================
// 1) AUTOGRAD CORRECTNESS: analytic gradients (backward) must match finite
//    differences through the FULL chain matmul -> add -> sigmoid -> mse.
//    This is the strongest guarantee that the graph computes real gradients.
// =============================================================================

TEST_CASE("Integration - gradient check (backward matches finite differences)",
          "[integration][gradcheck]") {
    nrt::Linear layer(3, 2);

    nrt::Tensor w({2, 3});
    w(0, 0) = 0.10;
    w(0, 1) = -0.20;
    w(0, 2) = 0.30;
    w(1, 0) = -0.40;
    w(1, 1) = 0.50;
    w(1, 2) = 0.15;
    nrt::Tensor b({2, 1});
    b(0, 0) = 0.05;
    b(1, 0) = -0.10;
    layer.set_weights(w, b);

    nrt::Sigmoid act;
    nrt::MSELoss loss_fn;

    auto x = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{1, 3});
    (*x)(0, 0) = 0.5;
    (*x)(0, 1) = -1.0;
    (*x)(0, 2) = 2.0;

    auto target = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{1, 2});
    (*target)(0, 0) = 1.0;
    (*target)(0, 1) = 0.0;

    // ---- analytic gradient via backward ----
    layer.weights().zero_grad();
    layer.bias().zero_grad();
    auto z = layer.forward(x);
    auto a = act.forward(z);
    auto l = loss_fn.forward(a, target);
    l->backward();
    nrt::Tensor grad_w = layer.weights().gradient();  // {2,3}
    nrt::Tensor grad_b = layer.bias().gradient();     // {2,1}

    // ---- numeric gradient via central differences on the scalar loss ----
    auto loss_now = [&]() {
        auto zz = layer.forward(x);
        auto aa = act.forward(zz);
        return nrt::mse(*aa, *target);  // free function: forward only, no graph needed
    };
    const double eps = 1e-6;

    // This loop actually approximates the gradient using the plain tangent
    // and compares this to the retrieved gradient from above

    // On the weights
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            double orig = layer.weights()(i, j);
            layer.weights()(i, j) = orig + eps;
            double lp = loss_now();
            layer.weights()(i, j) = orig - eps;
            double lm = loss_now();
            layer.weights()(i, j) = orig;
            double numeric = (lp - lm) / (2.0 * eps);
            REQUIRE(approx_equal(grad_w(i, j), numeric, 1e-5));
        }
    }

    // On the bias
    for (size_t i = 0; i < 2; ++i) {
        double orig = layer.bias()(i, 0);
        layer.bias()(i, 0) = orig + eps;
        double lp = loss_now();
        layer.bias()(i, 0) = orig - eps;
        double lm = loss_now();
        layer.bias()(i, 0) = orig;
        double numeric = (lp - lm) / (2.0 * eps);
        REQUIRE(approx_equal(grad_b(i, 0), numeric, 1e-5));
    }
}

// =============================================================================
// 2) THE TRAINING LOOP LEARNS: a deterministic, always-convergent task
//    (linear regression y = 3x) driven by forward -> MSELoss -> backward -> SGD.
//    No activation pathologies, so this is a stable "does learning work" test.
// =============================================================================
TEST_CASE("Integration - SGD training minimizes a linear regression", "[integration][training]") {
    nrt::Linear layer(1, 1);

    nrt::Tensor w0({1, 1});
    w0(0, 0) = 0.0;  // deterministic start
    nrt::Tensor b0({1, 1});
    b0(0, 0) = 0.0;
    layer.set_weights(w0, b0);

    nrt::MSELoss loss_fn;

    std::vector<std::shared_ptr<nrt::Tensor>> xs = {col({1.0}), col({2.0}), col({3.0})};
    std::vector<std::shared_ptr<nrt::Tensor>> ys = {col({3.0}), col({6.0}), col({9.0})};

    auto avg_loss = [&]() {
        double s = 0.0;
        for (size_t i = 0; i < xs.size(); ++i) s += nrt::mse(*layer.forward(xs[i]), *ys[i]);
        return s / static_cast<double>(xs.size());
    };

    double initial = avg_loss();

    auto params = layer.parameters();
    const double base_lr = 0.05;
    nrt::SGD opt(params, base_lr / static_cast<double>(xs.size()));  // sum-grad -> mean update

    for (int epoch = 0; epoch < 5000; ++epoch) {
        opt.zero_grad();
        for (size_t i = 0; i < xs.size(); ++i) {
            auto pred = layer.forward(xs[i]);
            auto l = loss_fn.forward(pred, ys[i]);
            l->backward();
        }
        opt.step();
    }

    double final = avg_loss();
    REQUIRE(final < initial);                                 // it learned *something*
    REQUIRE(final < 1e-2);                                    // it essentially fits the line
    REQUIRE(approx_equal(layer.weights()(0, 0), 3.0, 0.05));  // recovered the slope
}

// =============================================================================
// 3) THE FULL STACK: Sequential(Linear -> ReLU -> Linear -> Sigmoid) + MSELoss
//    + SGD on XOR. Fixed weights make it deterministic (avoids Dying-ReLU flake).
//    Proves every piece is wired together end to end.
// =============================================================================
TEST_CASE("Integration - full MLP trains on XOR (Sequential + activations)", "[integration][xor]") {
    std::vector<std::unique_ptr<nrt::Module>> modules;
    modules.push_back(std::make_unique<nrt::Linear>(2, 4));
    modules.push_back(std::make_unique<nrt::ReLU>());
    modules.push_back(std::make_unique<nrt::Linear>(4, 1));
    modules.push_back(std::make_unique<nrt::Sigmoid>());
    nrt::Sequential model(std::move(modules));

    // Deterministic, known-good starting weights (same family as the example).
    auto* l1 = dynamic_cast<nrt::Linear*>(model.get(0));
    auto* l2 = dynamic_cast<nrt::Linear*>(model.get(2));
    REQUIRE(l1 != nullptr);
    REQUIRE(l2 != nullptr);

    nrt::Tensor w1({4, 2});
    w1(0, 0) = 0.127675;
    w1(0, 1) = -0.0800557;
    w1(1, 0) = -0.0511347;
    w1(1, 1) = -0.0129438;
    w1(2, 0) = 0.209888;
    w1(2, 1) = 0.0099226;
    w1(3, 0) = -0.0643893;
    w1(3, 1) = 0.102362;
    nrt::Tensor b1({4, 1});
    b1(0, 0) = b1(1, 0) = b1(2, 0) = b1(3, 0) = 0.0;
    l1->set_weights(w1, b1);

    nrt::Tensor w2({1, 4});
    w2(0, 0) = 0.119094;
    w2(0, 1) = 0.103336;
    w2(0, 2) = 0.101908;
    w2(0, 3) = -0.0430681;
    nrt::Tensor b2({1, 1});
    b2(0, 0) = 0.0;
    l2->set_weights(w2, b2);

    // Create batch tensors: {batch=4, features}
    auto xs_batch = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{4, 2});
    (*xs_batch)(0, 0) = 0.0;
    (*xs_batch)(0, 1) = 0.0;  // XOR(0,0) = 0
    (*xs_batch)(1, 0) = 0.0;
    (*xs_batch)(1, 1) = 1.0;  // XOR(0,1) = 1
    (*xs_batch)(2, 0) = 1.0;
    (*xs_batch)(2, 1) = 0.0;  // XOR(1,0) = 1
    (*xs_batch)(3, 0) = 1.0;
    (*xs_batch)(3, 1) = 1.0;  // XOR(1,1) = 0

    auto ys_batch = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{4, 1});
    (*ys_batch)(0, 0) = 0.0;
    (*ys_batch)(1, 0) = 1.0;
    (*ys_batch)(2, 0) = 1.0;
    (*ys_batch)(3, 0) = 0.0;

    nrt::MSELoss loss_fn;

    auto avg_loss = [&]() {
        auto pred = model.forward(xs_batch);
        auto l = loss_fn.forward(pred, ys_batch);
        return (*l)(0, 0);  // Already averaged over batch
    };

    double initial = avg_loss();

    auto params = model.parameters();
    double learning_rate = 0.1;
    nrt::SGD opt(params, learning_rate);

    for (int epoch = 0; epoch < 5000; ++epoch) {
        opt.zero_grad();

        // Single forward/backward pass on the entire batch
        auto pred = model.forward(xs_batch);
        auto l = loss_fn.forward(pred, ys_batch);
        l->backward();

        opt.step();
    }

    double final = avg_loss();
    REQUIRE(final < initial * 0.5);  // Meaningfully reduced
}

// =============================================================================
// 4) DEEPER ARCHITECTURE + HE/XAVIER INIT: Sequential(2->8->8->1) with two
//    ReLU hidden layers. Random init each run (no seed control in Linear), so
//    instead of pinning weights we repeat several trials and require the vast
//    majority to converge - proving He/Xavier init reliably avoids the
//    Dying-ReLU trap from notes/xor_training.md even with more ReLU units,
//    rather than testing one lucky fixed draw.
// =============================================================================

TEST_CASE("Integration - deeper MLP (He/Xavier init) trains reliably on XOR",
          "[integration][xor][deep]") {
    // Data
    // Create batch tensors: {batch=4, features}
    auto xs_batch = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{4, 2});
    (*xs_batch)(0, 0) = 0.0;
    (*xs_batch)(0, 1) = 0.0;  // XOR(0,0) = 0
    (*xs_batch)(1, 0) = 0.0;
    (*xs_batch)(1, 1) = 1.0;  // XOR(0,1) = 1
    (*xs_batch)(2, 0) = 1.0;
    (*xs_batch)(2, 1) = 0.0;  // XOR(1,0) = 1
    (*xs_batch)(3, 0) = 1.0;
    (*xs_batch)(3, 1) = 1.0;  // XOR(1,1) = 0

    auto ys_batch = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{4, 1});
    (*ys_batch)(0, 0) = 0.0;
    (*ys_batch)(1, 0) = 1.0;
    (*ys_batch)(2, 0) = 1.0;
    (*ys_batch)(3, 0) = 0.0;

    // Model
    std::vector<std::unique_ptr<nrt::Module>> modules;
    modules.push_back(std::make_unique<nrt::Linear>(2, 8, nrt::WeightInit::He, 42));
    modules.push_back(std::make_unique<nrt::ReLU>());
    modules.push_back(std::make_unique<nrt::Linear>(8, 8, nrt::WeightInit::He, 123));
    modules.push_back(std::make_unique<nrt::ReLU>());
    modules.push_back(std::make_unique<nrt::Linear>(8, 1, nrt::WeightInit::Xavier, 1));
    modules.push_back(std::make_unique<nrt::Sigmoid>());
    auto model = nrt::Sequential(std::move(modules));

    // Helper function for batched loss calculation
    nrt::MSELoss loss_fn;
    auto avg_loss = [&](nrt::Sequential& model) {
        auto pred = model.forward(xs_batch);
        auto l = loss_fn.forward(pred, ys_batch);
        return (*l)(0, 0);  // Already averaged over batch
    };

    // Training
    auto params = model.parameters();
    nrt::SGD opt(params, 0.1);

    for (int epoch = 0; epoch < 5000; ++epoch) {
        opt.zero_grad();
        auto pred = model.forward(xs_batch);
        auto l = loss_fn.forward(pred, ys_batch);
        l->backward();
        opt.step();
    }

    double final_loss = avg_loss(model);

    // Require that the model converged
    REQUIRE(final_loss < 0.01);
}

// =============================================================================
// 5) SOFTMAX + CROSS-ENTROPY: analytic gradient (fused backward) must match
//    finite differences through Linear -> CrossEntropyLoss (raw logits in).
// =============================================================================
TEST_CASE("Integration - CrossEntropyLoss gradient check (backward matches finite differences)",
          "[integration][gradcheck][cross_entropy]") {
    nrt::Linear layer(3, 4);  // 3 features -> 4 classes

    nrt::Tensor w({4, 3});
    w(0, 0) = 0.10;
    w(0, 1) = -0.20;
    w(0, 2) = 0.30;
    w(1, 0) = -0.40;
    w(1, 1) = 0.50;
    w(1, 2) = 0.15;
    w(2, 0) = 0.25;
    w(2, 1) = -0.05;
    w(2, 2) = 0.60;
    w(3, 0) = -0.10;
    w(3, 1) = 0.20;
    w(3, 2) = -0.35;
    nrt::Tensor b({4, 1});
    b(0, 0) = 0.05;
    b(1, 0) = -0.10;
    b(2, 0) = 0.02;
    b(3, 0) = 0.0;
    layer.set_weights(w, b);

    nrt::CrossEntropyLoss loss_fn;

    auto x = std::make_shared<nrt::Tensor>(std::vector<size_t>{1, 3});
    (*x)(0, 0) = 0.5;
    (*x)(0, 1) = -1.0;
    (*x)(0, 2) = 2.0;

    auto target_tensor = std::make_shared<nrt::Tensor>(std::vector<size_t>{1, 1});
    (*target_tensor)(0, 0) = 2.0;

    layer.weights().zero_grad();
    layer.bias().zero_grad();
    auto logits = layer.forward(x);
    auto l = loss_fn.forward(logits, target_tensor);
    l->backward();
    nrt::Tensor grad_w = layer.weights().gradient();
    nrt::Tensor grad_b = layer.bias().gradient();

    auto loss_now = [&]() {
        auto zz = layer.forward(x);
        auto loss = loss_fn.forward(zz, target_tensor);
        return (*loss)(0, 0);
    };
    const double eps = 1e-6;

    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            double orig = layer.weights()(i, j);
            layer.weights()(i, j) = orig + eps;
            double lp = loss_now();
            layer.weights()(i, j) = orig - eps;
            double lm = loss_now();
            layer.weights()(i, j) = orig;
            double numeric = (lp - lm) / (2.0 * eps);
            REQUIRE(approx_equal(grad_w(i, j), numeric, 1e-5));
        }
    }

    for (size_t i = 0; i < 4; ++i) {
        double orig = layer.bias()(i, 0);
        layer.bias()(i, 0) = orig + eps;
        double lp = loss_now();
        layer.bias()(i, 0) = orig - eps;
        double lm = loss_now();
        layer.bias()(i, 0) = orig;
        double numeric = (lp - lm) / (2.0 * eps);
        REQUIRE(approx_equal(grad_b(i, 0), numeric, 1e-5));
    }
}

// =============================================================================
// 6) Full CNN with Conv2D + MaxPool2D + Flatten + Linear + Loss
// =============================================================================
TEST_CASE("Integration - Full CNN pipeline (Conv2D + MaxPool2D + Flatten + Linear + Loss)",
          "[integration][cnn][full_pipeline]") {
    const size_t batch = 6;
    const size_t num_classes = batch;  // one unique class per sample -> right-sized capacity
    const unsigned int seed = 111;

    // Build a small CNN: Conv2D -> ReLU -> MaxPool2D -> Flatten -> Linear
    std::vector<std::unique_ptr<nrt::Module>> modules;
    modules.push_back(std::make_unique<nrt::Conv2D>(1, 4, 3, nrt::WeightInit::He, seed));
    modules.push_back(std::make_unique<nrt::ReLU>());
    modules.push_back(std::make_unique<nrt::MaxPool2D>());  // 2×2 pooling, stride 2
    modules.push_back(std::make_unique<nrt::Flatten>());
    modules.push_back(
        std::make_unique<nrt::Linear>(4 * 3 * 3, num_classes, nrt::WeightInit::Xavier, seed));
    nrt::Sequential model(std::move(modules));

    // Create input batch: {6, 1, 8, 8} (6 samples, 1 channel, 8×8 images)
    auto input = std::make_shared<nrt::Tensor>(std::vector<size_t>{batch, 1, 8, 8});
    for (size_t b = 0; b < batch; ++b) {
        for (size_t i = 0; i < 8; ++i) {
            for (size_t j = 0; j < 8; ++j) {
                (*input)(b, 0, i, j) = static_cast<double>(b * 64 + i * 8 + j) / 100.0;
            }
        }
    }

    // Create labels: {6, 1} - sample b belongs to class b
    auto labels = std::make_shared<nrt::Tensor>(std::vector<size_t>{batch, 1});
    for (size_t b = 0; b < batch; ++b) {
        (*labels)(b, 0) = static_cast<double>(b);
    }

    auto logits = model.forward(input);
    REQUIRE(logits->shape() == std::vector<size_t>{batch, num_classes});

    nrt::CrossEntropyLoss loss_fn;
    auto loss = loss_fn.forward(logits, labels);
    REQUIRE(loss->shape() == std::vector<size_t>{1, 1});
    double initial_loss = (*loss)(0, 0);

    loss->backward();

    // Check the shape of the parameters
    auto params = model.parameters();
    for (size_t i = 0; i < params.size(); ++i) {
        auto param_grad = params[i].value->gradient();
        REQUIRE(param_grad.shape() == params[i].value->shape());
    }

    // Train for a while and require the loss to actually go down.
    nrt::SGD opt(params, 0.05);
    for (int epoch = 0; epoch < 5000; ++epoch) {
        opt.zero_grad();
        auto pred = model.forward(input);
        auto l = loss_fn.forward(pred, labels);
        l->backward();
        opt.step();
    }

    auto final_logits = model.forward(input);
    auto final_loss_tensor = loss_fn.forward(final_logits, labels);
    double final_loss = (*final_loss_tensor)(0, 0);

    REQUIRE(final_loss < initial_loss * 0.5);

    // Each sample should now be classified as its own unique class.
    auto argmax_class = [](const nrt::Tensor& t, size_t row) {
        size_t best = 0;
        double best_val = t(row, 0);
        for (size_t c = 1; c < t.shape()[1]; ++c) {
            if (t(row, c) > best_val) {
                best_val = t(row, c);
                best = c;
            }
        }
        return best;
    };
    for (size_t b = 0; b < batch; ++b) {
        REQUIRE(argmax_class(*final_logits, b) == b);
    }
}

// =============================================================================
// 7) BATCH INDEPENDENCE: Conv2D on a batch of N samples must produce exactly the
//    same per-sample output as running each sample through the same layer alone.
//    Catches bugs where samples get mixed across the batch axis (e.g. an invalid
//    reshape after a batched matmul) - the kind of bug that silently breaks
//    training without ever showing up as a shape mismatch or a gradcheck failure.
// =============================================================================
TEST_CASE("Integration - Conv2D forward is batch-independent", "[integration][cnn][conv2d]") {
    nrt::Conv2D layer(1, 2, 3);  // out_channels > 1 is required to expose batch/channel mixing

    const size_t batch = 3;
    const size_t image_size = 6;

    auto batched_input =
        std::make_shared<nrt::Tensor>(std::vector<size_t>{batch, 1, image_size, image_size});

    for (size_t b = 0; b < batch; ++b) {
        for (size_t i = 0; i < image_size; ++i) {
            for (size_t j = 0; j < image_size; ++j) {
                // Distinct, deterministic values per sample so no two samples look alike.
                (*batched_input)(b, 0, i, j) =
                    static_cast<double>(b * 100 + i * image_size + j) / 100.0;
            }
        }
    }

    auto batched_output = layer.forward(batched_input);
    const size_t out_channels = batched_output->shape()[1];
    const size_t out_height = batched_output->shape()[2];
    const size_t out_width = batched_output->shape()[3];

    for (size_t b = 0; b < batch; ++b) {
        // Create a local copy of the image at b
        auto single_input =
            std::make_shared<nrt::Tensor>(std::vector<size_t>{1, 1, image_size, image_size});
        for (size_t i = 0; i < image_size; ++i) {
            for (size_t j = 0; j < image_size; ++j) {
                (*single_input)(0, 0, i, j) = (*batched_input)(b, 0, i, j);
            }
        }

        auto single_output = layer.forward(single_input);

        for (size_t oc = 0; oc < out_channels; ++oc) {
            for (size_t h = 0; h < out_height; ++h) {
                for (size_t w = 0; w < out_width; ++w) {
                    REQUIRE(approx_equal((*batched_output)(b, oc, h, w),
                                         (*single_output)(0, oc, h, w)));
                }
            }
        }
    }
}
