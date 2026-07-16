#include <iostream>
#include <vector>

#include "nrt/activations.hpp"
#include "nrt/linear.hpp"
#include "nrt/loss.hpp"
#include "nrt/module.hpp"
#include "nrt/optimizer.hpp"
#include "nrt/parameter.hpp"
#include "nrt/sequential.hpp"
#include "nrt/tensor.hpp"

/*
Manual MLP training on the XOR problem: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0.
Architecture: Linear -> ReLU -> Linear -> Sigmoid -> MSE.
Now uses batched training: single forward/backward pass for all 4 samples per epoch.
*/

namespace {

// Print the initial weights for the comparison to PyTorch
void print_weights(nrt::Linear& layer1, nrt::Linear& layer2) {
    std::cout << "=== INITIAL WEIGHTS (for reference validation) ===\n";

    std::cout << "\n// Layer 1 weights (4x2)\n";
    std::cout << "std::vector<double> w1 = {\n";
    const auto& w1 = layer1.weights();
    for (size_t i = 0; i < w1.size(); ++i) {
        if (i % 2 == 0) std::cout << "    ";
        std::cout << w1(i / 2, i % 2);
        if (i < w1.size() - 1) std::cout << ", ";
        if (i % 2 == 1) std::cout << "\n";
    }
    std::cout << "};\n";

    std::cout << "\n// Layer 1 bias (4x1)\n";
    std::cout << "std::vector<double> b1 = {\n";
    const auto& b1 = layer1.bias();
    for (size_t i = 0; i < b1.size(); ++i) {
        std::cout << "    " << b1(i, 0) << ",\n";
    }
    std::cout << "};\n";

    std::cout << "\n// Layer 2 weights (1x4)\n";
    std::cout << "std::vector<double> w2 = {\n";
    const auto& w2 = layer2.weights();
    for (size_t i = 0; i < w2.size(); ++i) {
        std::cout << "    " << w2(0, i);
        if (i < w2.size() - 1) std::cout << ",";
        std::cout << "\n";
    }
    std::cout << "};\n";

    std::cout << "\n// Layer 2 bias (1x1)\n";
    std::cout << "std::vector<double> b2 = {\n";
    std::cout << "    " << layer2.bias()(0, 0) << "\n";
    std::cout << "};\n\n";
}

// Set fixed weights to match PyTorch reference validation (v0.2)
void set_reference_weights(nrt::Linear& layer1, nrt::Linear& layer2) {
    // Layer 1: 2 -> 4 with ReLU
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
    b1(0, 0) = 0.0;
    b1(1, 0) = 0.0;
    b1(2, 0) = 0.0;
    b1(3, 0) = 0.0;

    // Layer 2: 4 -> 1 with Sigmoid
    nrt::Tensor w2({1, 4});
    w2(0, 0) = 0.119094;
    w2(0, 1) = 0.103336;
    w2(0, 2) = 0.101908;
    w2(0, 3) = -0.0430681;

    nrt::Tensor b2({1, 1});
    b2(0, 0) = 0.0;

    layer1.set_weights(w1, b1);
    layer2.set_weights(w2, b2);
}

double evaluate_average_loss(nrt::Sequential& model, const std::shared_ptr<nrt::Tensor>& inputs,
                             const std::shared_ptr<nrt::Tensor>& targets) {
    auto y_hat = model.forward(inputs);
    auto loss = nrt::mse(*y_hat, *targets);
    return loss;
}

}  // namespace

int main() {
    // Create the neural network using Sequential: 2 -> 4 (ReLU) -> 1 (Sigmoid)
    std::vector<std::unique_ptr<nrt::Module>> modules;
    modules.push_back(std::make_unique<nrt::Linear>(2, 4));
    modules.push_back(std::make_unique<nrt::ReLU>());
    modules.push_back(std::make_unique<nrt::Linear>(4, 1));
    modules.push_back(std::make_unique<nrt::Sigmoid>());

    nrt::Sequential model(std::move(modules));

    // Set reference weights (access layers via index)
    auto layer1 = dynamic_cast<nrt::Linear*>(model.get(0));
    auto layer2 = dynamic_cast<nrt::Linear*>(model.get(2));
    set_reference_weights(*layer1, *layer2);

    // Define all 4 possible training samples
    // (0, 0) = 0, (0, 1) = 1, (1, 0) = 1, (1, 1) = 0
    auto inputs = std::make_shared<nrt::Tensor>(std::vector<size_t>{4, 2});
    (*inputs)(0, 0) = 0.0;
    (*inputs)(0, 1) = 0.0;  // (0, 0)
    (*inputs)(1, 0) = 0.0;
    (*inputs)(1, 1) = 1.0;  // (0, 1)
    (*inputs)(2, 0) = 1.0;
    (*inputs)(2, 1) = 0.0;  // (1, 0)
    (*inputs)(3, 0) = 1.0;
    (*inputs)(3, 1) = 1.0;  // (1, 1)

    auto targets = std::make_shared<nrt::Tensor>(std::vector<size_t>{4, 1});
    (*targets)(0, 0) = 0.0;
    (*targets)(1, 0) = 1.0;
    (*targets)(2, 0) = 1.0;
    (*targets)(3, 0) = 0.0;

    // Define the training parameters
    const int epochs = 5000;
    const double learning_rate = 0.1;
    nrt::MSELoss loss_fn;
    nrt::SGD optimizer(model.parameters(), learning_rate);

    double loss_before = evaluate_average_loss(model, inputs, targets);
    std::cout << "Average loss BEFORE update: " << loss_before << '\n';

    for (int epoch = 0; epoch < epochs; ++epoch) {
        optimizer.zero_grad();

        // Single batched forward/backward
        auto y_hat = model.forward(inputs);
        auto loss = loss_fn.forward(y_hat, targets);
        loss->backward();

        optimizer.step();

        if ((epoch % 1000) == 0) {
            const double avg_loss = evaluate_average_loss(model, inputs, targets);
            std::cout << "Loss (" << epoch << "/" << epochs << "):  " << avg_loss << '\n';
        }
    }

    // Final loss output
    double loss_after = evaluate_average_loss(model, inputs, targets);
    std::cout << "Loss (" << epochs << "/" << epochs << "):  " << loss_after << '\n';

    return 0;
}
