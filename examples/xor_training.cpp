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
Full-batch gradient descent: forward + backward accumulate gradients across all 4 samples,
then weights/biases are updated manually (no Optimizer class yet, see roadmap v0.2).

Note: depending on random weight initialization, this can occasionally get stuck due to the
"Dying ReLU" problem - see notes/dying_relu_observation.md for details.
*/

namespace {

// Average accumulated gradients by batch size
void average_gradients(std::vector<nrt::Parameter>& params, size_t batch_size) {
    for (auto& param : params) {
        *param.gradient = *param.gradient * (1.0 / batch_size);
    }
}

// Print the initial weights for the comparison to PyTorch
void print_weights(nrt::Linear& layer1, nrt::Linear& layer2) {
    // Print initial weights for reference validation
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

nrt::Tensor make_input(double a, double b) {
    nrt::Tensor x({2, 1});
    x(0, 0) = a;
    x(1, 0) = b;
    return x;
}

nrt::Tensor make_target(double t) {
    nrt::Tensor y({1, 1});
    y(0, 0) = t;
    return y;
}

// Execute the full forward-backward iteration for one single step
// and returns the loss for this sample
double forward_backward_step(nrt::Linear& layer1, nrt::Linear& layer2, const nrt::Tensor& x,
                             const nrt::Tensor& target) {
    // Forward
    nrt::Tensor z1 = layer1.forward(x);
    nrt::Tensor a1 = nrt::relu(z1);
    nrt::Tensor z2 = layer2.forward(a1);
    nrt::Tensor y_hat = nrt::sigmoid(z2);

    double loss = nrt::mse(y_hat, target);

    // Backward
    nrt::Tensor grad_y_hat = nrt::mse_derivative(y_hat, target);
    nrt::Tensor grad_z2 = nrt::sigmoid_backward(grad_y_hat, z2);
    nrt::Tensor grad_a1 = layer2.backward(grad_z2);
    nrt::Tensor grad_z1 = nrt::relu_backward(grad_a1, z1);
    layer1.backward(grad_z1);

    return loss;
}

double evaluate_average_loss(nrt::Sequential& model, const std::vector<nrt::Tensor>& inputs,
                             const std::vector<nrt::Tensor>& targets) {
    double total_loss = 0.0;

    for (size_t i = 0; i < inputs.size(); ++i) {
        // Forward pass using the model
        nrt::Tensor y_hat = model.forward(inputs[i]);

        // Loss calculation
        total_loss += nrt::mse(y_hat, targets[i]);
    }

    return total_loss / static_cast<double>(inputs.size());
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

    // Get all parameters and create optimizer
    auto params = model.parameters();
    const double learning_rate = 0.1;
    nrt::SGD optimizer(params, learning_rate);

    // Define all 4 possible training samples
    // (0, 0) = 0, (0, 1) = 1, (1, 0) = 1, (1, 1) = 0
    std::vector<nrt::Tensor> inputs = {make_input(0.0, 0.0), make_input(0.0, 1.0),
                                       make_input(1.0, 0.0), make_input(1.0, 1.0)};
    std::vector<nrt::Tensor> targets = {make_target(0.0), make_target(1.0), make_target(1.0),
                                        make_target(0.0)};

    // Define the training parameters
    const int epochs = 5000;

    // Calculate the loss before training
    double loss_before = evaluate_average_loss(model, inputs, targets);
    std::cout << "Average loss BEFORE update: " << loss_before << '\n';

    for (int i = 0; i < epochs; ++i) {
        // One complete full-batch gradient descent
        optimizer.zero_grad();

        // Accumulate gradients across all samples
        for (size_t j = 0; j < inputs.size(); ++j) {
            forward_backward_step(*layer1, *layer2, inputs[j], targets[j]);
        }

        // Average gradients and apply optimizer step - Averaging step to be refactored later
        average_gradients(params, inputs.size());
        optimizer.step();

        // Check if the loss has improved
        if ((i % 1000) == 0) {
            double loss_after = evaluate_average_loss(model, inputs, targets);
            std::cout << "Loss (" << i << "/" << epochs << "):  " << loss_after << '\n';
        }
    }

    // Final loss output
    double loss_after = evaluate_average_loss(model, inputs, targets);
    std::cout << "Loss (" << epochs << "/" << epochs << "):  " << loss_after << '\n';

    return 0;
}
