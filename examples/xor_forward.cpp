#include <iostream>

#include "nrt/activations.hpp"
#include "nrt/linear.hpp"
#include "nrt/loss.hpp"
#include "nrt/tensor.hpp"

int main() {
    // NN: 2 -> Linear -> 4 -> ReLU -> Linear -> 1 -> Sigmoid
    nrt::Linear layer1(2, 4);
    nrt::Linear layer2(4, 1);

    // XOR-Table: Inputs and expected targets
    std::vector<nrt::Tensor> inputs;
    std::vector<nrt::Tensor> targets;

    // Lambda function to create an input Tensor (e.g. {0, 1})
    auto make_input = [](double a, double b) {
        nrt::Tensor x({2, 1});
        x(0, 0) = a;
        x(1, 0) = b;
        return x;
    };

    // Lambda function to create an output tensor (e.g. {1})
    auto make_target = [](double t) {
        nrt::Tensor y({1, 1});
        y(0, 0) = t;
        return y;
    };

    // Fill the inputs
    inputs.push_back(make_input(0.0, 0.0));
    inputs.push_back(make_input(0.0, 1.0));
    inputs.push_back(make_input(1.0, 0.0));
    inputs.push_back(make_input(1.0, 1.0));

    // Fill the outputs / labels
    targets.push_back(make_target(0.0));
    targets.push_back(make_target(1.0));
    targets.push_back(make_target(1.0));
    targets.push_back(make_target(0.0));

    double total_loss = 0.0;

    for (size_t i = 0; i < inputs.size(); ++i) {
        const nrt::Tensor& x = inputs[i];
        const nrt::Tensor& target = targets[i];

        // Forward Pass
        nrt::Tensor hidden = nrt::relu(layer1.forward(x));
        nrt::Tensor output = nrt::sigmoid(layer2.forward(hidden));

        double loss = nrt::mse(output, target);
        total_loss += loss;

        std::cout << "Input: (" << x(0, 0) << ", " << x(1, 0) << ")"
                  << " -> Output: " << output(0, 0) << " (Target: " << target(0, 0) << ")"
                  << ", Loss: " << loss << '\n';
    }

    std::cout << "Average loss: " << total_loss / inputs.size() << '\n';

    return 0;
}
