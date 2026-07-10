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
    std::vector<std::shared_ptr<nrt::Tensor>> inputs;
    std::vector<std::shared_ptr<nrt::Tensor>> targets;

    // Lambda function to create an input Tensor (e.g. {0, 1})
    auto make_input = [](double a, double b) {
        auto x = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{2, 1});
        (*x)(0, 0) = a;
        (*x)(1, 0) = b;
        return x;
    };

    // Lambda function to create an output tensor (e.g. {1})
    auto make_target = [](double t) {
        auto y = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{1, 1});
        (*y)(0, 0) = t;
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

    // Create the ReLU and Sigmoid activation functions
    auto relu = nrt::ReLU{};
    auto sigmoid = nrt::Sigmoid{};
    auto criterion = nrt::MSELoss{};

    for (size_t i = 0; i < inputs.size(); ++i) {
        auto x = inputs[i];
        auto target = targets[i];

        // Forward Pass
        auto hidden = relu.forward(layer1.forward(x));
        auto output = sigmoid.forward(layer2.forward(hidden));

        auto loss = criterion.forward(output, target);
        total_loss += (*loss)(0, 0);

        std::cout << "Loss: " << (*loss)(0, 0) << '\n';
    }

    std::cout << "Average loss: " << total_loss / inputs.size() << '\n';

    return 0;
}
