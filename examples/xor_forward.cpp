#include <iostream>
#include <memory>

#include "nrt/activations.hpp"
#include "nrt/linear.hpp"
#include "nrt/loss.hpp"
#include "nrt/tensor.hpp"

int main() {
    // NN: 2 -> Linear -> 4 -> ReLU -> Linear -> 1 -> Sigmoid
    nrt::Linear layer1(2, 4);
    nrt::Linear layer2(4, 1);

    // XOR table: {4, 2} batch of inputs
    auto inputs = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{4, 2});
    (*inputs)(0, 0) = 0.0;
    (*inputs)(0, 1) = 0.0;  // XOR(0, 0) = 0
    (*inputs)(1, 0) = 0.0;
    (*inputs)(1, 1) = 1.0;  // XOR(0, 1) = 1
    (*inputs)(2, 0) = 1.0;
    (*inputs)(2, 1) = 0.0;  // XOR(1, 0) = 1
    (*inputs)(3, 0) = 1.0;
    (*inputs)(3, 1) = 1.0;  // XOR(1, 1) = 0

    // Expected outputs: {4, 1} batch of targets
    auto targets = std::make_shared<nrt::Tensor>(std::vector<std::size_t>{4, 1});
    (*targets)(0, 0) = 0.0;
    (*targets)(1, 0) = 1.0;
    (*targets)(2, 0) = 1.0;
    (*targets)(3, 0) = 0.0;

    auto relu = nrt::ReLU{};
    auto sigmoid = nrt::Sigmoid{};
    auto criterion = nrt::MSELoss{};

    // Single batched forward pass
    auto hidden = relu.forward(layer1.forward(inputs));
    auto output = sigmoid.forward(layer2.forward(hidden));

    auto loss = criterion.forward(output, targets);
    double avg_loss = (*loss)(0, 0);  // MSE already averaged over batch

    std::cout << "Sample losses: ";
    for (size_t i = 0; i < 4; ++i) {
        std::cout << (*output)(i, 0) << " ";
    }
    std::cout << "\nAverage loss: " << avg_loss << '\n';

    return 0;

    return 0;
}
