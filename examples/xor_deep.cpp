#include <iostream>
#include <vector>

#include "nrt/activations.hpp"
#include "nrt/linear.hpp"
#include "nrt/loss.hpp"
#include "nrt/module.hpp"
#include "nrt/optimizer.hpp"
#include "nrt/sequential.hpp"
#include "nrt/tensor.hpp"

/*
Deeper MLP on XOR: Linear(2,8,He) -> ReLU -> Linear(8,8,He) -> ReLU -> Linear(8,1,Xavier) ->
Sigmoid. Demonstrates He/Xavier weight init (see include/nrt/linear.hpp) reducing the Dying ReLU
risk described in notes/xor_training.md, even with more ReLU units than the original 2->4->1
example.
*/

namespace {

std::shared_ptr<nrt::Tensor> make_input(double a, double b) {
    auto x = std::make_shared<nrt::Tensor>(std::vector<size_t>{2, 1});
    (*x)(0, 0) = a;
    (*x)(1, 0) = b;
    return x;
}

std::shared_ptr<nrt::Tensor> make_target(double t) {
    auto y = std::make_shared<nrt::Tensor>(std::vector<size_t>{1, 1});
    (*y)(0, 0) = t;
    return y;
}

double evaluate_average_loss(nrt::Sequential& model,
                             std::vector<std::shared_ptr<nrt::Tensor>>& inputs,
                             const std::vector<std::shared_ptr<nrt::Tensor>>& targets) {
    double total_loss = 0.0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto y_hat = model.forward(inputs[i]);
        total_loss += nrt::mse(*y_hat, *targets[i]);
    }
    return total_loss / static_cast<double>(inputs.size());
}

}  // namespace

int main() {
    /**************************/
    /*    Create the model    */
    /**************************/
    std::vector<std::unique_ptr<nrt::Module>> modules;
    modules.push_back(std::make_unique<nrt::Linear>(2, 8, nrt::WeightInit::He, 42));
    modules.push_back(std::make_unique<nrt::ReLU>());
    modules.push_back(std::make_unique<nrt::Linear>(8, 8, nrt::WeightInit::He, 123));
    modules.push_back(std::make_unique<nrt::ReLU>());
    modules.push_back(std::make_unique<nrt::Linear>(8, 1, nrt::WeightInit::Xavier, 1));
    modules.push_back(std::make_unique<nrt::Sigmoid>());

    nrt::Sequential model(std::move(modules));

    /**************************/
    /*   Create the dataset   */
    /**************************/
    std::vector<std::shared_ptr<nrt::Tensor>> inputs = {make_input(0.0, 0.0), make_input(0.0, 1.0),
                                                        make_input(1.0, 0.0), make_input(1.0, 1.0)};
    std::vector<std::shared_ptr<nrt::Tensor>> targets = {make_target(0.0), make_target(1.0),
                                                         make_target(1.0), make_target(0.0)};

    /**************************/
    /*        Training        */
    /**************************/
    const int epochs = 5000;
    const double learning_rate = 0.1;
    const size_t batch_size = inputs.size();
    nrt::MSELoss loss_fn;
    nrt::SGD optimizer(model.parameters(), learning_rate / batch_size);

    double loss_before = evaluate_average_loss(model, inputs, targets);
    std::cout << "Average loss BEFORE update: " << loss_before << '\n';

    for (int epoch = 0; epoch < epochs; ++epoch) {
        optimizer.zero_grad();
        for (size_t j = 0; j < inputs.size(); ++j) {
            auto y_hat = model.forward(inputs[j]);
            auto loss = loss_fn.forward(y_hat, targets[j]);
            loss->backward();
        }
        optimizer.step();

        if ((epoch % 1000) == 0) {
            double loss = evaluate_average_loss(model, inputs, targets);
            std::cout << "Loss (" << epoch << "/" << epochs << "):  " << loss << '\n';
        }
    }

    double final_loss = evaluate_average_loss(model, inputs, targets);
    std::cout << "Loss (" << epochs << "/" << epochs << "):  " << final_loss << '\n';

    return 0;
}
