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
Sigmoid. Now uses batched training for all 4 samples per epoch.
*/

namespace {

double evaluate_average_loss(nrt::Sequential& model, const std::shared_ptr<nrt::Tensor>& inputs,
                             const std::shared_ptr<nrt::Tensor>& targets) {
    auto y_hat = model.forward(inputs);
    auto loss = nrt::mse(*y_hat, *targets);
    return loss;
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

    /**************************/
    /*        Training        */
    /**************************/
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
            double loss = evaluate_average_loss(model, inputs, targets);
            std::cout << "Loss (" << epoch << "/" << epochs << "):  " << loss << '\n';
        }
    }

    double final_loss = evaluate_average_loss(model, inputs, targets);
    std::cout << "Loss (" << epochs << "/" << epochs << "):  " << final_loss << '\n';

    return 0;
}
