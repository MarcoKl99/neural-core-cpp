#include <iostream>
#include <string>
#include <vector>

#include "cli_args.hpp"
#include "csv_logger.hpp"
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

int main(int argc, char** argv) {
    // Optional CLI args: [csv_path] and --log-interval <n>.
    // Defaults to 250 so a 5000-epoch run gives ~20 points.
    nrt_examples::ParsedArgs args = nrt_examples::parse_args(argc, argv, /*default_log_interval=*/250);
    const std::string csv_path = args.positional.empty() ? "xor_deep.csv" : args.positional[0];
    const int log_interval = args.log_interval;

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

    // Log the training curve: epoch vs. MSE loss.
    nrt_examples::CsvLogger csv(csv_path, "epoch,loss");
    std::cout << "Writing training curve to: " << csv_path << '\n';

    for (int epoch = 0; epoch < epochs; ++epoch) {
        optimizer.zero_grad();

        // Single batched forward/backward
        auto y_hat = model.forward(inputs);
        auto loss = loss_fn.forward(y_hat, targets);
        loss->backward();

        optimizer.step();

        // Log to the CSV every `log_interval` epochs (plus the final epoch) - independent of
        // the console print below.
        if ((epoch % log_interval == 0) || (epoch == epochs - 1)) {
            csv.row({static_cast<double>(epoch), evaluate_average_loss(model, inputs, targets)});
        }

        // Console progress: every 1000 epochs.
        if ((epoch % 1000) == 0) {
            double avg_loss = evaluate_average_loss(model, inputs, targets);
            std::cout << "Loss (" << epoch << "/" << epochs << "):  " << avg_loss << '\n';
        }
    }

    double final_loss = evaluate_average_loss(model, inputs, targets);
    std::cout << "Loss (" << epochs << "/" << epochs << "):  " << final_loss << '\n';

    return 0;
}
