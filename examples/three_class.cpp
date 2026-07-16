#include <iostream>
#include <memory>
#include <vector>

#include "nrt/activations.hpp"
#include "nrt/linear.hpp"
#include "nrt/loss.hpp"
#include "nrt/module.hpp"
#include "nrt/optimizer.hpp"
#include "nrt/sequential.hpp"
#include "nrt/tensor.hpp"

/*
Toy 3-class classification: three well-separated 2D clusters, classified via
Sequential(Linear(2,8,He) -> ReLU -> Linear(8,3,Xavier)) feeding raw logits into
CrossEntropyLoss (softmax applied internally - see include/nrt/loss.hpp).

Demonstrates:
- Batched training (all 12 samples in one forward/backward)
- Multi-class classification with CrossEntropyLoss
- Softmax + negative log-likelihood fused in the loss
*/

namespace {

// Predicted class = index of the largest logit (softmax is monotonic)
size_t argmax(const nrt::Tensor& logits_row) {
    // logits_row is shape {1, num_classes}, extract first (and only) sample
    size_t best = 0;
    for (size_t i = 1; i < logits_row.shape()[1]; ++i) {
        if (logits_row(0, i) > logits_row(0, best)) best = i;
    }
    return best;
}

double evaluate_accuracy(nrt::Sequential& model, const std::shared_ptr<nrt::Tensor>& inputs,
                         const std::shared_ptr<nrt::Tensor>& targets) {
    auto logits = model.forward(inputs);  // {batch, num_classes}

    size_t correct = 0;
    for (size_t i = 0; i < inputs->shape()[0]; ++i) {
        // Find argmax for sample i
        size_t best = 0;
        for (size_t c = 1; c < logits->shape()[1]; ++c) {
            if ((*logits)(i, c) > (*logits)(i, best)) best = c;
        }
        size_t true_class = static_cast<size_t>((*targets)(i, 0));
        if (best == true_class) ++correct;
    }
    return static_cast<double>(correct) / static_cast<double>(inputs->shape()[0]);
}

double evaluate_average_loss(nrt::Sequential& model, nrt::CrossEntropyLoss& loss_fn,
                             const std::shared_ptr<nrt::Tensor>& inputs,
                             const std::shared_ptr<nrt::Tensor>& targets) {
    auto logits = model.forward(inputs);           // {batch, num_classes}
    auto loss = loss_fn.forward(logits, targets);  // targets {batch, 1}
    return (*loss)(0, 0);                          // Scalar loss, already averaged over batch
}

}  // namespace

int main() {
    // Three well-separated clusters in 2D, 4 points each
    // Create batch tensor: {12, 2}
    auto inputs = std::make_shared<nrt::Tensor>(std::vector<size_t>{12, 2});

    // Class 0, centered near (-2, -2)
    (*inputs)(0, 0) = -2.0;
    (*inputs)(0, 1) = -2.0;
    (*inputs)(1, 0) = -2.5;
    (*inputs)(1, 1) = -1.5;
    (*inputs)(2, 0) = -1.5;
    (*inputs)(2, 1) = -2.5;
    (*inputs)(3, 0) = -2.0;
    (*inputs)(3, 1) = -1.0;

    // Class 1, centered near (2, -2)
    (*inputs)(4, 0) = 2.0;
    (*inputs)(4, 1) = -2.0;
    (*inputs)(5, 0) = 2.5;
    (*inputs)(5, 1) = -1.5;
    (*inputs)(6, 0) = 1.5;
    (*inputs)(6, 1) = -2.5;
    (*inputs)(7, 0) = 2.0;
    (*inputs)(7, 1) = -1.0;

    // Class 2, centered near (0, 2)
    (*inputs)(8, 0) = 0.0;
    (*inputs)(8, 1) = 2.0;
    (*inputs)(9, 0) = 0.5;
    (*inputs)(9, 1) = 2.5;
    (*inputs)(10, 0) = -0.5;
    (*inputs)(10, 1) = 2.5;
    (*inputs)(11, 0) = 0.0;
    (*inputs)(11, 1) = 1.0;

    // Target classes: {12, 1}
    auto targets = std::make_shared<nrt::Tensor>(std::vector<size_t>{12, 1});
    for (size_t i = 0; i < 4; ++i) (*targets)(i, 0) = 0.0;   // Class 0
    for (size_t i = 4; i < 8; ++i) (*targets)(i, 0) = 1.0;   // Class 1
    for (size_t i = 8; i < 12; ++i) (*targets)(i, 0) = 2.0;  // Class 2

    // Model: 2 -> 8 (ReLU, He) -> 3 (raw logits, Xavier). No final activation -
    // CrossEntropyLoss consumes logits directly and applies softmax internally.
    std::vector<std::unique_ptr<nrt::Module>> modules;
    modules.push_back(std::make_unique<nrt::Linear>(2, 8, nrt::WeightInit::He));
    modules.push_back(std::make_unique<nrt::ReLU>());
    modules.push_back(std::make_unique<nrt::Linear>(8, 3, nrt::WeightInit::Xavier));
    nrt::Sequential model(std::move(modules));

    nrt::CrossEntropyLoss loss_fn;
    const int epochs = 200;
    const double learning_rate = 0.01;
    nrt::SGD optimizer(model.parameters(), learning_rate);

    std::cout << "Initial loss:     " << evaluate_average_loss(model, loss_fn, inputs, targets)
              << '\n';
    std::cout << "Initial accuracy: " << evaluate_accuracy(model, inputs, targets) << '\n';

    for (int epoch = 0; epoch < epochs; ++epoch) {
        optimizer.zero_grad();

        // Single batched forward/backward
        auto logits = model.forward(inputs);
        auto loss = loss_fn.forward(logits, targets);
        loss->backward();

        optimizer.step();

        if ((epoch % 10) == 0) {
            std::cout << "Epoch " << epoch << "/" << epochs
                      << " - loss: " << evaluate_average_loss(model, loss_fn, inputs, targets)
                      << " - accuracy: " << evaluate_accuracy(model, inputs, targets) << '\n';
        }
    }

    std::cout << "Final loss:       " << evaluate_average_loss(model, loss_fn, inputs, targets)
              << '\n';
    std::cout << "Final accuracy:   " << evaluate_accuracy(model, inputs, targets) << '\n';

    return 0;
}
