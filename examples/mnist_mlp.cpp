#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "mnist_loader.hpp"
#include "nrt/activations.hpp"
#include "nrt/linear.hpp"
#include "nrt/loss.hpp"
#include "nrt/module.hpp"
#include "nrt/optimizer.hpp"
#include "nrt/sequential.hpp"
#include "nrt/tensor.hpp"

/*
MNIST with a raw MLP: Linear(784,256,He) -> ReLU -> Linear(256,128,He) -> ReLU ->
Linear(128,10,Xavier) -> CrossEntropyLoss, trained with mini-batch SGD.

Runs on a SUBSET of the full 60k/10k MNIST split by default (kTrainSubsetSize /
kTestSubsetSize below) - Tensor::matmul is a plain unvectorized triple loop with
no batch dimension, so a full-dataset run is genuinely slow. Once this converges
on the subset, dial the sizes up (they clamp to the dataset's real size, so a
large value like 60000 safely means "all of it").
*/

namespace {

constexpr size_t kBatchSize = 32;
constexpr int kEpochs = 15;
constexpr double kLearningRate = 0.1;
constexpr unsigned int kSeed = 42;
constexpr size_t kDefaultTrainSubsetSize = 5000;
constexpr size_t kDefaultTestSubsetSize = 1000;

size_t parse_size_arg(int argc, char** argv, int index, size_t default_value) {
    if (argc <= index) return default_value;
    return static_cast<size_t>(std::stoul(argv[index]));
}

// Truncate a loaded dataset down to its first `size` samples (subset for fast iteration).
mnist::Dataset take_subset(mnist::Dataset dataset, size_t size, unsigned int seed) {
    size = std::min(size, dataset.images.size());

    std::vector<size_t> indices(dataset.images.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);

    mnist::Dataset subset;
    subset.images.reserve(size);
    subset.labels.reserve(size);
    for (size_t i = 0; i < size; ++i) {
        subset.images.push_back(dataset.images[indices[i]]);
        subset.labels.push_back(dataset.labels[indices[i]]);
    }
    return subset;
}

size_t argmax(const nrt::Tensor& logits) {
    size_t best = 0;
    for (size_t i = 1; i < logits.shape()[0]; ++i) {
        if (logits(i, 0) > logits(best, 0)) best = i;
    }
    return best;
}

double evaluate_accuracy(nrt::Sequential& model, const mnist::Dataset& dataset) {
    size_t correct = 0;
    for (size_t i = 0; i < dataset.images.size(); ++i) {
        auto logits = model.forward(dataset.images[i]);
        if (argmax(*logits) == dataset.labels[i]) ++correct;
    }
    return static_cast<double>(correct) / static_cast<double>(dataset.images.size());
}

double evaluate_average_loss(nrt::Sequential& model, nrt::CrossEntropyLoss& loss_fn,
                             const mnist::Dataset& dataset) {
    double total_loss = 0.0;
    for (size_t i = 0; i < dataset.images.size(); ++i) {
        auto logits = model.forward(dataset.images[i]);
        total_loss += nrt::cross_entropy(*logits, dataset.labels[i]);
    }
    return total_loss / static_cast<double>(dataset.images.size());
}

void print_progress_bar(size_t current, size_t total, int epoch, int total_epochs,
                        std::chrono::steady_clock::time_point epoch_start) {
    constexpr int bar_width = 30;
    double fraction = static_cast<double>(current) / static_cast<double>(total);
    int filled = static_cast<int>(fraction * bar_width);

    double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - epoch_start).count();
    double rate = elapsed > 0.0 ? current / elapsed : 0.0;     // batches per second
    double eta = rate > 0.0 ? (total - current) / rate : 0.0;  // seconds remaining

    std::cout << "\r\033[K";  // clear the current line before redrawing
    std::cout << "Epoch " << epoch << "/" << total_epochs << " [";
    for (int i = 0; i < bar_width; ++i) std::cout << (i < filled ? '#' : '-');
    std::cout << "] " << current << "/" << total << " (" << static_cast<int>(fraction * 100)
              << "%) " << std::fixed << std::setprecision(5) << rate << " batch/s, ETA "
              << static_cast<int>(eta) << "s" << std::flush;
}

}  // namespace

int main(int argc, char** argv) {
    size_t train_subset_size;
    size_t test_subset_size;
    try {
        train_subset_size = parse_size_arg(argc, argv, 1, kDefaultTrainSubsetSize);
        test_subset_size = parse_size_arg(argc, argv, 2, kDefaultTestSubsetSize);
    } catch (const std::exception&) {
        std::cerr << "Usage: " << argv[0] << " [train_subset_size] [test_subset_size]\n";
        return 1;
    }

    std::cout << "Loading MNIST...\n";
    auto train_data = take_subset(mnist::load("data/MNIST/raw/train-images-idx3-ubyte",
                                              "data/MNIST/raw/train-labels-idx1-ubyte"),
                                  train_subset_size, kSeed);
    auto test_data = take_subset(mnist::load("data/MNIST/raw/t10k-images-idx3-ubyte",
                                             "data/MNIST/raw/t10k-labels-idx1-ubyte"),
                                 test_subset_size, kSeed);

    std::cout << "Train subset: " << train_data.images.size()
              << " images, test subset: " << test_data.images.size() << " images\n";

    // Model: 784 -> 256 (ReLU, He) -> 128 (ReLU, He) -> 10 (raw logits, Xavier)
    std::vector<std::unique_ptr<nrt::Module>> modules;
    modules.push_back(std::make_unique<nrt::Linear>(784, 256, nrt::WeightInit::He, kSeed));
    modules.push_back(std::make_unique<nrt::ReLU>());
    modules.push_back(std::make_unique<nrt::Linear>(256, 128, nrt::WeightInit::He, kSeed + 1));
    modules.push_back(std::make_unique<nrt::ReLU>());
    modules.push_back(std::make_unique<nrt::Linear>(128, 10, nrt::WeightInit::Xavier, kSeed + 2));
    nrt::Sequential model(std::move(modules));

    nrt::CrossEntropyLoss loss_fn;
    nrt::SGD optimizer(model.parameters(), kLearningRate / static_cast<double>(kBatchSize));

    std::cout << "Calculating initial train loss and test accuracy..." << '\n';
    double init_train_loss = evaluate_average_loss(model, loss_fn, train_data);
    double init_test_accuracy = evaluate_accuracy(model, test_data);
    std::cout << "\nInitial train loss: " << init_train_loss << '\n';
    std::cout << "Initial test accuracy: " << init_test_accuracy << "\n\n";

    std::vector<size_t> indices(train_data.images.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(kSeed);

    for (int epoch = 0; epoch < kEpochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), rng);

        size_t total_batches = (indices.size() + kBatchSize - 1) / kBatchSize;
        size_t batch_num = 0;
        auto epoch_start = std::chrono::steady_clock::now();

        for (size_t batch_start = 0; batch_start < indices.size(); batch_start += kBatchSize) {
            size_t batch_end = std::min(batch_start + kBatchSize, indices.size());

            optimizer.zero_grad();
            for (size_t i = batch_start; i < batch_end; ++i) {
                size_t idx = indices[i];
                auto logits = model.forward(train_data.images[idx]);
                auto loss = loss_fn.forward(logits, train_data.labels[idx]);
                loss->backward();
            }
            optimizer.step();

            ++batch_num;
            print_progress_bar(batch_num, total_batches, epoch + 1, kEpochs, epoch_start);
        }
        std::cout << '\n';  // move past the progress bar before printing the epoch summary

        double train_loss = evaluate_average_loss(model, loss_fn, train_data);
        double test_accuracy = evaluate_accuracy(model, test_data);
        std::cout << "Epoch " << (epoch + 1) << "/" << kEpochs << " - train loss: " << train_loss
                  << " - test accuracy: " << test_accuracy << "\n\n";
    }

    double final_train_loss = evaluate_average_loss(model, loss_fn, train_data);
    double final_test_accuracy = evaluate_accuracy(model, test_data);
    std::cout << "Final train loss: " << final_train_loss << '\n';
    std::cout << "Final test accuracy: " << final_test_accuracy << '\n';

    return 0;
}
