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
#include "nrt/conv2d.hpp"
#include "nrt/flatten.hpp"
#include "nrt/linear.hpp"
#include "nrt/loss.hpp"
#include "nrt/maxpool2d.hpp"
#include "nrt/module.hpp"
#include "nrt/optimizer.hpp"
#include "nrt/sequential.hpp"
#include "nrt/tensor.hpp"

/*
MNIST with a simple CNN: Conv2D(1,16,3) -> ReLU -> Flatten -> Linear(10816,10)
*/

namespace {

constexpr size_t kBatchSize = 32;
constexpr int kDefaultEpochs = 10;
constexpr double kLearningRate = 0.01;
constexpr unsigned int kSeed = 42;
constexpr size_t kDefaultTrainSubsetSize = 5000;
constexpr size_t kDefaultTestSubsetSize = 1000;

size_t parse_size_arg(int argc, char** argv, int index, size_t default_value) {
    if (argc <= index) return default_value;
    return static_cast<size_t>(std::stoul(argv[index]));
}

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

// Build batch from dataset: reshape {784,1} images to {batch,1,28,28}
std::pair<std::shared_ptr<nrt::Tensor>, std::shared_ptr<nrt::Tensor>> build_batch(
    const mnist::Dataset& dataset, const std::vector<size_t>& batch_indices) {
    size_t batch_size = batch_indices.size();
    auto images = std::make_shared<nrt::Tensor>(std::vector<size_t>{batch_size, 1, 28, 28});
    auto labels = std::make_shared<nrt::Tensor>(std::vector<size_t>{batch_size, 1});

    for (size_t i = 0; i < batch_size; ++i) {
        size_t idx = batch_indices[i];
        // Reshape {784, 1} → {1, 28, 28} for this sample
        for (size_t p = 0; p < 784; ++p) {
            size_t h = p / 28;
            size_t w = p % 28;
            (*images)(i, 0, h, w) = (*dataset.images[idx])(p, 0);
        }
        (*labels)(i, 0) = static_cast<double>(dataset.labels[idx]);
    }
    return {images, labels};
}

double evaluate_accuracy(nrt::Sequential& model, const mnist::Dataset& dataset) {
    std::vector<size_t> all_indices(dataset.images.size());
    std::iota(all_indices.begin(), all_indices.end(), 0);

    auto [images_batch, _] = build_batch(dataset, all_indices);
    auto logits_batch = model.forward(images_batch);

    size_t correct = 0;
    for (size_t i = 0; i < dataset.images.size(); ++i) {
        size_t best = 0;
        for (size_t c = 1; c < 10; ++c) {
            if ((*logits_batch)(i, c) > (*logits_batch)(i, best)) best = c;
        }
        if (best == dataset.labels[i]) ++correct;
    }
    return static_cast<double>(correct) / static_cast<double>(dataset.images.size());
}

double evaluate_average_loss(nrt::Sequential& model, nrt::CrossEntropyLoss& loss_fn,
                             const mnist::Dataset& dataset) {
    std::vector<size_t> all_indices(dataset.images.size());
    std::iota(all_indices.begin(), all_indices.end(), 0);

    auto [images_batch, labels_batch] = build_batch(dataset, all_indices);
    auto logits = model.forward(images_batch);
    auto loss = loss_fn.forward(logits, labels_batch);

    return (*loss)(0, 0);
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
    // Load MNIST data
    size_t train_subset_size;
    size_t test_subset_size;
    int epochs;
    try {
        train_subset_size = parse_size_arg(argc, argv, 1, kDefaultTrainSubsetSize);
        test_subset_size = parse_size_arg(argc, argv, 2, kDefaultTestSubsetSize);
        epochs = parse_size_arg(argc, argv, 3, kDefaultEpochs);
    } catch (const std::exception&) {
        std::cerr << "Usage: " << argv[0] << " [train_subset_size] [test_subset_size] [epochs]\n";
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

    // Model: Conv2D -> ReLU -> MaxPool2D -> Flatten -> Linear
    std::vector<std::unique_ptr<nrt::Module>> modules;
    modules.push_back(std::make_unique<nrt::Conv2D>(1, 16, 3, nrt::WeightInit::He));
    modules.push_back(std::make_unique<nrt::ReLU>());
    modules.push_back(std::make_unique<nrt::MaxPool2D>());  // 2 x 2 MaxPooling
    modules.push_back(std::make_unique<nrt::Flatten>());
    modules.push_back(std::make_unique<nrt::Linear>(16 * 13 * 13, 10, nrt::WeightInit::Xavier));
    nrt::Sequential model(std::move(modules));

    std::cout << "Number of parameters: " << model.parameter_count() << '\n';

    // Training Setup
    nrt::CrossEntropyLoss loss_fn;
    nrt::SGD optimizer(model.parameters(), kLearningRate);

    std::cout << "Calculating initial train loss and test accuracy..." << '\n';
    double init_train_loss = evaluate_average_loss(model, loss_fn, train_data);
    double init_test_accuracy = evaluate_accuracy(model, test_data);
    std::cout << "\nInitial train loss: " << init_train_loss << '\n';
    std::cout << "Initial test accuracy: " << init_test_accuracy << "\n\n";

    std::vector<size_t> indices(train_data.images.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(kSeed);

    // Training Loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), rng);

        size_t total_batches = (indices.size() + kBatchSize - 1) / kBatchSize;
        size_t batch_num = 0;
        auto epoch_start = std::chrono::steady_clock::now();

        // Iterate over all batches
        for (size_t batch_start = 0; batch_start < indices.size(); batch_start += kBatchSize) {
            std::vector<size_t> batch_indices;
            for (size_t i = batch_start; i < std::min(batch_start + kBatchSize, indices.size());
                 ++i) {
                batch_indices.push_back(indices[i]);
            }
            auto [images_batch, labels_batch] = build_batch(train_data, batch_indices);

            optimizer.zero_grad();
            auto logits = model.forward(images_batch);
            auto loss = loss_fn.forward(logits, labels_batch);
            loss->backward();
            optimizer.step();

            ++batch_num;
            print_progress_bar(batch_num, total_batches, epoch + 1, epochs, epoch_start);
        }
        std::cout << '\n';

        double train_loss = evaluate_average_loss(model, loss_fn, train_data);
        double test_accuracy = evaluate_accuracy(model, test_data);
        std::cout << "Epoch " << (epoch + 1) << "/" << epochs << " - train loss: " << train_loss
                  << " - test accuracy: " << test_accuracy << "\n\n";
    }

    double final_train_loss = evaluate_average_loss(model, loss_fn, train_data);
    double final_test_accuracy = evaluate_accuracy(model, test_data);
    std::cout << "Final train loss: " << final_train_loss << '\n';
    std::cout << "Final test accuracy: " << final_test_accuracy << '\n';

    return 0;
}
