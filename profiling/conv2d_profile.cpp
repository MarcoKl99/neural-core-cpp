#include <iostream>
#include <memory>
#include <string>

#include "nrt/conv2d.hpp"
#include "nrt/tensor.hpp"

namespace {

// Matches BM_Conv2D_Forward / BM_Conv2D_BackwardPass in benchmarks/bench_cnn.cpp,
// so this profiles the exact same workload shape.
constexpr size_t kBatch = 32;
constexpr size_t kInChannels = 1;
constexpr size_t kOutChannels = 16;
constexpr size_t kKernelSize = 3;
constexpr size_t kImageSize = 28;
constexpr int kIterations = 500;

void profile_forward() {
    auto input = std::make_shared<nrt::Tensor>(
        std::vector<size_t>{kBatch, kInChannels, kImageSize, kImageSize});
    input->randomize(42);

    nrt::Conv2D conv(kInChannels, kOutChannels, kKernelSize, nrt::WeightInit::He, 42);

    // Accumulate into `sink` and print it at the end - without this, a sufficiently
    // aggressive optimizer could see the loop's results are never used and delete it.
    double sink = 0.0;
    for (int i = 0; i < kIterations; ++i) {
        auto output = conv.forward(input);
        sink += output->sum();
    }

    std::cout << "forward sink=" << sink << '\n';
}

void profile_backward() {
    auto input = std::make_shared<nrt::Tensor>(
        std::vector<size_t>{kBatch, kInChannels, kImageSize, kImageSize});
    input->randomize(42);

    nrt::Conv2D conv(kInChannels, kOutChannels, kKernelSize, nrt::WeightInit::He, 42);

    // One-time, unmeasured setup - build the graph once. See note above on why reusing it
    // across iterations is valid.
    auto output = conv.forward(input);

    double sink = 0.0;
    for (int i = 0; i < kIterations; ++i) {
        conv.weights().zero_grad();
        conv.bias().zero_grad();

        output->backward();

        sink += conv.weights().gradient().sum() + conv.bias().gradient().sum();
    }

    std::cout << "backward sink=" << sink << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2 || (std::string(argv[1]) != "forward" && std::string(argv[1]) != "backward")) {
        std::cerr << "Usage: " << argv[0] << " <forward|backward>\n";
        return 1;
    }

    if (std::string(argv[1]) == "forward") {
        profile_forward();
    } else if (std::string(argv[1]) == "backward") {
        profile_backward();
    } else {
        throw std::invalid_argument("Argument must be forward or backward");
    }

    return 0;
}
