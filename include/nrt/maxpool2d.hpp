#pragma once

#include <memory>
#include <vector>

#include "nrt/module.hpp"
#include "nrt/parameter.hpp"
#include "nrt/tensor.hpp"

namespace nrt {

class MaxPool2D : public Module {
public:
    MaxPool2D() = default;

    // Forward: {batch, channels, height, width} → {batch, channels, height/2, width/2}
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x) override;

    // No learnable parameters
    std::vector<Parameter> parameters() override {
        return {};
    }

private:
    static constexpr size_t pool_size_ = 2;  // 2×2 kernel
    static constexpr size_t stride_ = 2;     // stride of 2

    // Helper struct: position of max element in a 2×2 pool patch
    struct PoolInfo {
        size_t max_i;  // row index within patch (0 or 1)
        size_t max_j;  // col index within patch (0 or 1)
    };
};

}  // namespace nrt
