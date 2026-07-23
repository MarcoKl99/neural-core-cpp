#include <optional>

#include "nrt/linear.hpp"
#include "nrt/module.hpp"

namespace nrt {

class Conv2D : public Module {
public:
    Conv2D(size_t in_channels, size_t out_channels, size_t kernel_size,
           WeightInit init = WeightInit::He, std::optional<unsigned int> seed = std::nullopt);

    // Forward pass: {batch, in_channels, height, width} -> {batch, out_channels, out_height,
    // out_width}
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x) override;

    // Return learnable parameters
    std::vector<Parameter> parameters() override;

    // Getters for parameters
    Tensor& weights() {
        return *weights_;
    }
    Tensor& bias() {
        return *bias_;
    }

    void set_weights(const Tensor& w, const Tensor& b);

private:
    size_t in_channels_;
    size_t out_channels_;
    size_t kernel_size_;

    // Learnable parameters
    std::shared_ptr<Tensor> weights_;  // {out_channels, in_channels, kernel_size, kernel_size}
    std::shared_ptr<Tensor> bias_;     // {out_channels}
};

}  // namespace nrt
