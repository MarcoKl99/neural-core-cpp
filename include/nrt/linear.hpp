#pragma once

#include <optional>

#include "nrt/module.hpp"
#include "nrt/parameter.hpp"
#include "nrt/tensor.hpp"

namespace nrt {

class Linear : public nrt::Module {
public:
    /*
    Computes an affine transformation y = Wx + b with weights W.shape = {out_features, in_features}
    and b.shape = {out_feautes, 1},
    -> y.shape = {out_features, 1}
    */

    Linear(size_t in_features, size_t out_features);
    virtual ~Linear() noexcept = default;

    // Implement the Module-methods
    Tensor forward(Tensor& x) override;
    std::vector<Parameter> parameters() override;

    // Overwrite randomly init weights for testing
    void set_weights(const Tensor& w, const Tensor& b);

    Tensor backward(const Tensor& grad_output);
    void zero_grad();

    // Note: Not const Tensor as a return type to enable std::move behavior by default
    Tensor average_grad_weights() const;
    Tensor average_grad_bias() const;
    size_t in_features() const;
    size_t out_features() const;

    // Return a const Tensor reference to avoid unneccesary copies at every call
    // (Tensor instances can be large)
    Tensor& weights();
    Tensor& bias();

private:
    size_t in_features_;
    size_t out_features_;
    Tensor weights_;
    Tensor bias_;
    std::optional<Tensor> last_input_;
    Tensor grad_weights_;
    Tensor grad_bias_;

    // Count how many times .backward(...) has been called for now (for averaged gradients)
    size_t accumulation_count_ = 0;
};

}  // namespace nrt
