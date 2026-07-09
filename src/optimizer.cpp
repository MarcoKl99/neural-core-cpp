#include "nrt/optimizer.hpp"

#include "nrt/parameter.hpp"
#include "nrt/tensor.hpp"

namespace nrt {

nrt::SGD::SGD(const std::vector<Parameter>& params, double learning_rate)
    : params_(params), learning_rate_(learning_rate) {}

void nrt::SGD::zero_grad() {
    // Set fresh gradients for the tensors
    for (auto param : params_) {
        *param.gradient = Tensor(param.gradient->shape());
    }
}

void nrt::SGD::step() {
    // Update each parameter according to its gradient and the learning rate
    for (auto param : params_) {
        *param.value = *param.value - *param.gradient * learning_rate_;
    }
}

double nrt::SGD::learning_rate() const {
    return learning_rate_;
}

size_t nrt::SGD::num_params() const {
    return params_.size();
}

}  // namespace nrt
