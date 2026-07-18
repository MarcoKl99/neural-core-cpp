#include "nrt/maxpool2d.hpp"

#include <stdexcept>

#include "nrt/operations.hpp"

namespace nrt {

std::shared_ptr<Tensor> MaxPool2D::forward(std::shared_ptr<Tensor> x) {
    // Validate input: {batch, channels, height, width}
    if (x->rank() != 4) {
        throw std::invalid_argument("MaxPool2D::forward: input must be rank 4");
    }

    // Use autodiff operation for forward + backward tracking
    return maxpool2d_autodiff(x);
}

}  // namespace nrt
