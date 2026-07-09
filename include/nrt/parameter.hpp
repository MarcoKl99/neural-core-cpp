#pragma once

#include "nrt/tensor.hpp"

namespace nrt {

// A learnable parameter: holds pointers to its value and accumulated gradient
struct Parameter {
    Tensor* value;     // The actual parameter tensor (weights, bias, etc.)
    Tensor* gradient;  // Its accumulated gradient (dL/dparam)
};

}  // namespace nrt
