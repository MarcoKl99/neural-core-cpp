#include "nrt/tensor.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace nrt {

Tensor::Tensor(std::vector<size_t> shape) : shape_(std::move(shape)) {
    if (shape_.empty()) {
        throw std::invalid_argument("Tensor: shape must have at least rank 1");
    }

    // Check if the given shape does not have a value of 0 in a dimension
    for (size_t dim : shape_) {
        if (dim == 0) {
            throw std::invalid_argument("Tensor: shape dimensions must be > 0");
        }
    }

    // Multiply the values of the dimensions in shape -> Get the number of total
    // elements
    size_t total =
        std::accumulate(shape_.begin(), shape_.end(), size_t{1}, std::multiplies<size_t>());

    // Assign 0.0 to all elements in the data vector
    data_.assign(total, 0.0);

    // Compute the strides
    compute_strides();
}

size_t Tensor::rank() const {
    return shape_.size();
}

const std::vector<size_t>& Tensor::shape() const {
    return shape_;
}

size_t Tensor::size() const {
    return data_.size();
}

void Tensor::print(std::size_t precision) const {
    // Define the number of decimal places to print for doubles
    std::cout << std::fixed << std::setprecision(precision);

    // Distinguish between 1D and 2D tensor
    if (rank() == 1) {
        // 1D Tensor
        for (auto el : data_) {
            std::cout << el << ' ';
        }
        std::cout << std::endl;
    } else if (rank() == 2) {
        // 2D Tensor
        for (std::size_t i = 0; i < shape()[0]; i++) {
            // Print every row
            for (std::size_t j = 0; j < shape()[1]; j++) {
                std::cout << data_[i * shape()[1] + j] << ' ';
            }
            std::cout << std::endl;
        }
    } else {
        // Cannot nicely display the values - fallback
        std::cout << "Tensor(";
        for (std::size_t i = 0; i < rank() - 1; ++i) {
            std::cout << shape_[i] << ",";
        }
        std::cout << shape_[rank() - 1] << ")" << '\n';
    }
}

Tensor& Tensor::operator+=(const Tensor& other) {
    // Compute what the broadcast shape would be
    std::vector<size_t> broadcast_shape = broadcast_shapes(shape_, other.shape_);

    // For in-place operations, the broadcast result must match our current shape
    if (broadcast_shape != shape_) {
        throw std::invalid_argument(
            "Tensor::operator+=: broadcast result shape does not match left operand");
    }

    // Compute broadcast strides for other
    std::vector<size_t> other_strides = broadcast_strides(other.shape_, other.strides_, shape_);

    // Add in-place
    for (size_t i = 0; i < size(); ++i) {
        // Convert flat index to coords
        std::vector<size_t> coords(shape_.size());
        size_t idx = i;
        for (int d = shape_.size() - 1; d >= 0; --d) {
            coords[d] = idx % shape_[d];
            idx /= shape_[d];
        }

        // Compute offset in other using broadcast strides
        size_t other_offset = 0;
        for (size_t d = 0; d < shape_.size(); ++d) {
            other_offset += coords[d] * other_strides[d];
        }

        data_[i] += other.data_[other_offset];
    }

    return *this;
}

Tensor Tensor::operator+(const Tensor& other) const {
    // Compute broadcasted shape
    std::vector<size_t> result_shape = broadcast_shapes(shape_, other.shape_);
    Tensor result(result_shape);

    // Compute broadcasted strides for both operands
    std::vector<size_t> this_strides = broadcast_strides(shape_, strides_, result_shape);
    std::vector<size_t> other_strides =
        broadcast_strides(other.shape_, other.strides_, result_shape);

    // Fill result by looping through all flat indices
    for (size_t i = 0; i < result.size(); ++i) {
        // Convert flat index to multi-dimensional coordinates
        std::vector<size_t> coords(result_shape.size());
        size_t idx = i;
        for (int d = result_shape.size() - 1; d >= 0; --d) {
            coords[d] = idx % result_shape[d];
            idx /= result_shape[d];
        }

        // Compute offsets in this and other using broadcast strides
        size_t this_offset = 0, other_offset = 0;
        for (size_t d = 0; d < result_shape.size(); ++d) {
            this_offset += coords[d] * this_strides[d];
            other_offset += coords[d] * other_strides[d];
        }

        result.data_[i] = data_[this_offset] + other.data_[other_offset];
    }

    return result;
}

Tensor& Tensor::operator-=(const Tensor& other) {
    // Compute what the broadcast shape would be
    std::vector<size_t> broadcast_shape = broadcast_shapes(shape_, other.shape_);

    // For in-place operations, the broadcast result must match our current shape
    if (broadcast_shape != shape_) {
        throw std::invalid_argument(
            "Tensor::operator-=: broadcast result shape does not match left operand");
    }

    // Compute broadcast strides for other
    std::vector<size_t> other_strides = broadcast_strides(other.shape_, other.strides_, shape_);

    // Subtract in-place
    for (size_t i = 0; i < size(); ++i) {
        // Convert flat index to coords
        std::vector<size_t> coords(shape_.size());
        size_t idx = i;
        for (int d = shape_.size() - 1; d >= 0; --d) {
            coords[d] = idx % shape_[d];
            idx /= shape_[d];
        }

        // Compute offset in other using broadcast strides
        size_t other_offset = 0;
        for (size_t d = 0; d < shape_.size(); ++d) {
            other_offset += coords[d] * other_strides[d];
        }

        data_[i] -= other.data_[other_offset];
    }

    return *this;
}

Tensor Tensor::operator-(const Tensor& other) const {
    // Compute broadcasted shape
    std::vector<size_t> result_shape = broadcast_shapes(shape_, other.shape_);
    Tensor result(result_shape);

    // Compute broadcasted strides for both operands
    std::vector<size_t> this_strides = broadcast_strides(shape_, strides_, result_shape);
    std::vector<size_t> other_strides =
        broadcast_strides(other.shape_, other.strides_, result_shape);

    // Fill result by looping through all flat indices
    for (size_t i = 0; i < result.size(); ++i) {
        // Convert flat index to multi-dimensional coordinates
        std::vector<size_t> coords(result_shape.size());
        size_t idx = i;
        for (int d = result_shape.size() - 1; d >= 0; --d) {
            coords[d] = idx % result_shape[d];
            idx /= result_shape[d];
        }

        // Compute offsets in this and other using broadcast strides
        size_t this_offset = 0, other_offset = 0;
        for (size_t d = 0; d < result_shape.size(); ++d) {
            this_offset += coords[d] * this_strides[d];
            other_offset += coords[d] * other_strides[d];
        }

        result.data_[i] = data_[this_offset] - other.data_[other_offset];
    }

    return result;
}

Tensor& Tensor::operator*=(double scalar) {
    for (double& val : data_) {
        val *= scalar;
    }
    return *this;
}

Tensor Tensor::operator*(double scalar) const {
    Tensor result = *this;
    result *= scalar;
    return result;
}

Tensor Tensor::hadamard(const Tensor& other) const {
    // Compute broadcasted shape
    std::vector<size_t> result_shape = broadcast_shapes(shape_, other.shape_);
    Tensor result(result_shape);

    // Compute broadcasted strides for both operands
    std::vector<size_t> this_strides = broadcast_strides(shape_, strides_, result_shape);
    std::vector<size_t> other_strides =
        broadcast_strides(other.shape_, other.strides_, result_shape);

    // Fill result by looping through all flat indices
    for (size_t i = 0; i < result.size(); ++i) {
        // Convert flat index to multi-dimensional coordinates
        std::vector<size_t> coords(result_shape.size());
        size_t idx = i;
        for (int d = result_shape.size() - 1; d >= 0; --d) {
            coords[d] = idx % result_shape[d];
            idx /= result_shape[d];
        }

        // Compute offsets in this and other using broadcast strides
        size_t this_offset = 0, other_offset = 0;
        for (size_t d = 0; d < result_shape.size(); ++d) {
            this_offset += coords[d] * this_strides[d];
            other_offset += coords[d] * other_strides[d];
        }

        result.data_[i] = data_[this_offset] * other.data_[other_offset];
    }

    return result;
}

Tensor Tensor::matmul(const Tensor& other) const {
    if (rank() != 2 || other.rank() != 2) {
        throw std::invalid_argument("Tensor::matmul: both tensors must be rank 2");
    }
    if (shape_[1] != other.shape_[0]) {
        throw std::invalid_argument("Tensor::matmul: inner dimensions do not match");
    }

    size_t m = shape_[0];
    size_t n = shape_[1];
    size_t p = other.shape_[1];

    Tensor result({m, p});

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < p; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += (*this)(i, k) * other(k, j);
            }
            result(i, j) = sum;
        }
    }

    return result;
}

Tensor Tensor::transpose() const {
    // Only defined for rank 2 here
    if (rank() != 2) {
        throw std::invalid_argument("Tensor::transpose: requires rank 2");
    }

    // Create the result Tensor
    size_t rows = shape_[0];
    size_t cols = shape_[1];
    Tensor result({cols, rows});

    // Full in the correct values
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(j, i) = (*this)(i, j);
        }
    }

    return result;
}

Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
    size_t new_total =
        std::accumulate(new_shape.begin(), new_shape.end(), size_t{1}, std::multiplies<size_t>());
    if (new_total != size()) {
        throw std::invalid_argument("Tensor::reshape: element count mismatch");
    }

    // Simply create the new tensor and copy the data vector, as only the offset computation in the
    // constructor changes
    Tensor result(new_shape);
    result.data_ = data_;
    return result;
}

// Helper function e.g. for the MSE-loss
double Tensor::sum() const {
    double total = 0.0;
    for (double val : data_) {
        total += val;
    }
    return total;
}

// Autograd methods
Tensor Tensor::gradient() {
    Tensor g(gradient_shape_);
    g.data_ = gradient_data_;
    return g;
}

void Tensor::backward() {
    if (!creator_node_) return;  // Leaf tensor, nothing to backward

    // Initialize gradient: for a scalar loss, gradient is 1.0
    gradient_data_ = std::vector<double>(data_.size(), 1.0);
    gradient_shape_ = shape_;

    // Start the backward traversal
    backward_impl(gradient());
}

void Tensor::backward_impl(const Tensor& grad_output) {
    if (!creator_node_) return;  // Leaf tensor, stop recursion

    // Call the backward function for this operation
    // This will compute gradients for input tensors and recurse on them
    creator_node_->backward_fn(*this, grad_output, creator_node_->inputs);
}

void Tensor::accumulate_gradient(const Tensor& grad) {
    if (gradient_data_.empty()) {
        gradient_data_ = grad.data_;
        gradient_shape_ = grad.shape_;
    } else {
        for (size_t i = 0; i < gradient_data_.size(); ++i) {
            gradient_data_[i] += grad.data_[i];
        }
    }
}

bool Tensor::is_leaf() {
    return !creator_node_.has_value();
}

void Tensor::zero_grad() {
    gradient_shape_ = shape_;
    gradient_data_.assign(data_.size(), 0.0);
}

void Tensor::compute_strides() {
    // Init all values with 1 as the last value is always 1
    // Note that this initialization produces the correct result for a 1D tensor,
    // even if the loop below is never entered
    strides_.assign(shape_.size(), 1);

    // Reverse loop to apply the general strides formula
    for (int d = static_cast<int>(shape_.size()) - 2; d >= 0; --d) {
        strides_[d] = strides_[d + 1] * shape_[d + 1];
    }
}

std::vector<size_t> Tensor::broadcast_shapes(const std::vector<size_t>& shape_a,
                                             const std::vector<size_t>& shape_b) {
    // Find max rank and pad both shapes with leading 1s to align from the right
    size_t max_rank = std::max(shape_a.size(), shape_b.size());

    std::vector<size_t> padded_a(max_rank, 1);
    std::vector<size_t> padded_b(max_rank, 1);

    // Copy original shapes to the right
    std::copy(shape_a.begin(), shape_a.end(), padded_a.begin() + (max_rank - shape_a.size()));
    std::copy(shape_b.begin(), shape_b.end(), padded_b.begin() + (max_rank - shape_b.size()));

    // Check compatibility and compute result
    std::vector<size_t> result(max_rank);
    for (size_t i = 0; i < max_rank; ++i) {
        size_t dim_a = padded_a[i];
        size_t dim_b = padded_b[i];

        if (dim_a == dim_b) {
            result[i] = dim_a;
        } else if (dim_a == 1) {
            result[i] = dim_b;
        } else if (dim_b == 1) {
            result[i] = dim_a;
        } else {
            throw std::invalid_argument("Tensor::broadcast_shapes: shapes not broadcastable");
        }
    }

    return result;
}

std::vector<size_t> Tensor::broadcast_strides(const std::vector<size_t>& original_shape,
                                              const std::vector<size_t>& original_strides,
                                              const std::vector<size_t>& target_shape) {
    int original_rank = original_shape.size();
    int target_rank = target_shape.size();
    int pad_amount = target_rank - original_rank;

    std::vector<size_t> result(target_rank);

    // Padded dimensions (leading 1s) get stride 0
    for (int i = 0; i < pad_amount; ++i) {
        result[i] = 0;
    }

    // Original dimensions: keep stride if size matches, else set to 0 (broadcast)
    for (int i = 0; i < original_rank; ++i) {
        int target_idx = pad_amount + i;
        if (original_shape[i] == target_shape[target_idx]) {
            result[target_idx] = original_strides[i];
        } else if (original_shape[i] == 1) {
            result[target_idx] = 0;
        } else {
            throw std::logic_error("Tensor::broadcast_strides: invalid broadcast state");
        }
    }

    return result;
}

}  // namespace nrt
