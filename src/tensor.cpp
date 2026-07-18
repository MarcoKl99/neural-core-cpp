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
    std::cout << "Initializing gradient..." << '\n';
    gradient_data_ = std::vector<double>(data_.size(), 1.0);
    gradient_shape_ = shape_;

    // Start the backward traversal
    std::cout << "Calling backward_impl..." << '\n';
    backward_impl(gradient());
}

void Tensor::backward_impl(const Tensor& grad_output) {
    if (!creator_node_) return;  // Leaf tensor, stop recursion

    // Call the backward function for this operation
    // This will compute gradients for input tensors and recurse on them
    std::cout << "Calling backward_fn..." << '\n';
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

/***********************************/
/** Un-Broadcast Helper Functions **/
/***********************************/

// Compute what the shape becomes after summing over specified axes
static std::vector<size_t> compute_summed_shape(const std::vector<size_t>& original_shape,
                                                const std::vector<size_t>& axes) {
    std::vector<size_t> result_shape;
    for (size_t i = 0; i < original_shape.size(); ++i) {
        if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
            result_shape.push_back(original_shape[i]);
        }
    }
    if (result_shape.empty()) {
        result_shape = {1};  // Summing all dimensions gives scalar
    }
    return result_shape;
}

// Take coords in reduced space and expand to full space by inserting 0s at summed axes
static std::vector<size_t> expand_coordinates(const std::vector<size_t>& reduced_coords,
                                              const std::vector<size_t>& axes, size_t full_rank) {
    std::vector<size_t> full_coords(full_rank, 0);
    size_t reduced_idx = 0;
    for (size_t i = 0; i < full_rank; ++i) {
        if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
            // Index i is not being removed / summed over
            full_coords[i] = reduced_coords[reduced_idx++];
        }
    }

    // All summed dimensions keep the value 0
    return full_coords;
}

// Recursive helper: iterate over all combinations of summed axes and sum values
static double sum_over_axes_recursive(std::vector<size_t>& coords,
                                      const std::vector<size_t>& axes_to_sum, size_t axis_idx,
                                      const std::vector<size_t>& shape,
                                      const std::vector<size_t>& strides,
                                      const std::vector<double>& data) {
    /*
    coords: Coordinates of a visited data entry, expanded (e.g. {0, 0, 2} -> {0, 1, 2} -> {0, 2, 2})
    axes_to_sum: The subset of axes idx that we want to sum over, e.g. {1}
    axis_idx: Iterator index over the indices af axes to sum over (read that again...)
    shape: Shape of the tensor (member)
    strides: Strides of the tensor (member)
    data: Data of the tensor (member)
    */

    if (axis_idx == axes_to_sum.size()) {
        // Base case: all summed axes have been iterated, compute offset and return value
        size_t offset = 0;
        for (size_t d = 0; d < shape.size(); ++d) {
            offset += coords[d] * strides[d];
        }
        return data[offset];
    }

    // Recursive case: iterate over current axis
    double sum = 0.0;
    size_t axis = axes_to_sum[axis_idx];
    for (size_t coord = 0; coord < shape[axis]; ++coord) {
        coords[axis] = coord;
        sum += sum_over_axes_recursive(coords, axes_to_sum, axis_idx + 1, shape, strides, data);
    }
    return sum;
}

// Main helper function
Tensor Tensor::sum_over_axes(const std::vector<size_t>& axes) const {
    // Compute output shape
    std::vector<size_t> result_shape = compute_summed_shape(shape_, axes);
    Tensor result(result_shape);

    // For each output element, sum the corresponding input values
    for (size_t out_idx = 0; out_idx < result.size(); ++out_idx) {
        // Convert flat output index to reduced coordinates
        std::vector<size_t> reduced_coords(result_shape.size());
        size_t idx = out_idx;
        for (int d = result_shape.size() - 1; d >= 0; --d) {
            reduced_coords[d] = idx % result_shape[d];
            idx /= result_shape[d];
        }

        // Expand to full coordinates
        std::vector<size_t> full_coords = expand_coordinates(reduced_coords, axes, shape_.size());

        // Sum over all combinations of summed axes
        result.data_[out_idx] =
            sum_over_axes_recursive(full_coords, axes, 0, shape_, strides_, data_);
    }

    return result;
}

// Helper function to un-broadcast gradients during backward pass (sum over broadcasted dimensions)
Tensor Tensor::unbroadcast_gradient(const Tensor& grad, const std::vector<size_t>& original_shape) {
    // grad has been broadcast from original_shape to grad.shape()
    // We need to sum over the dimensions that were broadcast

    std::vector<size_t> grad_shape = grad.shape();
    int original_rank = original_shape.size();
    int grad_rank = grad_shape.size();
    int pad_amount = grad_rank - original_rank;

    // Pad original_shape with leading 1s to match grad's rank
    std::vector<size_t> padded_original(grad_rank, 1);
    std::copy(original_shape.begin(), original_shape.end(), padded_original.begin() + pad_amount);

    // Find which dimensions were broadcast (original size 1, grad size > 1)
    std::vector<size_t> axes_to_sum;
    for (size_t i = 0; i < grad_rank; ++i) {
        if (padded_original[i] == 1 && grad_shape[i] > 1) {
            // This dimension was broadcast
            axes_to_sum.push_back(i);
        } else if (padded_original[i] != grad_shape[i]) {
            // Shouldn't happen if broadcast_shapes was correct
            throw std::logic_error("Tensor::unbroadcast_gradient: shape mismatch");
        }
    }

    // Sum over all broadcast dimensions
    if (axes_to_sum.empty()) {
        // No broadcasting happened, return grad as-is
        return grad;
    }

    Tensor summed = grad.sum_over_axes(axes_to_sum);
    return summed.reshape(original_shape);
}

double Tensor::at(const std::vector<size_t>& indices) const {
    if (indices.size() != rank()) {
        throw std::invalid_argument("Tensor::at: index count mismatch");
    }

    // Compute flat offset using strides (same as variadic operator does internally)
    size_t offset = 0;
    for (size_t d = 0; d < indices.size(); ++d) {
        offset += indices[d] * strides_[d];
    }
    return data_[offset];
}

void Tensor::set_at(const std::vector<size_t>& indices, double value) {
    if (indices.size() != rank()) {
        throw std::invalid_argument("Tensor::set_at: index count mismatch");
    }

    size_t offset = 0;
    for (size_t d = 0; d < indices.size(); ++d) {
        offset += indices[d] * strides_[d];
    }
    data_[offset] = value;
}

}  // namespace nrt
