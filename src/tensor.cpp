#include "nrt/tensor.hpp"

#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace nrt {

Tensor::Tensor(std::vector<size_t> shape) : shape_(std::move(shape)) {
    // Check if the shape is correct - only 1D and 2D for now
    if (shape_.size() != 1 && shape_.size() != 2) {
        throw std::invalid_argument("Tensor: shape must have rank 1 or 2");
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

double& Tensor::operator()(size_t i) {
    // This operator overload is only for Tensors of rank 1
    if (rank() != 1) {
        throw std::invalid_argument("Tensor: 1-argument access requires rank 1");
    }

    // If an element out of ranged is accessed
    if (i >= shape_[0]) {
        throw std::out_of_range("Tensor: index out of range");
    }

    // Simply return the value at the index, as the tensor is 1D
    return data_[i];
}

// Same as for the operator overload above (non-const version)
double Tensor::operator()(size_t i) const {
    if (rank() != 1) {
        throw std::invalid_argument("Tensor: 1-argument access requires rank 1");
    }
    if (i >= shape_[0]) {
        throw std::out_of_range("Tensor: index out of range");
    }
    return data_[i];
}

double& Tensor::operator()(size_t i, size_t j) {
    if (rank() != 2) {
        throw std::invalid_argument("Tensor: 2-argument access requires rank 2");
    }
    if (i >= shape_[0] || j >= shape_[1]) {
        throw std::out_of_range("Tensor: index out of range");
    }

    // Access the element based on the row-major pattern
    // -> Flat representation of the Tensor is an array that can be accessed
    // like seen below
    return data_[i * shape_[1] + j];
}

// Same as for the operator overload above (non-const version)
double Tensor::operator()(size_t i, size_t j) const {
    if (rank() != 2) {
        throw std::invalid_argument("Tensor: 2-argument access requires rank 2");
    }
    if (i >= shape_[0] || j >= shape_[1]) {
        throw std::out_of_range("Tensor: index out of range");
    }
    return data_[i * shape_[1] + j];
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
        // Error
        throw std::logic_error("unreachable - Hmm... we should not be here...");
    }
}

Tensor& Tensor::operator+=(const Tensor& other) {
    // The shapes must be identical
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor::operator+=: shape mismatch");
    }

    // Perform the summation
    for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] += other.data_[i];
    }

    return *this;
}

Tensor Tensor::operator+(const Tensor& other) const {
    // Delegate the logic to the operator+= and re-use it here

    // Copy the Tensor
    Tensor result = *this;

    // Add the other tensor using the operator+=
    result += other;

    // Return the new created object
    return result;
}

Tensor& Tensor::operator-=(const Tensor& other) {
    // The shapes must be identical
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor::operator+=: shape mismatch");
    }

    // Perform the subtraction
    for (size_t i = 0; i < data_.size(); ++i) {
        data_[i] -= other.data_[i];
    }

    return *this;
}

Tensor Tensor::operator-(const Tensor& other) const {
    // Delegate the logic to the operator-= and re-use it here

    // Copy the Tensor
    Tensor result = *this;

    // Add the other tensor using the operator-=
    result -= other;

    // Return the new created object
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
    // The shapes must be equal here
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor::hadamard: shape mismatch");
    }

    // Create a new Tensor instance - no in-place method required for now
    Tensor result = *this;
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] *= other.data_[i];
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
    creator_node_->backward_fn(*this, grad_output);
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

}  // namespace nrt
