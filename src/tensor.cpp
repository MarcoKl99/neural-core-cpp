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

}  // namespace nrt
