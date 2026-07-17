// include/nrt/tensor.hpp
#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

#include "nrt/computation_node.hpp"

namespace nrt {

class Tensor {
public:
    // Broadcasting helpers for elementwise operations
    static std::vector<size_t> broadcast_shapes(const std::vector<size_t>& shape_a,
                                                const std::vector<size_t>& shape_b);

    static std::vector<size_t> broadcast_strides(const std::vector<size_t>& original_shape,
                                                 const std::vector<size_t>& original_strides,
                                                 const std::vector<size_t>& target_shape);

    // Constructor with given shape - init all values to 0.0
    explicit Tensor(std::vector<size_t> shape);

    size_t rank() const;
    const std::vector<size_t>& shape() const;
    size_t size() const;  // Total number of elements in the tensor

    // Access element by vector of indices (works for any rank)
    double at(const std::vector<size_t>& indices) const;

    // Set element by vector of indices (works for any rank)
    void set_at(const std::vector<size_t>& indices, double value);

    /****************************/
    /*    Operator Overloads    */
    /****************************/

    // Variadic access operator for Tensors of arbitrary shape
    template <typename... Args>
    double& operator()(Args... indices);

    template <typename... Args>
    double operator()(Args... indices) const;

    // Utility function: print
    void print(std::size_t precision = 5) const;

    /****************************/
    /*  Mathematical operators  */
    /****************************/

    // Addition
    Tensor& operator+=(const Tensor& other);
    Tensor operator+(const Tensor& other) const;

    // Subtraction
    Tensor& operator-=(const Tensor& other);
    Tensor operator-(const Tensor& other) const;

    // Scalar multiplication
    Tensor& operator*=(double scalar);
    Tensor operator*(double scalar) const;

    // Hadamard product
    Tensor hadamard(const Tensor& other) const;

    // Matrix multiplication
    Tensor matmul(const Tensor& other) const;

    // Transpose
    Tensor transpose() const;

    // Reshape
    Tensor reshape(const std::vector<size_t>& new_shape) const;

    // Helper function
    double sum() const;

    /**************/
    /*  Autograd  */
    /**************/

    /*
    General Idea:
        - We call backward() on a result - so on an edge of the comp-graph (e.g. the loss)
            -> backward() initializes the gradients to 1.0 (w.r.t. themselves)
            -> backward() calls backward_impl(grad) on itself
        - backward_impl(grad) calls backward_fn on its creator node
        - backward_fn calculates the correct gradients w.r.t. the operation, calles
          accumulate_gradient on the inputs and calls backward_impl(grad) on the inputs recursively
    */

    void backward();

    void accumulate_gradient(const Tensor& grad);

    void backward_impl(const Tensor& grad_ouput);

    Tensor gradient();

    bool is_leaf();

    // Computation node if not leaf - Public for now, maybe to refactor later
    std::optional<ComputationNode> creator_node_;

    // Reset the accumulated gradients on this Tensor instance
    void zero_grad();

    // Helper method to un-broadcast gradients for the backward pass (sum up gradients over
    // broadcasted dimensions)
    static Tensor unbroadcast_gradient(const Tensor& grad,
                                       const std::vector<size_t>& original_shape);

private:
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    std::vector<double> data_;

    // Autograd members
    std::vector<double> gradient_data_;   // Stores computed gradients
    std::vector<size_t> gradient_shape_;  // Shape of the gradient vector

    // Private helper
    void compute_strides();
    template <typename... Args>
    size_t flat_offset(Args... indices) const;

    // Sum over specified axes, removing those dimensions from the result shape
    // Parameter axes holds all dimensions to sum over and remove from the shape
    Tensor sum_over_axes(const std::vector<size_t>& axes) const;
};

// Note: Template method must be defined in the header, therefore it's here and not in the cpp file
template <typename... Args>
size_t Tensor::flat_offset(Args... indices) const {
    if (sizeof...(indices) != rank()) {
        throw std::invalid_argument("Tensor: number of indices must match rank");
    }

    // Expand the arguments into a container to iterate over it
    std::array<size_t, sizeof...(Args)> idx{static_cast<size_t>(indices)...};

    // Calculate the offset that we need to access the correct data
    size_t offset = 0;
    for (size_t d = 0; d < idx.size(); ++d) {
        // Check if the current idx is legal
        if (idx[d] >= shape_[d]) {
            throw std::out_of_range("Tensor: index out of range");
        }
        offset += idx[d] * strides_[d];
    }
    return offset;
}

template <typename... Args>
double& Tensor::operator()(Args... indices) {
    return data_[flat_offset(indices...)];
}

template <typename... Args>
double Tensor::operator()(Args... indices) const {
    return data_[flat_offset(indices...)];
}

}  // namespace nrt
