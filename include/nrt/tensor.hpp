// include/nrt/tensor.hpp
#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "nrt/computation_node.hpp"

namespace nrt {

class Tensor {
public:
    // Constructor with given shape - init all values to 0.0
    explicit Tensor(std::vector<size_t> shape);

    size_t rank() const;
    const std::vector<size_t>& shape() const;
    size_t size() const;  // Total number of elements in the tensor

    // Note: In the following, 2 operator overloads are implemented
    //       The compiler chooses the correct one depending on if the respective
    //       tensor is const or not

    // Function call operator overload for 1D
    // Works with a reference to be able to change the value
    double& operator()(size_t i);
    double operator()(size_t i) const;

    // Function call operator overload for 2D
    // Works with a reference to be able to change the value
    double& operator()(size_t i, size_t j);
    double operator()(size_t i, size_t j) const;

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

    // Helper function
    double sum() const;

    /**************/
    /*  Autograd  */
    /**************/

    void backward();

    Tensor gradient();

    bool is_leaf();

    void accumulate_gradient(const Tensor& grad);

    // Helper method for backward traversal
    void backward_impl(const Tensor& grad_ouput);

    // Computation node if not leaf - Public for now, maybe to refactor later
    std::optional<ComputationNode> creator_node_;

private:
    std::vector<size_t> shape_;
    std::vector<double> data_;

    // Autograd members
    std::vector<double> gradient_data_;   // Stores computed gradients
    std::vector<size_t> gradient_shape_;  // Shape of the gradient vector

    // Those classes can call private methods and attributes
    // Here: Layers must attach backward functions to Tensors
    friend class Linear;
    friend class ReLU;
    friend class Sigmoid;
};

}  // namespace nrt
