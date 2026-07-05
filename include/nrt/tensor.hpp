// include/nrt/tensor.hpp
#pragma once

#include <cstddef>
#include <vector>

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

    // Hadamard product
    Tensor hadamard(const Tensor& other) const;

private:
    std::vector<size_t> shape_;
    std::vector<double> data_;
};

}  // namespace nrt
