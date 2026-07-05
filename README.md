# Neural Runtime C++ 🧠

A small Deep Learning runtime in modern C++, implemented from first principles - Tensor,
Matrixoperations, Backprop, Autograd and more! Step by step without ML dependencies in the core.

## Status 🔎

In active development.

First building block:

- Tensor-class (1D/2D, row-major, double precision)
- Initialization
- Element access
- Tests

Implemented operations:
- Elementwise addition (`operator+`, `operator+=`)
- Elementwise subtraction (`operator-`, `operator-=`)
- Hadamard product (`hadamard(...)`)
- Matrix multiplication (`matmul(...)`, rank 2 only)
- Transpose (`transpose()`, rank 2 only)
- Scalar multiplication (`operator*`, `operator*=`)

More to come! 🚀
