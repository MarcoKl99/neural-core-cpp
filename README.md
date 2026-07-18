# Neural Core C++ 🧠

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)<br>
[![CI](https://github.com/MarcoKl99/neural-core-cpp/actions/workflows/ci.yaml/badge.svg)](https://github.com/MarcoKl99/neural-core-cpp/actions/workflows/ci.yaml)

A C++20 project on neural networks, implementing the core functionalities of deep learning from first principles, without external ML libraries.

## Components 🧱

- Reverse-mode automatic differentiation
- N-dimensional tensor library with row-major strides and broadcasting
- Convolutional neural networks (Conv2D + MaxPool2D layers)
- Neural network modules, optimizers and loss functions
- Gradient checking and PyTorch validation
- End-to-end MNIST training pipeline with batched processing

## Motivation 🏃‍➡️

The goal of this project is to develop a deeper understanding of the architecture and implementation behind deep learning functionalities.

By implementing every component from scratch in C++, this project explores topics like

- Tensor computation
- Automatic differentiation
- Numerical stability
- Architecture

## Example: Convolutional Neural Networks 🏙️

To give an example, we can create convolutional architectures for image classification using just our own implementations!

```cpp
// A CNN for MNIST: Conv2D -> ReLU -> MaxPool2D -> Flatten -> Dense -> ReLU -> Dense
std::vector<std::unique_ptr<nrt::Module>> modules;
modules.push_back(std::make_unique<nrt::Conv2D>(1, 16, 3, nrt::WeightInit::He));
modules.push_back(std::make_unique<nrt::ReLU>());
modules.push_back(std::make_unique<nrt::MaxPool2D>());      // 2×2 pooling, stride 2
modules.push_back(std::make_unique<nrt::Flatten>());
modules.push_back(std::make_unique<nrt::Linear>(16*13*13, 128, nrt::WeightInit::He));
modules.push_back(std::make_unique<nrt::ReLU>());
modules.push_back(std::make_unique<nrt::Linear>(128, 10, nrt::WeightInit::Xavier));
nrt::Sequential model(std::move(modules));

nrt::CrossEntropyLoss loss_fn;
nrt::SGD              optimizer(model.parameters(), /*lr=*/0.01);

// Batched training (e.g., {32, 1, 28, 28} input)
optimizer.zero_grad();
auto logits = model.forward(images);          // all 32 samples at once
auto loss   = loss_fn.forward(logits, targets);
loss->backward();                             // gradients flow through entire CNN
optimizer.step();
```

---

## Features ✨

**Tensor** (`nrt/tensor.hpp`)
- N-dimensional (rank >= 1), row-major storage with stride-based indexing
- Double precision, bounds-checked variadic element access via `operator()(i, j, k, ...)`
- Arithmetic: `+`, `-`, `+=`, `-=`, scalar `*`, Hadamard product, `matmul`, `transpose`, `reshape`, `sum`
- NumPy-style broadcasting with automatic gradient un-broadcasting

**Autograd engine** (`nrt/computation_node.hpp`, `nrt/operations.hpp`)
- Define-by-run **computational graph** built during the forward pass
- Reverse-mode automatic differentiation via `Tensor::backward()`
- Graph edges use shared ownership (`std::shared_ptr`), so intermediate values stay alive for the backward pass
- Differentiable ops: `matmul_autodiff`, `add_autodiff`, `subtract_autodiff`, `scalar_mult_autodiff`, `reshape_autodiff`, `transpose_autodiff`, `conv2d_autodiff`, `maxpool2d_autodiff`
- Broadcasting-aware gradient computation: gradients automatically reduce over broadcasted dimensions
- Per-tensor gradient storage: `gradient()`, `accumulate_gradient()`, `zero_grad()`

**Modules** (`nrt/module.hpp`)
- `Module` base class (`forward` / `parameters`), mirroring `torch.nn.Module`
- `Linear` — affine transform `y = W·x + b` with **He / Xavier weight initialization** and optional seed for reproducibility
- `Conv2D` — 2D convolution (fixed 3×3 kernel, stride 1, no padding), with full backward pass
- `MaxPool2D` — 2×2 max pooling with stride 2, gradient routing only to max positions
- `Flatten` — reshape N-D tensors to 2D (`{batch, ...}` → `{batch, flattened}`)
- `ReLU`, `Sigmoid`, `Softmax` — activations as graph nodes, working with any rank tensor
- `Sequential` — chains modules and aggregates their parameters for batch processing

**Losses** (`nrt/loss.hpp`)
- `MSELoss` — mean-squared error as a differentiable graph node (`forward(y_hat, target)` → scalar tensor)
- `CrossEntropyLoss` — softmax + negative-log-likelihood fused into one graph node (`forward(logits, target_class)` → scalar tensor); takes **raw logits**, softmax is applied internally
- Free functions `mse` / `mse_derivative`, `cross_entropy` / `cross_entropy_grad` for graph-free use

**Optimizer** (`nrt/optimizer.hpp`)
- `SGD` — reads gradients straight from the parameter tensors (`step()`, `zero_grad()`)

**Tested & validated**
- Catch2 unit tests for every component
- **Integration tests** that exercise the whole system:
  - a **finite-difference gradient check** (analytic `backward()` vs. numerical gradients)
  - a training loop that provably minimizes a convex regression
  - a full `Sequential` MLP learning XOR
- Forward and backward numerically validated against a PyTorch reference

---

## Project layout 🗂️

| Path | Purpose |
|------|---------|
| `include/nrt/` | Public headers (tensor, autograd, modules, loss, optimizer) |
| `src/` | Implementations |
| `tests/` | Catch2 unit + integration tests (`test_*.cpp`) |
| `examples/` | Runnable demos (`xor_forward`, `xor_training`, `xor_deep`, `three_class`) |
| `notes/` | Write-ups, incl. the Dying-ReLU observation |
| `CMakeLists.txt` | Build configuration |

---

## Build & run 🛠️

Requires a C++20 compiler and CMake >= 3.20. Catch2 is fetched automatically.

```bash
# Configure & build
cmake -B build && cmake --build build

# Run the full test suite
cd build && ctest --output-on-failure
#   (or run the test binary directly: ./build/nrt_tests)

# Run the examples
./build/xor_training   # full training loop (autograd + SGD)
./build/xor_forward    # forward pass only
./build/xor_deep       # deeper XOR MLP (He/Xavier init)
./build/three_class    # 3-class classification (Softmax + CrossEntropyLoss)
./build/mnist_mlp      # MLP baseline on MNIST
./build/mnist_conv     # CNN (Conv2D + MaxPool2D + Dense) on MNIST — the star of the show! 🌟
```

---

## How the autograd works 🔬

Each differentiable operation returns a new tensor that stores a **computation node**: the inputs it was built from and a closure describing how to push a gradient back to them. Calling `backward()` on the final (scalar) output seeds a gradient of `1.0` and walks the graph in reverse, applying the chain rule at each node and accumulating gradients into the leaf tensors (your parameters). Because nodes co-own their inputs via `shared_ptr`, the whole graph stays alive from output back to the leaves for the duration of the backward pass.

```
x ──▶ Linear ──▶ ReLU ──▶ Linear ──▶ Sigmoid ──▶ MSELoss ──▶ loss (scalar)
                                                               │
                        gradients flow back through the graph  ▼
        dL/dW, dL/db accumulate into each Linear's parameters via backward()
```

---

## Roadmap 🚀

**Done:**
- ✅ N-dimensional tensor support with stride-based indexing
- ✅ Batched processing (PyTorch-style batch-first convention)
- ✅ Conv2D layer with full autodiff
- ✅ MaxPool2D with gradient routing
- ✅ Broadcasting with automatic gradient un-broadcasting
- ✅ Reshape, transpose, and flatten operations with autodiff tracking

**Next phase: System Design & Performance Optimization**

- Shifting from feature breadth to systems depth
- Understand the performance characteristics of deep learning frameworks through profiling, measurement, and systematic optimization
- Maintain correctness via comprehensive testing

---

> **⚠️ Work in progress.** This is a learning-focused project under active development. APIs may change between versions, and the feature set is intentionally growing step by step. Feedback and curiosity welcome! 🚧
