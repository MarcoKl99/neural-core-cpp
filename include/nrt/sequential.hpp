#pragma once

#include <memory>
#include <vector>

#include "nrt/module.hpp"
#include "nrt/parameter.hpp"
#include "nrt/tensor.hpp"

namespace nrt {

// Sequential container: applies modules in sequence
// Inspired by torch.nn.Sequential
class Sequential : public Module {
public:
    // Constructor: takes ownership of all modules via unique_ptr
    explicit Sequential(std::vector<std::unique_ptr<Module>> modules);

    // Forward pass: applies each module in sequence
    Tensor forward(const Tensor& x) override;

    // Collect all parameters from all contained modules
    std::vector<Parameter> parameters() override;

    // Add a module to the sequence (optional convenience method)
    void add(std::unique_ptr<Module> module);

    // Get module at given index (for accessing specific layers)
    Module* get(size_t index);

private:
    std::vector<std::unique_ptr<Module>> modules_;
};

}  // namespace nrt
