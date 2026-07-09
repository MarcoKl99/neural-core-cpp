#include "nrt/sequential.hpp"

#include "nrt/parameter.hpp"

namespace nrt {

Sequential::Sequential(std::vector<std::unique_ptr<Module>> modules)
    : modules_(std::move(modules)) {}

Tensor Sequential::forward(const Tensor& x) {
    // Create the output in the first step as the input
    Tensor output = x;

    // Transform the output through all layers of the Sequential class
    for (auto& module : modules_) {
        output = module->forward(output);
    }
    return output;
}

std::vector<Parameter> Sequential::parameters() {
    // Create the container to store all parameters of the model
    std::vector<Parameter> all_params;

    // Iterate over all modules and insert their parameters
    for (auto& module : modules_) {
        auto params = module->parameters();
        all_params.insert(all_params.end(), params.begin(), params.end());
    }
    return all_params;
}

void Sequential::add(std::unique_ptr<Module> module) {
    modules_.push_back(std::move(module));
}

// Get module at given index (for accessing specific layers)
Module* Sequential::get(size_t index) {
    if (index >= modules_.size()) return nullptr;
    return modules_[index].get();
}

}  // namespace nrt
