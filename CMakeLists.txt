cmake_minimum_required(VERSION 3.20)
project(neural_runtime_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Core Library
add_library(nrt_core
    src/tensor.cpp
    src/linear.cpp
    src/activations.cpp
    src/loss.cpp
    src/optimizer.cpp
    src/sequential.cpp
    src/operations.cpp
)

# Add the include directory to use #include "nrt/..."
target_include_directories(nrt_core PUBLIC include)
add_library(nrt::core ALIAS nrt_core)

option(NRT_ENABLE_COVERAGE "Enable coverage instrumentation (gcov/lcov)" OFF)
if(NRT_ENABLE_COVERAGE)
    target_compile_options(nrt_core PRIVATE -O0 -g --coverage)
    target_link_options(nrt_core PRIVATE --coverage)
endif()

# Tests (Catch2)
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.6.0
)
FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

enable_testing()
add_executable(
    nrt_tests
    tests/test_tensor.cpp
    tests/test_linear.cpp
    tests/test_activations.cpp
    tests/test_loss.cpp
    tests/test_optimizer.cpp
    tests/test_operations.cpp
    tests/test_integration.cpp
)
target_link_libraries(nrt_tests PRIVATE nrt::core Catch2::Catch2WithMain)

if(NRT_ENABLE_COVERAGE)
    target_compile_options(nrt_tests PRIVATE -O0 -g --coverage)
    target_link_options(nrt_tests PRIVATE --coverage)
endif()

include(Catch)
catch_discover_tests(nrt_tests)

# Examples
add_executable(xor_forward
    examples/xor_forward.cpp
)
target_link_libraries(xor_forward PRIVATE nrt::core)

add_executable(xor_training
    examples/xor_training.cpp
)
target_link_libraries(xor_training PRIVATE nrt::core)

add_executable(xor_deep
    examples/xor_deep.cpp
)
target_link_libraries(xor_deep PRIVATE nrt::core)

add_executable(three_class
    examples/three_class.cpp
)
target_link_libraries(three_class PRIVATE nrt::core)

add_executable(mnist_mlp
    examples/mnist_mlp.cpp
    examples/mnist_loader.cpp
)
target_link_libraries(mnist_mlp PRIVATE nrt::core)


# Optional local scratch file (gitignore)
if(EXISTS ${CMAKE_SOURCE_DIR}/dev.cpp)
    add_executable(dev dev.cpp examples/mnist_loader.cpp)
    target_link_libraries(dev PRIVATE nrt::core)
endif()
