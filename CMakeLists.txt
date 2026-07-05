cmake_minimum_required(VERSION 3.20)
project(neural_runtime_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Core Library
add_library(nrt_core
    src/tensor.cpp
)

# Add the include directory to use #include "nrt/..."
target_include_directories(nrt_core PUBLIC include)
add_library(nrt::core ALIAS nrt_core)

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
add_executable(nrt_tests tests/test_tensor.cpp)
target_link_libraries(nrt_tests PRIVATE nrt::core Catch2::Catch2WithMain)

include(Catch)
catch_discover_tests(nrt_tests)

# Optional local scratch file (gitignore)
if(EXISTS ${CMAKE_SOURCE_DIR}/dev.cpp)
    add_executable(dev dev.cpp)
    target_link_libraries(dev PRIVATE nrt::core)
endif()
