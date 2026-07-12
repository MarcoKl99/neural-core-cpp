#!/usr/bin/env bash
set -euo pipefail

cmake -B build -DNRT_ENABLE_COVERAGE=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target nrt_tests
ctest --test-dir build --output-on-failure
gcovr -r . --object-directory build --exclude '.*_deps.*' --exclude '.*tests.*' --exclude '.*examples.*'
