#!/usr/bin/env bash
set -euo pipefail

# Usage: ./profiling/profile.sh <forward|backward>
# Example: ./profiling/profile.sh backward
#
# Builds conv2d_profile in a dedicated Release+debug-symbols build directory and
# records a Time Profiler trace with xctrace, ready to open directly in Instruments.

if [ $# -ne 1 ] || { [ "$1" != "forward" ] && [ "$1" != "backward" ]; }; then
    echo "Usage: $0 <forward|backward>" >&2
    exit 1
fi

MODE="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build-profile"
TRACE_DIR="${ROOT_DIR}/profiling/traces"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
TRACE_FILE="${TRACE_DIR}/${MODE}-${TIMESTAMP}.trace"

mkdir -p "${TRACE_DIR}"

echo "==> Configuring dedicated Release+debug-symbols build in ${BUILD_DIR}"
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-g

echo "==> Building conv2d_profile"
cmake --build "${BUILD_DIR}" --target conv2d_profile -j

echo "==> Recording Time Profiler trace -> ${TRACE_FILE}"
xcrun xctrace record \
    --template 'Time Profiler' \
    --output "${TRACE_FILE}" \
    --launch -- "${BUILD_DIR}/conv2d_profile" "${MODE}"

echo "==> Done. Open with: open \"${TRACE_FILE}\""
