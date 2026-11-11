#!/usr/bin/env bash
# Build the experimental integer reference model. It is not RTL signoff.
# Static C++ runtimes avoid mixing different MinGW DLL versions on Windows.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/toolchain.sh"
cd "$ROOT"
mkdir -p out
extension=''
case "${OSTYPE:-}" in msys*|cygwin*|win*) extension='.exe';; esac
"$CXX" -O2 -std=c++20 -Wall -Wextra -D_GLIBCXX_ASSERTIONS \
  -static-libgcc -static-libstdc++ -o "out/fast_model$extension" tb/fast_model.cpp
printf 'Built out/fast_model%s (experimental; see validation/hardware_report.md)\n' "$extension"
