#!/usr/bin/env bash
# Directed RTL regression; fails on a cadence or event-control assertion.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/toolchain.sh"
find_verilator
cd "$ROOT"
mkdir -p out/build_control
"$VERILATOR" --cc --exe --main --timing --assert -Wno-fatal \
  -CFLAGS '-std=c++20 -O2' -LDFLAGS '-static-libgcc -static-libstdc++' \
  -Irtl rtl/grand_piano_pkg.sv rtl/grand_piano_phys.sv rtl/grand_piano_core.sv \
  tb/control_regression.sv --top-module control_regression \
  -o control_regression --Mdir out/build_control
build_generated out/build_control control_regression
if [[ -x out/build_control/control_regression.exe ]]; then
  ./out/build_control/control_regression.exe
else
  ./out/build_control/control_regression
fi
