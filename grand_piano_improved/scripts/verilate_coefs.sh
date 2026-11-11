#!/usr/bin/env bash
# Rebuild and verify elaboration-generated coefficients against Python.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/toolchain.sh"
find_verilator
cd "$ROOT"
mkdir -p out/coefs out/build_coefs
"$VERILATOR" --cc --exe --main --timing -Wno-fatal \
  -CFLAGS '-std=c++20 -O2' -LDFLAGS '-static-libgcc -static-libstdc++' \
  -Irtl rtl/grand_piano_pkg.sv rtl/grand_piano_phys.sv tb/coef_dump.sv \
  --top-module coef_dump -o coefdump --Mdir out/build_coefs
build_generated out/build_coefs coef_dump
if [[ -x out/build_coefs/coefdump.exe ]]; then
  ./out/build_coefs/coefdump.exe
else
  ./out/build_coefs/coefdump
fi
"$PYTHON" tools/check_rtl_coefs.py
