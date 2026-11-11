#!/usr/bin/env bash
# Rebuild both hardware simulators in fresh, project-local build directories.
# Overrides: OSS_CAD_SUITE, VERILATOR, VERILATOR_ROOT, CXX, PYTHON, MAKE, JOBS.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/toolchain.sh"
find_verilator
cd "$ROOT"
mkdir -p out/build_top out/build_strings

"$VERILATOR" --cc --exe -Wall -Wno-fatal --assert \
  -CFLAGS '-std=c++20 -O2' -LDFLAGS '-static-libgcc -static-libstdc++' \
  --top-module grand_piano_top -Irtl \
  rtl/grand_piano_pkg.sv rtl/grand_piano_phys.sv rtl/grand_piano_core.sv \
  rtl/grand_piano_fx.sv rtl/grand_piano_top.sv tb/sim_main.cpp \
  -o piano_sim --Mdir out/build_top
build_generated out/build_top grand_piano_top

"$VERILATOR" --cc --exe -Wall -Wno-fatal --assert \
  -CFLAGS '-std=c++20 -O2' -LDFLAGS '-static-libgcc -static-libstdc++' \
  --top-module grand_piano_core -Irtl \
  rtl/grand_piano_pkg.sv rtl/grand_piano_phys.sv rtl/grand_piano_core.sv \
  tb/sim_strings.cpp -o piano_strings --Mdir out/build_strings
build_generated out/build_strings grand_piano_core
printf 'Built simulators in out/build_top and out/build_strings\n'
