#!/usr/bin/env bash
# Shared discovery. Nothing is installed or copied outside this project.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CXX="${CXX:-g++}"
PYTHON="${PYTHON:-python}"
JOBS="${JOBS:-4}"
if ! command -v "$CXX" >/dev/null 2>&1 && [[ -x /c/Strawberry/c/bin/g++.exe ]]; then
  export PATH="/c/Strawberry/c/bin:$PATH"
fi
MAKE="${MAKE:-$(command -v make || command -v gmake || true)}"
if [[ -z "$MAKE" && -x /c/Strawberry/c/bin/gmake.exe ]]; then
  MAKE=/c/Strawberry/c/bin/gmake.exe
fi

find_verilator() {
  if [[ -z "${VERILATOR:-}" ]]; then
    VERILATOR="$(command -v verilator || true)"
  fi
  if [[ -z "${VERILATOR:-}" || -n "${OSS_CAD_SUITE:-}" ]]; then
    for suite in "${OSS_CAD_SUITE:-}" /c/oss-cad-suite /f/VLSI-tools/oss-cad-suite-20260527/oss-cad-suite; do
      if [[ -n "$suite" && -x "$suite/bin/verilator_bin.exe" ]]; then
        VERILATOR="$suite/bin/verilator_bin.exe"
        VERILATOR_ROOT="$suite/share/verilator"
        if command -v cygpath >/dev/null 2>&1; then
          VERILATOR_ROOT="$(cygpath -m "$VERILATOR_ROOT")"
        fi
        export VERILATOR_ROOT
        break
      fi
    done
  fi
  [[ -n "${VERILATOR:-}" ]] || { echo 'Verilator not found. Set VERILATOR or OSS_CAD_SUITE.' >&2; exit 1; }
  [[ -n "$MAKE" ]] || { echo 'GNU make not found. Set MAKE.' >&2; exit 1; }
  command -v "$CXX" >/dev/null || { echo 'C++ compiler not found. Set CXX.' >&2; exit 1; }
  command -v "$PYTHON" >/dev/null || { echo 'Python not found. Set PYTHON.' >&2; exit 1; }
}

build_generated() {
  local directory="$1" top="$2"
  "$MAKE" -C "$directory" -f "V${top}.mk" -j "$JOBS" \
    "PYTHON3=$PYTHON" "CXX=$CXX" "LINK=$CXX"
}
