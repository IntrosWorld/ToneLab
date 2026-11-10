"""Compare the RTL's self-generated coefficient tables against the Python model.

The RTL now builds its own tables at elaboration from `grand_piano_phys.sv`
rather than reading generated .hex files.  That is only worth anything if the
two agree, so this compares them entry by entry -- 69,496 numbers.

    scripts/verilate_coefs.sh      # build and run the dumper
    python tools/check_rtl_coefs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import hw_model as hw   # noqa: E402

DUMP = ROOT / "out" / "coefs"


def load(name):
    path = DUMP / f"{name}.txt"
    if not path.exists():
        sys.exit(f"missing {path} -- run scripts/verilate_coefs.sh first")
    return np.loadtxt(path, dtype=np.int64, ndmin=1)


def compare(name, want, got):
    want = np.asarray(want, dtype=np.int64)
    got = np.asarray(got, dtype=np.int64)
    if len(want) != len(got):
        print(f"  {name:<10} LENGTH {len(got)} vs {len(want)}")
        return False, len(want)
    bad = np.nonzero(want != got)[0]
    if not len(bad):
        print(f"  {name:<10} {len(want):>6} values, exact")
        return True, len(want)
    worst = int(np.abs(want[bad] - got[bad]).max())
    print(f"  {name:<10} {len(bad)} of {len(want)} differ, worst {worst} LSB")
    for i in bad[:5]:
        print(f"      [{i}] python {want[i]}  rtl {got[i]}")
    return False, len(want)


def main():
    rom = hw.build_rom()
    stereo = hw.stereo_rom()

    checks = [
        ("k", rom["k"]), ("loss", rom["loss"]),
        ("cphi", rom["cphi"]), ("sphi", rom["sphi"]),
        ("amp0", rom["amp"][0]), ("amp1", rom["amp"][1]),
        ("amp2", rom["amp"][2]), ("amp3", rom["amp"][3]),
        ("base", rom["base"]), ("count", rom["count"]),
        ("attack", rom["attack"]), ("damper", rom["damper"]),
        ("undamped", rom["undamped"]),
        ("delay_l", [s[0] for s in stereo]),
        ("delay_r", [s[1] for s in stereo]),
        ("pan_l", [s[2] for s in stereo]),
        ("pan_r", [s[3] for s in stereo]),
    ]

    body = hw.body_rom()
    for col, name in enumerate(("body_k", "body_loss", "body_gl", "body_gr")):
        checks.append((name, [r[col] for r in body]))

    symp = hw.sympathetic_rom()
    for col, name in enumerate(("symp_k", "symp_loss", "symp_gain",
                                "symp_pl", "symp_pr", "symp_und",
                                "symp_pitch")):
        checks.append((name, [r[col] for r in symp]))

    print("RTL elaboration-time tables vs the Python model:\n")
    ok, total = True, 0
    for name, want in checks:
        good, n = compare(name, want, load(name))
        ok &= good
        total += n

    print(f"\n{total} numbers compared")
    if ok:
        print("every entry identical -- the RTL needs no generated tables")
    else:
        print("MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
