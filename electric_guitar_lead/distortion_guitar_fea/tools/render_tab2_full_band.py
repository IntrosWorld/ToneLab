"""Rebuild every tab2 FEA stem and the complete four-track arrangement."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def call(*parts):
    subprocess.run([sys.executable, *map(str, parts)], cwd=ROOT, check=True)


def main():
    source = ROOT / "ref" / "tab2.gp"
    call(ROOT/"tools"/"render_gp_score.py", source, "--track", 0, "--seconds", 23.78404,
         "--out", ROOT/"out"/"tab2_lead_guitar_fea.wav")
    call(ROOT/"tools"/"render_gp_score.py", source, "--track", 1, "--seconds", 23.78404,
         "--out", ROOT/"out"/"tab2_background_guitar_fea.wav")
    call(ROOT/"bass_fea"/"render_tab2_bass.py")
    call(ROOT/"drum_set_fea"/"render_tab2_drums.py")
    call(ROOT/"tools"/"mix_tab2_full_band.py")


if __name__ == "__main__":
    main()
