"""Rebuild all four tab3.gp tracks and the complete FEA arrangement."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"tests"/"tab3.gp"
SECONDS=178.77894736842083


def call(*parts):
    subprocess.run([sys.executable,*map(str,parts)],cwd=ROOT,check=True)


def main():
    lead=ROOT/"out"/"tab3_lead_guitar_fea.wav"
    rhythm=ROOT/"out"/"tab3_rhythm_guitar_fea.wav"
    bass=ROOT/"bass_fea"/"tab3_bass_fea.wav"
    drums=ROOT/"drum_set_fea"/"tab3_drums_fea.wav"
    call(ROOT/"tools"/"render_gp_score.py",SOURCE,"--track",0,"--seconds",SECONDS,
         "--delay-level",.55,"--out",lead)
    call(ROOT/"tools"/"render_gp_score.py",SOURCE,"--track",1,"--seconds",SECONDS,
         "--delay-level",0,"--out",rhythm)
    call(ROOT/"bass_fea"/"render_gp_bass.py",SOURCE,"--track",2,"--seconds",SECONDS,"--out",bass)
    call(ROOT/"drum_set_fea"/"render_gp_drums.py",SOURCE,"--track",3,"--seconds",SECONDS,"--out",drums)
    call(ROOT/"tools"/"mix_gp_full_band.py",
         "--lead",lead,"--rhythm",rhythm,"--bass",bass,"--drums",drums,
         "--drum-score",drums.with_name(drums.stem+"_score.json"),
         "--out",ROOT/"out"/"tab3_full_band_fea.wav",
         "--source",SOURCE,"--title","Zara Sa","--tempo",95)


if __name__=="__main__": main()
