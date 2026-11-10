"""Mix all four tab2 GP tracks from their independently rendered FEA stems."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

from mix_tab2_guitars import read_pcm24, write_pcm24

ROOT = Path(__file__).resolve().parents[1]
SR = 48_000


def pad(audio, length):
    return np.pad(audio, ((0, length-len(audio)), (0, 0)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lead", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--single-output", action="store_true")
    args = parser.parse_args()
    paths = {
        "lead": args.lead or ROOT / "out" / "tab2_lead_guitar_fea.wav",
        "rhythm": ROOT / "out" / "tab2_background_guitar_fea.wav",
        "bass": ROOT / "bass_fea" / "tab2_bass_fea.wav",
        "drums": ROOT / "drum_set_fea" / "tab2_drums_fea.wav",
    }
    stems = {name: read_pcm24(path) for name, path in paths.items()}
    length = max(map(len, stems.values()))
    lead, rhythm, bass, drums = (pad(stems[name], length) for name in ("lead", "rhythm", "bass", "drums"))

    # Lead remains focused; the supporting guitar is darker and wider.
    lead_mid = lead.mean(axis=1)
    lead_side = (lead[:, 0]-lead[:, 1])*.5
    lead = np.column_stack([lead_mid+.25*lead_side, lead_mid-.25*lead_side])
    rhythm_mid = rhythm.mean(axis=1)
    rhythm_side = (rhythm[:, 0]-rhythm[:, 1])*.5
    rhythm_mid = sosfilt(butter(2, 4300, "lowpass", fs=SR, output="sos"), rhythm_mid)
    rhythm = np.column_stack([.64*rhythm_mid+1.15*rhythm_side, .64*rhythm_mid-1.15*rhythm_side])

    # Keep fundamental energy mono-compatible and leave guitar midrange intact.
    bass_mono = sosfilt(butter(2, 3300, "lowpass", fs=SR, output="sos"), bass.mean(axis=1))
    bass = np.column_stack([bass_mono, bass_mono])
    drums = sosfilt(butter(2, 30, "highpass", fs=SR, output="sos"), drums, axis=0)
    drum_sub = sosfilt(butter(2, 145, "lowpass", fs=SR, output="sos"), drums, axis=0)
    drum_air = sosfilt(butter(2, 5200, "highpass", fs=SR, output="sos"), drums, axis=0)
    drums = drums + .23*drum_sub + .24*drum_air

    # Score-aware sidechain: make space for each physical kick in the bass and
    # each snare in the rhythm guitar without changing any musical timing.
    drum_score = json.loads((ROOT/"drum_set_fea"/"tab2_drum_score.json").read_text())
    bass_gain = np.ones(length)
    rhythm_gain = np.ones(length)
    for event in drum_score["events"]:
        start = int(event["start"]*SR)
        if event["piece"] == "kick":
            span = min(int(.105*SR), length-start)
            release = .72 + .28*(1-np.exp(-np.arange(span)/(SR*.050)))
            bass_gain[start:start+span] = np.minimum(bass_gain[start:start+span], release)
        elif event["piece"] == "snare":
            span = min(int(.075*SR), length-start)
            release = .84 + .16*(1-np.exp(-np.arange(span)/(SR*.032)))
            rhythm_gain[start:start+span] = np.minimum(rhythm_gain[start:start+span], release)
    bass *= bass_gain[:, None]
    rhythm *= rhythm_gain[:, None]

    mix = .71*lead + .34*rhythm + .56*bass + .74*drums
    mix = np.tanh(1.12*mix)/np.tanh(1.12)
    mix -= mix.mean(axis=0)
    mix *= .94/max(np.max(np.abs(mix)), 1e-9)
    output = args.out or ROOT / "out" / "tab2_full_band_fea.wav"
    write_pcm24(output, mix)
    enhanced_output = ROOT / "out" / "tab2_full_band_fea_enhanced.wav"
    no_bell_output = ROOT / "out" / "tab2_full_band_fea_no_bell.wav"
    if not args.single_output:
        write_pcm24(enhanced_output, mix)
        write_pcm24(no_bell_output, mix)

    manifest = {
        "source": "ref/tab2.gp", "tempo_bpm": 163,
        "included_tracks": [
            {"index": 0, "name": "Lead Guitar", "stem": str(paths["lead"])},
            {"index": 1, "name": "Rhythm Guitar", "stem": str(paths["rhythm"])},
            {"index": 2, "name": "Electric Bass (finger)", "stem": str(paths["bass"]), "events": 41},
            {"index": 3, "name": "Drums", "stem": str(paths["drums"]), "events": 103},
        ],
        "duration_s": length/SR, "peak": float(np.max(np.abs(mix))),
        "rms": float(np.sqrt(np.mean(mix*mix))), "output": str(output),
        "enhanced_output": None if args.single_output else str(enhanced_output),
        "no_bell_output": None if args.single_output else str(no_bell_output),
        "mix_features": ["kick-to-bass sidechain", "snare-to-rhythm sidechain",
                         "parallel drum sub/air bands", "stereo bus saturation"],
    }
    manifest_path = output.with_name(output.stem + "_manifest.json") if args.single_output else ROOT/"out"/"tab2_full_band_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
