"""Render two selected GPIF tracks as a steel-string acoustic duet."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from gp_score import load_gp_score
from guitar_performance import prepare_performance_events as performance_events
from render_model import AcousticGuitar, write_wav
from acoustic_engine import (SPATIAL_BODY_GAIN, UNRESOLVED_BODY_GAIN,
                             UNRESOLVED_CROSSOVER_HZ)

ROOT = Path(__file__).resolve().parents[1]


def stereo_position(audio, pan, width=.38):
    mid = audio.mean(axis=1)
    side = .5 * (audio[:, 0] - audio[:, 1]) * width
    angle = (pan + 1.) * np.pi / 4.
    return np.column_stack(((mid + side) * np.cos(angle),
                            (mid - side) * np.sin(angle))) * np.sqrt(2.)


def exact_excerpt(audio, frames, rate):
    out = np.zeros((frames, 2), dtype=np.float64)
    span = min(frames, len(audio))
    out[:span] = audio[:span]
    # Cut mid-song at exactly the requested frame, with a 25 ms end fade.
    fade = min(frames, round(.025 * rate))
    if fade:
        out[-fade:] *= np.cos(np.linspace(0., np.pi / 2., fade))[:, None]**2
    return out


def metrics(audio):
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio**2)))
    return {"peak_dbfs": float(20. * np.log10(max(peak, 1e-15))),
            "rms_dbfs": float(20. * np.log10(max(rms, 1e-15))),
            "finite": bool(np.all(np.isfinite(audio))),
            "clipped_samples": int(np.sum(np.abs(audio) >= 1.)),
            "dc": np.mean(audio, axis=0).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("score", type=Path)
    parser.add_argument("--seconds", type=float, default=30.)
    parser.add_argument("--rate", type=int, default=48_000)
    parser.add_argument("--lead-track", type=int, default=0)
    parser.add_argument("--rhythm-track", type=int, default=1)
    parser.add_argument("--reference", type=Path, help="Analysis reference, recorded in manifest; never loaded as synthesis audio")
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "assamj_first30s_acoustic_duet.wav")
    args = parser.parse_args()
    if args.seconds <= 0 or args.lead_track == args.rhythm_track:
        parser.error("duration must be positive and the two tracks must differ")
    score = load_gp_score(args.score, (args.lead_track, args.rhythm_track), args.seconds)
    frames = round(args.seconds * args.rate)
    rendered = {}
    settings = {"lead": {"technique": "finger", "pan": .25, "gain": 2.24, "seed": 1946,
                         "source": "continuous"},
                "rhythm": {"technique": "finger", "pan": -.32, "gain": .90, "seed": 7251,
                           "source": "continuous"}}
    for role, track in zip(("lead", "rhythm"), score["tracks"]):
        setting = settings[role]
        events = performance_events(track["events"], role, setting["seed"])
        track["render_role"] = role
        track["render_instrument"] = "FEA steel-string acoustic guitar"
        track["performance_events"] = events
        guitar = AcousticGuitar(args.rate, setting["technique"], setting["seed"],
                                source=setting["source"])
        print(f"Rendering {role}: {track['name']}, {len(events)} attacks -> acoustic", flush=True)
        audio = guitar.render(events, tail=.35, normalize=False)
        audio = exact_excerpt(audio, frames, args.rate)
        rendered[role] = setting["gain"] * stereo_position(audio, setting["pan"])
    mix = rendered["lead"] + rendered["rhythm"]
    # One shared gain preserves dynamics and exact stem reconstruction.
    highest_peak = max(float(np.max(np.abs(x))) for x in (*rendered.values(), mix))
    gain = 10.**(-1.4 / 20.) / max(highest_peak, 1e-15)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    paths = {"duet": args.out,
             "lead": args.out.with_name(args.out.stem + "_lead.wav"),
             "rhythm": args.out.with_name(args.out.stem + "_rhythm.wav")}
    audio_outputs = {"duet": mix, **rendered}
    output_info = {}
    for role, audio in audio_outputs.items():
        audio *= gain
        info = metrics(audio)
        if not info["finite"] or info["clipped_samples"]:
            raise RuntimeError(f"invalid {role} waveform: {info}")
        write_wav(paths[role], audio, args.rate)
        output_info[role] = {"path": str(paths[role].resolve()), **info,
                             "sha256": hashlib.sha256(paths[role].read_bytes()).hexdigest()}
        print(f"{role}: {frames / args.rate:.6f}s, peak {info['peak_dbfs']:.2f}dBFS -> {paths[role]}", flush=True)
    fingerprints = {}
    for relative in ("tools/guitar_physics.py", "tools/acoustic_engine.py",
                     "tools/render_model.py", "tools/plucked_string.py", "tools/gp_score.py", "tools/render_gp_duet.py",
                     "tools/string_performance.py", "tools/guitar_performance.py",
                     "tools/soundboard_v3.py", "model/body_modes_v4.json"):
        fingerprints[relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    manifest = {"score": score, "sample_rate": args.rate, "frames": frames,
                "channels": 2, "sample_bits": 24, "settings": settings,
                "radiation_profile": {"spatial_body_gain": SPATIAL_BODY_GAIN,
                    "unresolved_gain": UNRESOLVED_BODY_GAIN,
                    "unresolved_crossover_hz": UNRESOLVED_CROSSOVER_HZ},
                "master_gain": gain, "source_sha256": hashlib.sha256(args.score.read_bytes()).hexdigest(),
                "model_sha256": fingerprints, "outputs": output_info,
                "notes": ["Only the selected lead and rhythm tracks are synthesized.",
                          "Both tracks use the improved steel-string acoustic model.",
                          "Pitch, score tempo, rests, tablature and ties are preserved.",
                          "Correlated phrase touch varies by at most 8 velocity units; chord sweeps span at most 50 ms.",
                          "Persistent string state carries the score's hammer-on and pull-offs between frets.",
                          "Geometry-derived spatial body radiation and passive bridge loss replace the dominant low/mid force bypass.",
                          "Final 25 ms fades to zero; stems use the same master gain and sum to the duet.",
                          "No samples, external soundfont, electric amp or distortion are used."]}
    if args.reference:
        manifest["analysis_reference"] = {"path": str(args.reference.resolve()),
            "sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
            "use": "Timbre/decay analysis only; no reference audio used in the synthesized waveform"}
    path = args.out.with_suffix(".json")
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest: {path}", flush=True)


if __name__ == "__main__":
    main()
