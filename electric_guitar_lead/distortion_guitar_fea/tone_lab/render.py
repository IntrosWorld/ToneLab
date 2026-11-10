"""Rebuild the GP1 tone suite and short GP2 bass/kit diagnostics.

Run from the distortion project: python -m tone_lab.render
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from tone_lab import dsp
from tone_lab.score import ROOT, load_guitar
from tone_lab.guitar import CleanGuitar, PRESETS
from tone_lab.bass.model import FingerBass, BassNote, load_bass
from tone_lab.drums.model import AcousticKit, DrumHit, load_drums

HERE = Path(__file__).resolve().parent


def legacy_bass(notes, seconds):
    sys.path.insert(0, str(ROOT/"bass_fea"))
    import bass_model
    model = json.loads((ROOT/"bass_fea/bass_fea_modes.json").read_text())
    old = [bass_model.BassNote(n.start, n.duration, int(n.midi), n.velocity, n.string_index) for n in notes]
    return bass_model.render_bass(model, old, seconds)


def legacy_drums(hits, seconds):
    sys.path.insert(0, str(ROOT/"drum_set_fea"))
    import drum_model
    model = json.loads((ROOT/"drum_set_fea/drum_set_fea_modes.json").read_text())
    old = [drum_model.DrumHit(h.start, h.piece, h.velocity, h.open_amount) for h in hits]
    return drum_model.render_drums(model, old, seconds)


def legacy_clean(events, guitar, seconds):
    from render_solo import LeadGuitar
    original = LeadGuitar()
    out = np.zeros(round(seconds*dsp.SR))
    ordered = sorted(events, key=lambda e: e.start)
    for i, event in enumerate(ordered):
        note = original.gesture(event)
        following = next((e for e in ordered[i+1:] if e.string_index == event.string_index and e.start > event.start), None)
        if following:
            cut = round((following.start-event.start)*dsp.SR)
            if cut < len(note):
                note[cut:] *= np.exp(-np.log(1000)*np.arange(len(note)-cut)/(.006*dsp.SR))
        start = round(event.start*dsp.SR)
        span = min(len(note), len(out)-start)
        if span > 0:
            out[start:start+span] += (.17+.83*event.velocity/127)**1.3*note[:span]
    return guitar.process(out, "clean_neck")


def drum_diagnostic():
    """Each piece plays soft/medium/hard; final bar demonstrates open-hat choking."""
    hits = []
    t = .25
    for piece in ("kick", "snare", "hihat", "tom_high", "tom_mid", "tom_low", "tom_floor", "crash", "ride"):
        for velocity in (48, 80, 116):
            hits.append(DrumHit(t, piece, velocity, .5 if piece == "hihat" else 0.))
            t += .62 if piece not in {"crash", "ride"} else 1.1
        t += .3
    hits.extend([DrumHit(t, "hihat", 108, 1.), DrumHit(t+.20, "hihat", 65, 0.),
                 DrumHit(t+1, "hihat", 108, 1.)])
    return hits, t+3.5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gp1", type=Path, default=ROOT/"ref/tab1.gp")
    parser.add_argument("--gp2", type=Path, default=ROOT/"ref/tab2.gp")
    parser.add_argument("--guitar-seconds", type=float, default=20.)
    parser.add_argument("--band-seconds", type=float, default=12.3)
    parser.add_argument("--out", type=Path, default=HERE/"out")
    parser.add_argument("--only", choices=("all", "guitar", "band", "diagnostics"), default="all")
    args = parser.parse_args()
    if args.guitar_seconds <= 0 or args.band_seconds <= 0:
        parser.error("clip limits must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    manifest_path = args.out/"manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() and args.only != "all" else {}
    manifest.update({"sample_rate": dsp.SR, "format": "PCM24 stereo", "seeded": True,
                     "scope": "Python design model; no recorded samples; original distortion files untouched",
                     "presets": PRESETS,
                     "gp1_bass_drums": "GP1 has no drums; its bass starts at 12.3727 s, after the 12.26 s lead phrase. A separate GP1 bass entrance and a GP2 rhythm excerpt are supplied.",
                     "level_matching": "Same gated active RMS target per comparison (-20 dBFS); not an LUFS claim.",
                     "model_sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in (ROOT/"model/structure_modes.json", ROOT/"bass_fea/bass_fea_modes.json",
                                                 ROOT/"drum_set_fea/drum_set_fea_modes.json", ROOT/"drum_set_fea/tab3_drums_fea_modes.json")},
                     "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.gp1, args.gp2)}})
    manifest.setdefault("renders", {})

    def save(name, raw, target=-20.):
        audio, mastering = dsp.master(raw, target_db=target)
        path = args.out/f"{name}.wav"
        dsp.write_wav(path, audio)
        report = dsp.metrics(audio)
        manifest["renders"][name] = {"file": str(path), **report, "mastering": mastering}
        print(f"{name}: {len(audio)/dsp.SR:.2f}s, peak {report['true_peak_4x_dbtp']:.2f} dBTP, active RMS {report['active_rms_dbfs']:.2f}", flush=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return audio

    if args.only in {"all", "guitar"}:
        events, report = load_guitar(args.gp1, 0, args.guitar_seconds)
        if not events:
            raise ValueError("selected GP1 opening contains no lead notes")
        seconds = max(e.start+e.duration for e in events)+3.2
        manifest["gp1_score"] = report
        manifest["gp1_score"]["articulations"] = dict(Counter(e.articulation for e in events))
        guitar = CleanGuitar(report["tuning"])
        dry = {}
        for preset in PRESETS:
            voice = "clean_neck" if preset in {"clean_neck", "chorus_classic"} else "clean_glass"
            if voice not in dry:
                dry[voice] = guitar.render_di(events, voice, seconds)
            save(f"gp1_{preset}", guitar.process(dry[voice], preset))
        save("gp1_clean_neck_DI", dry["clean_neck"])
        save("gp1_legacy_string_clean_reference", legacy_clean(events, guitar, seconds))

    if args.only in {"all", "band"}:
        # An identical short source window, with 2.4 seconds retained for release/room.
        seconds = args.band_seconds+2.4
        notes, bass_report = load_bass(args.gp2, args.band_seconds)
        hits, drum_report = load_drums(args.gp2, args.band_seconds)
        manifest["gp2_bass_score"], manifest["gp2_drum_score"] = bass_report, drum_report
        bass = FingerBass(bass_report["tuning"])
        kit = AcousticKit()
        new_bass = save("gp2_bass_finger", bass.render(notes, seconds))
        old_bass = save("gp2_bass_legacy_matched", legacy_bass(notes, seconds))
        new_drums = save("gp2_drums_acoustic", kit.render(hits, seconds))
        old_drums = save("gp2_drums_legacy_matched", legacy_drums(hits, seconds))
        gap = np.zeros((round(.65*dsp.SR), 2))
        # Existing matched stems are concatenated without an extra normalization pass.
        for name, before, after in (("bass", old_bass, new_bass), ("drums", old_drums, new_drums)):
            path = args.out/f"gp2_{name}_before_then_after.wav"
            joined = np.concatenate([before, gap, after])
            dsp.write_wav(path, joined)
            manifest["renders"][path.stem] = {"file": str(path), **dsp.metrics(joined),
                                               "after_starts_seconds": seconds+.65}
        lead_events, lead_report = load_guitar(args.gp2, 0, args.band_seconds)
        rhythm_events, rhythm_report = load_guitar(args.gp2, 1, args.band_seconds)
        lead_guitar = CleanGuitar(lead_report["tuning"])
        rhythm_guitar = CleanGuitar(rhythm_report["tuning"], seed=1962)
        lead = lead_guitar.process(lead_guitar.render_di(lead_events, "clean_neck", seconds), "chorus_classic")
        rhythm = rhythm_guitar.process(rhythm_guitar.render_di(rhythm_events, "clean_glass", seconds), "clean_glass")
        lead, _ = dsp.master(lead, -22.)
        rhythm, _ = dsp.master(rhythm, -29.)
        save("gp2_clean_chorus_band", .88*lead+.75*rhythm+.80*new_bass+.92*new_drums, -19.)
        gp1_bass, gp1_bass_report = load_bass(args.gp1, 20.)
        if gp1_bass:
            offset = min(n.start for n in gp1_bass)-.20
            entrance = [replace(n, start=n.start-offset) for n in gp1_bass]
            gp1_bass_report["output_time_offset_seconds"] = offset
            manifest["gp1_bass_score"] = gp1_bass_report
            duration = max(n.start+n.duration for n in entrance)+1.0
            save("gp1_bass_entrance", FingerBass(gp1_bass_report["tuning"]).render(entrance, duration))

    if args.only in {"all", "diagnostics"}:
        notes = []
        t = .20
        for midi, string in ((28, 0), (33, 1), (38, 2), (43, 3), (40, 0)):
            for velocity in (48, 80, 116):
                notes.append(BassNote(t, .55, midi, velocity, string))
                t += .78
        save("bass/dynamics_and_strings", FingerBass().render(notes, t+1))
        hits, seconds = drum_diagnostic()
        manifest["drum_diagnostic_order"] = ["kick", "snare", "hihat", "tom_high", "tom_mid", "tom_low", "tom_floor", "crash", "ride", "hat_choke"]
        save("drums/kit_and_velocity_sweep", AcousticKit().render(hits, seconds), -23.)
    manifest["render_elapsed_seconds"] = round(time.perf_counter()-start, 2)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done in {manifest['render_elapsed_seconds']} s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
