"""Render supplied MIDI demos, measure outputs, and save reproducible evidence.

python tools/validate_render.py --seconds 30 --numbers 9 10 11 12 13 14 15
python tools/validate_render.py --full 15 --baseline-seconds 4
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

from audio_io import audio_metrics, write_wav
from midi_events import clip_events, load_performance
from render_model import GrandPiano

ROOT = Path(__file__).resolve().parents[1]


def measure_render(piano, events, spans, destination, tail):
    started = time.perf_counter()
    audio = piano.render(events, spans, tail=tail)
    render_seconds = time.perf_counter() - started
    started = time.perf_counter()
    write_wav(destination, audio, piano.sample_rate)
    metrics = audio_metrics(audio, piano.sample_rate)
    try:
        report_path = destination.resolve().relative_to(ROOT)
    except ValueError:
        report_path = destination.resolve()
    metrics.update({"path": str(report_path), "notes": len(events),
                    "pedal_spans": len(spans), "render_seconds": render_seconds,
                    "write_seconds": time.perf_counter() - started,
                    "cache_mb": piano.cache_bytes / 1024**2 if hasattr(piano, "cache_bytes") else None})
    assert metrics["finite"] and metrics["clipped_samples"] == 0
    assert metrics["rms_dbfs"] > -80, "render is unexpectedly silent"
    assert np.all(audio[-1] == 0) if hasattr(piano, "cache_bytes") else True
    print(json.dumps(metrics), flush=True)
    destination.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numbers", nargs="*", type=int, default=[])
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--full", type=int, choices=range(9, 16))
    parser.add_argument("--rate", type=int, default=48000)
    parser.add_argument("--tail", type=float, default=4)
    parser.add_argument("--baseline-seconds", type=float)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "output" / "improved")
    args = parser.parse_args()
    if any(number not in range(9, 16) for number in args.numbers):
        parser.error("demo numbers must be 9 through 15")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    piano = GrandPiano(args.rate)
    jobs = [(number, False) for number in args.numbers]
    if args.full is not None:
        jobs.append((args.full, True))
    for number, full in jobs:
        performance = load_performance(ROOT / "midi" / f"demo{number}.mid")
        events, spans = performance.events, performance.pedal_spans
        if not full:
            events, spans = clip_events(events, spans, args.seconds)
        label = "full" if full else f"{args.seconds:g}s"
        print(f"Rendering demo{number} {label}: {len(events)} notes", flush=True)
        result = measure_render(piano, events, spans, args.out_dir / f"demo{number}_{label}_improved.wav", args.tail)
        result["midi_duration_seconds"] = performance.duration
        result["midi_warnings"] = list(performance.warnings)
        results.append(result)
        (args.out_dir / "suite_report.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    if args.baseline_seconds is not None:
        performance = load_performance(ROOT / "midi" / "demo15.mid")
        events, spans = clip_events(performance.events, performance.pedal_spans, args.baseline_seconds)
        spec = importlib.util.spec_from_file_location("original_render_model", ROOT / "validation" / "baseline_tools" / "render_model.py")
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        # Fresh engines, equal notes/rate/tail, no warm template-cache advantage.
        baseline = measure_render(original.GrandPiano(args.rate), events, spans,
                                  args.out_dir / "demo15_comparison_original.wav", 2)
        revised = measure_render(GrandPiano(args.rate), events, spans,
                                 args.out_dir / "demo15_comparison_improved.wav", 2)
        comparison = {"performance_seconds": args.baseline_seconds,
                      "original": baseline, "improved": revised,
                      "speedup": baseline["render_seconds"] / revised["render_seconds"]}
        (args.out_dir / "comparison_report.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
        print(f"Cold render speedup: {comparison['speedup']:.2f}x", flush=True)


if __name__ == "__main__":
    main()
