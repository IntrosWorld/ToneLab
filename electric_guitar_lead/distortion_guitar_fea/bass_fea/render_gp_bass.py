"""Render an exact Guitar Pro bass track with the four-string bass FEA model."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "tools"))

from bass_model import BassNote, configure_tuning, generate_model, render_bass
from mix_tab2_guitars import write_pcm24
from render_gp_score import DYNAMICS, merge_ties, parse_gpif


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("score", type=Path)
    parser.add_argument("--track", type=int, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with zipfile.ZipFile(args.score) as archive:
        root = ET.fromstring(archive.read("Content/score.gpif"))
    metadata_track = root.findall("./Tracks/Track")[args.track]
    tuning = [int(value) for value in metadata_track.findtext(".//Property[@name='Tuning']/Pitches").split()]
    configure_tuning(tuning)
    track_name, tempo, raw = parse_gpif(args.score, args.track, args.seconds)
    events = merge_ties(raw)
    score = [BassNote(e["start"], max(.075, e["duration"]*.94), int(e["midi"]),
                      DYNAMICS.get(e["dynamic"], 96), int(e["string"])) for e in events]
    model_path = args.out.with_name(args.out.stem + "_modes.json")
    model = generate_model(model_path)
    audio = render_bass(model, score, args.seconds)
    write_pcm24(args.out, audio)
    report = {
        "source": str(args.score), "track_index": args.track, "track_name": track_name,
        "tempo_bpm": tempo, "tuning_open_midi_low_to_high": tuning,
        "raw_notes": len(raw), "sounding_events": len(events),
        "pitch_range_midi": [min(n.midi for n in score), max(n.midi for n in score)],
        "dynamics": dict(Counter(e["dynamic"] for e in events)),
        "duration_s": args.seconds, "model": str(model_path), "output": str(args.out),
        "events": [n.__dict__ for n in score],
    }
    report_path = args.out.with_name(args.out.stem + "_score.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "events"}, indent=2))


if __name__ == "__main__":
    main()
