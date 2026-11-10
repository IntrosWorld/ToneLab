"""Render the exact Electric Bass (finger) track from tab2.gp."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "tools"))

from bass_model import BassNote, generate_model, render_bass
from mix_tab2_guitars import write_pcm24
from render_gp_score import DYNAMICS, merge_ties, parse_gpif

DURATION = 23.78404


def main():
    source = ROOT / "ref" / "tab2.gp"
    track_name, tempo, raw = parse_gpif(source, 2, DURATION)
    events = merge_ties(raw)
    score = [
        BassNote(e["start"], max(.075, e["duration"]*.94), int(e["midi"]), DYNAMICS.get(e["dynamic"], 96))
        for e in events
    ]
    model_path = HERE / "bass_fea_modes.json"
    model = generate_model(model_path)
    audio = render_bass(model, score, DURATION)
    output = HERE / "tab2_bass_fea.wav"
    write_pcm24(output, audio)
    report = {
        "source": str(source), "track_index": 2, "track_name": track_name,
        "tempo_bpm": tempo, "tuning_open_midi_low_to_high": [28, 33, 38, 43],
        "raw_notes": len(raw), "sounding_events": len(events),
        "pitch_range_midi": [min(n.midi for n in score), max(n.midi for n in score)],
        "dynamics": dict(Counter(e["dynamic"] for e in events)),
        "duration_s": DURATION, "model": str(model_path), "output": str(output),
        "events": [n.__dict__ for n in score],
    }
    (HERE / "tab2_bass_score.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "events"}, indent=2))


if __name__ == "__main__":
    main()
