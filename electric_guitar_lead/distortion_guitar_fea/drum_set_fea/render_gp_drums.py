"""Render an exact Guitar Pro drum track with the membrane/plate FEA kit."""
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

from drum_model import DrumHit, generate_model, render_drums
from mix_tab2_guitars import write_pcm24
from render_gp_score import DYNAMICS, properties, rhythm_quarters, scalar

MIDI_TO_PIECE = {
    36: "kick", 38: "snare", 41: "tom_floor", 45: "tom_low",
    47: "tom_mid", 48: "tom_high", 50: "tom_high",
    42: "hihat", 44: "hihat", 46: "hihat", 49: "crash", 57: "crash",
}


def parse_drum_track(source: Path, track_index: int):
    with zipfile.ZipFile(source) as archive:
        root = ET.fromstring(archive.read("Content/score.gpif"))
    bars = {x.get("id"): x for x in root.findall("./Bars/Bar")}
    voices = {x.get("id"): x for x in root.findall("./Voices/Voice")}
    beats = {x.get("id"): x for x in root.findall("./Beats/Beat")}
    notes = {x.get("id"): x for x in root.findall("./Notes/Note")}
    rhythms = {x.get("id"): x for x in root.findall("./Rhythms/Rhythm")}
    tempo_changes = {int(a.findtext("Bar")): float(a.findtext("Value").split()[0])
                     for a in root.findall("./MasterTrack/Automations/Automation")
                     if a.findtext("Type") == "Tempo"}
    tempo, elapsed, hits, mapping = tempo_changes.get(0, 120.), 0., [], Counter()
    for master_index, master in enumerate(root.findall("./MasterBars/MasterBar")):
        tempo = tempo_changes.get(master_index, tempo)
        numerator, denominator = map(int, master.findtext("Time").split("/"))
        quarters = numerator*4./denominator
        bar = bars[master.findtext("Bars").split()[track_index]]
        for voice_ref in (bar.findtext("Voices") or "").split():
            if voice_ref == "-1": continue
            cursor = 0.
            for beat_ref in voices[voice_ref].findtext("Beats").split():
                beat = beats[beat_ref]
                qduration = rhythm_quarters(rhythms[beat.find("Rhythm").get("ref")])
                start = elapsed + cursor*60./tempo
                cursor += qduration
                for note_ref in (beat.findtext("Notes") or "").split():
                    note = notes[note_ref]; props = properties(note)
                    midi = int(scalar(props.get("Midi"), "Number", -1))
                    if midi not in MIDI_TO_PIECE:
                        raise ValueError(f"unmapped GP drum MIDI note {midi}")
                    piece = MIDI_TO_PIECE[midi]
                    open_amount = 1.0 if midi == 46 else (.05 if midi == 44 else 0.)
                    velocity = DYNAMICS.get(beat.findtext("Dynamic") or "MF", 96)
                    hits.append(DrumHit(start, piece, velocity, open_amount))
                    mapping[(int(note.findtext("InstrumentArticulation", "-1")), midi, piece)] += 1
        elapsed += quarters*60./tempo
    name = root.findall("./Tracks/Track")[track_index].findtext("Name").strip()
    return name, tempo_changes.get(0, tempo), elapsed, hits, mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("score", type=Path)
    parser.add_argument("--track", type=int, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    track_name, tempo, score_end, score, mapping = parse_drum_track(args.score, args.track)
    model_path = args.out.with_name(args.out.stem + "_modes.json")
    model = generate_model(model_path)
    audio = render_drums(model, score, args.seconds)
    write_pcm24(args.out, audio)
    report = {
        "source": str(args.score), "track_index": args.track, "track_name": track_name,
        "tempo_bpm": tempo, "score_end_s": score_end, "hits": len(score),
        "pieces": dict(Counter(h.piece for h in score)),
        "gp_mapping": [{"articulation": a, "midi": m, "piece": p, "hits": count}
                       for (a, m, p), count in sorted(mapping.items())],
        "duration_s": args.seconds, "model": str(model_path), "output": str(args.out),
        "events": [h.__dict__ for h in score],
    }
    report_path = args.out.with_name(args.out.stem + "_score.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "events"}, indent=2))


if __name__ == "__main__":
    main()
