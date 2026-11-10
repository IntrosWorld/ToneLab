"""Render the exact drum track from tab2.gp into the physical FEA kit."""
from __future__ import annotations

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
from render_gp_score import DYNAMICS, rhythm_quarters

DURATION = 23.78404
ARTICULATION_TO_PIECE = {0: "snare", 5: "hihat", 8: "kick", 22: "crash"}


def parse_drum_track(source: Path):
    with zipfile.ZipFile(source) as archive:
        root = ET.fromstring(archive.read("Content/score.gpif"))
    bars = {x.get("id"): x for x in root.findall("./Bars/Bar")}
    voices = {x.get("id"): x for x in root.findall("./Voices/Voice")}
    beats = {x.get("id"): x for x in root.findall("./Beats/Beat")}
    notes = {x.get("id"): x for x in root.findall("./Notes/Note")}
    rhythms = {x.get("id"): x for x in root.findall("./Rhythms/Rhythm")}
    tempo_changes = {
        int(a.findtext("Bar")): float(a.findtext("Value").split()[0])
        for a in root.findall("./MasterTrack/Automations/Automation")
        if a.findtext("Type") == "Tempo"
    }
    tempo, elapsed, hits = tempo_changes.get(0, 120.), 0., []
    for master_index, master in enumerate(root.findall("./MasterBars/MasterBar")):
        tempo = tempo_changes.get(master_index, tempo)
        numerator, denominator = map(int, master.findtext("Time").split("/"))
        quarters_in_bar = numerator * 4. / denominator
        bar = bars[master.findtext("Bars").split()[3]]
        for voice_ref in (bar.findtext("Voices") or "").split():
            if voice_ref == "-1":
                continue
            cursor = 0.
            for beat_ref in voices[voice_ref].findtext("Beats").split():
                beat = beats[beat_ref]
                qduration = rhythm_quarters(rhythms[beat.find("Rhythm").get("ref")])
                start = elapsed + cursor*60./tempo
                cursor += qduration
                for note_ref in (beat.findtext("Notes") or "").split():
                    note = notes[note_ref]
                    articulation = int(note.findtext("InstrumentArticulation", "-1"))
                    if articulation not in ARTICULATION_TO_PIECE:
                        raise ValueError(f"unmapped GP drum articulation {articulation}")
                    piece = ARTICULATION_TO_PIECE[articulation]
                    velocity = DYNAMICS.get(beat.findtext("Dynamic") or "MF", 96)
                    hits.append(DrumHit(start, piece, velocity, 1.0 if articulation == 5 else 0.0))
        elapsed += quarters_in_bar*60./tempo
    return root.findall("./Tracks/Track")[3].findtext("Name").strip(), tempo_changes.get(0, tempo), hits


def main():
    source = ROOT / "ref" / "tab2.gp"
    track_name, tempo, score = parse_drum_track(source)
    model_path = HERE / "drum_set_fea_modes.json"
    model = generate_model(model_path)
    audio = render_drums(model, score, DURATION)
    output = HERE / "tab2_drums_fea.wav"
    write_pcm24(output, audio)
    report = {
        "source": str(source), "track_index": 3, "track_name": track_name,
        "tempo_bpm": tempo, "hits": len(score),
        "pieces": dict(Counter(h.piece for h in score)),
        "gp_articulation_mapping": ARTICULATION_TO_PIECE,
        "duration_s": DURATION, "model": str(model_path), "output": str(output),
        "events": [h.__dict__ for h in score],
    }
    (HERE / "tab2_drum_score.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "events"}, indent=2))


if __name__ == "__main__":
    main()
