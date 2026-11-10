"""Reuse the distortion project's tested GPIF articulation importer read-only."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from render_gp_score import parse_gpif, merge_ties, make_gestures, DYNAMICS
from render_solo import Gesture
import guitar_physics as physics


def load_guitar(source, track=0, seconds=20.):
    name, tempo, raw = parse_gpif(Path(source), track, seconds)
    gestures, rows = make_gestures(merge_ties(raw))
    with zipfile.ZipFile(source) as archive:
        xml = ET.fromstring(archive.read("Content/score.gpif"))
    node = xml.findall("./Tracks/Track")[track]
    tuning = [int(v) for v in node.findtext(".//Property[@name='Tuning']/Pitches").split()]
    return gestures, {"source": str(Path(source).resolve()), "track_index": track,
                      "track_name": name, "tempo_bpm": tempo, "tuning": tuning,
                      "raw_notes": len(raw), "sounding_events": len(gestures), "events": rows}
