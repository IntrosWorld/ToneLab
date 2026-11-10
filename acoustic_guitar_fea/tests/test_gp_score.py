"""Score-level regression checks: timing, tablature and attacks survive import."""
from __future__ import annotations

from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from gp_score import load_gp_score, rhythm_quarters  # noqa: E402


def _score(tmp_path, bars, *, tempos=((0, 0, 120),), repeat=False):
    """Construct an independent GPIF fixture from explicit notation."""
    root = ET.Element("GPIF")
    score = ET.SubElement(root, "Score")
    ET.SubElement(score, "Title").text = "Timing fixture"
    master_track = ET.SubElement(root, "MasterTrack")
    ET.SubElement(master_track, "Tracks").text = "7 8"
    automations = ET.SubElement(master_track, "Automations")
    for bar, position, bpm in tempos:
        automation = ET.SubElement(automations, "Automation")
        for key, value in {"Type": "Tempo", "Bar": bar, "Position": position,
                           "Linear": "false", "Value": f"{bpm} 2"}.items():
            ET.SubElement(automation, key).text = str(value)
    tracks = ET.SubElement(root, "Tracks")
    # XML definition order intentionally differs from MasterTrack order.
    for track_id in (8, 7):
        track = ET.SubElement(tracks, "Track", id=str(track_id))
        ET.SubElement(track, "Name").text = f"Guitar {track_id}"
        staff = ET.SubElement(ET.SubElement(track, "Staves"), "Staff")
        props = ET.SubElement(staff, "Properties")
        tuning = ET.SubElement(props, "Property", name="Tuning")
        ET.SubElement(tuning, "Pitches").text = "40 45 50 55 59 64"
    tables = {name: ET.SubElement(root, name) for name in ("MasterBars", "Bars", "Voices", "Beats", "Notes", "Rhythms")}
    for name in ("Whole", "Half", "Quarter", "Eighth"):
        ET.SubElement(ET.SubElement(tables["Rhythms"], "Rhythm", id=name), "NoteValue").text = name
    for bar_index, entries in enumerate(bars):
        master = ET.SubElement(tables["MasterBars"], "MasterBar")
        ET.SubElement(master, "Time").text = "4/4"
        ET.SubElement(master, "Bars").text = f"b{bar_index} unused{bar_index}"
        if repeat:
            ET.SubElement(master, "Repeat", start="true")
        bar = ET.SubElement(tables["Bars"], "Bar", id=f"b{bar_index}")
        ET.SubElement(bar, "Voices").text = f"v{bar_index} -1 -1 -1"
        voice = ET.SubElement(tables["Voices"], "Voice", id=f"v{bar_index}")
        beat_ids = []
        for beat_index, (rhythm, note_data) in enumerate(entries):
            beat_id = f"{bar_index}_{beat_index}"
            beat_ids.append(beat_id)
            beat = ET.SubElement(tables["Beats"], "Beat", id=beat_id)
            ET.SubElement(beat, "Rhythm", ref=rhythm)
            ET.SubElement(beat, "Dynamic").text = "P"
            if note_data is None:
                continue
            ET.SubElement(beat, "Notes").text = beat_id
            note = ET.SubElement(tables["Notes"], "Note", id=beat_id)
            props = ET.SubElement(note, "Properties")
            fret = note_data.get("fret", 0)
            for name, child, value in (("String", "String", 5), ("Fret", "Fret", fret),
                                       ("Midi", "Number", note_data.get("midi", 64 + fret))):
                ET.SubElement(ET.SubElement(props, "Property", name=name), child).text = str(value)
            if "tie" in note_data:
                ET.SubElement(note, "Tie", origin=str("o" in note_data["tie"]).lower(),
                              destination=str("d" in note_data["tie"]).lower())
            for enabled in note_data.get("enabled", []):
                ET.SubElement(ET.SubElement(props, "Property", name=enabled), "Enable")
            for name, value in note_data.get("bend", {}).items():
                ET.SubElement(ET.SubElement(props, "Property", name=name), "Float").text = str(value)
        ET.SubElement(voice, "Beats").text = " ".join(beat_ids)
    path = tmp_path / "fixture.gp"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("Content/score.gpif", ET.tostring(root))
    return path


def test_dotted_tuplet_duration():
    dotted = ET.fromstring('<Rhythm><NoteValue>Half</NoteValue><AugmentationDot count="2"/></Rhythm>')
    triplet = ET.fromstring('<Rhythm><NoteValue>Eighth</NoteValue><PrimaryTuplet num="3" den="2"/></Rhythm>')
    assert rhythm_quarters(dotted) == 3.5
    assert 3 * rhythm_quarters(triplet) == pytest.approx(1)


def test_opening_rest_ties_track_order_tempo_and_cutoff(tmp_path):
    path = _score(tmp_path, [
        [("Quarter", None), ("Quarter", {"fret": 3, "tie": "o"}), ("Half", {"fret": 3, "tie": "d"})],
        [("Whole", {"fret": 5})]], tempos=((0, 0, 120), (1, 0, 60)))
    result = load_gp_score(path, (0,), 2.5)
    track = result["tracks"][0]
    assert track["track_id"] == "7"
    assert [t["track_id"] for t in result["excluded_tracks"]] == ["8"]
    assert track["raw_notes"] == 3 and track["sounding_events"] == 2
    first, second = track["events"]
    assert (first["start"], first["duration"], first["midi"], first["string_index"], first["fret"], first["velocity"]) == (.5, 1.5, 67, 5, 3, 70)
    assert len(first["tied_note_ids"]) == 2
    assert (second["start"], second["duration"]) == (2.0, .5)
    assert result["played_bars"][1]["duration"] == 4.0


def test_note_spanning_within_bar_tempo_change(tmp_path):
    path = _score(tmp_path, [[("Whole", {"fret": 2})]], tempos=((0, 0, 120), (0, .5, 60)))
    event = load_gp_score(path, (0,), 10)["tracks"][0]["events"][0]
    assert event["start"] == 0
    assert event["duration"] == 3  # Two quarters at 120, then two at 60.


def test_hopo_keeps_destination_pitch_and_prebend_curve(tmp_path):
    path = _score(tmp_path, [[
        ("Quarter", {"fret": 5, "enabled": ["HopoOrigin"]}),
        ("Quarter", {"fret": 3, "enabled": ["HopoDestination"]}),
        ("Half", {"fret": 2, "enabled": ["Bended"], "bend": {"BendOriginValue": 100, "BendDestinationValue": 0}})]])
    events = load_gp_score(path, (0,), 10)["tracks"][0]["events"]
    assert events[1]["articulation"] == "pull_off"
    assert events[1]["midi"] == 67 and events[1]["source_midi"] == 69
    assert events[2]["midi"] == 66 and events[2]["articulation"] == "prebend_release"
    assert events[2]["bend_points"] == [[0, 2], [1, 0]]


def test_tie_does_not_join_notes_across_rest(tmp_path):
    path = _score(tmp_path, [[("Quarter", {"tie": "o"}), ("Quarter", None),
                            ("Half", {"tie": "d"})]])
    events = load_gp_score(path, (0,), 10)["tracks"][0]["events"]
    assert len(events) == 2
    assert events[0]["duration"] == .5
    assert events[1]["start"] == 1


def test_invalid_pitch_and_unsupported_repeats_are_explicit(tmp_path):
    path = _score(tmp_path, [[("Whole", {"midi": 40})]])
    with pytest.raises(ValueError, match="disagrees"):
        load_gp_score(path, (0,))
    path = _score(tmp_path, [[("Whole", {})]], repeat=True)
    with pytest.raises(ValueError, match="navigation"):
        load_gp_score(path, (0,))


def test_assamj_first_thirty_seconds_when_source_available():
    path = Path("C:/Users/Micro/Downloads/assamj.gp")
    if not path.exists():
        pytest.skip("user-provided Guitar Pro score is not installed")
    result = load_gp_score(path)
    lead, rhythm = result["tracks"]
    assert result["title"] == "Samjhawan"
    assert result["tempo_changes"][0]["bpm"] == 107
    assert (lead["program"], rhythm["program"]) == (26, 24)
    assert (lead["raw_notes"], lead["sounding_events"]) == (50, 46)
    assert (rhythm["raw_notes"], rhythm["sounding_events"]) == (71, 47)
    assert lead["first_note_s"] == pytest.approx(8 * 60 / 107)
    assert rhythm["first_note_s"] == 0
    assert lead["articulations"] == {"pick": 40, "pull_off": 5, "hammer_on": 1}
    assert all(0 <= e["start"] < 30 and e["start"] + e["duration"] <= 30 + 1e-8
               for t in result["tracks"] for e in t["events"])
