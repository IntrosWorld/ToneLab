"""Read modern Guitar Pro GPIF guitar parts without importing a synthesizer.

The basic GPIF table traversal and dynamic mapping are adapted from this
repository's ``electric_guitar_solo_fea/tools/render_gp_score.py``. This version
keeps exact tablature positions, voice-local ties and within-bar step tempos.
Events use seconds, MIDI velocities, and zero-based strings from low to high.
Repeat navigation and tempo ramps are explicitly rejected rather than silently
producing an incorrect performance; neither occurs in ``assamj.gp``.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile


NOTE_VALUES = {"Whole": 4.0, "Half": 2.0, "Quarter": 1.0, "Eighth": .5,
               "16th": .25, "32nd": .125, "64th": .0625, "128th": .03125}
DYNAMICS = {"PPP": 48, "PP": 58, "P": 70, "MP": 82, "MF": 96,
            "F": 108, "FF": 118, "FFF": 124}


def rhythm_quarters(node: ET.Element) -> float:
    """Notated beat length, including augmentation dots and tuplets."""
    base = NOTE_VALUES[node.findtext("NoteValue")]
    dot = node.find("AugmentationDot")
    count = 0 if dot is None else int(dot.get("count", "1"))
    duration = base * sum(2.0 ** -i for i in range(count + 1))
    for tuplet in node.findall("PrimaryTuplet") + node.findall("SecondaryTuplet"):
        duration *= int(tuplet.get("den")) / int(tuplet.get("num"))
    return duration


def _properties(node: ET.Element) -> dict:
    return {p.get("name"): p for p in node.findall("./Properties/Property")}


def _number(props: dict, name: str, child: str, default: float = 0) -> float:
    prop = props.get(name)
    value = None if prop is None else prop.findtext(child)
    return default if value is None else float(value)


def _enabled(props: dict, name: str) -> bool:
    prop = props.get(name)
    return prop is not None and prop.find("Disable") is None


def _text(node: ET.Element, path: str, default: str = "") -> str:
    return (node.findtext(path) or default).strip()


def _bend_points(props: dict) -> list[list[float]]:
    if not _enabled(props, "Bended"):
        return []
    points = [[_number(props, "BendOriginOffset", "Float") / 100,
               _number(props, "BendOriginValue", "Float") / 50]]
    if "BendMiddleValue" in props:
        for key, default in (("BendMiddleOffset1", 50), ("BendMiddleOffset2", 50)):
            point = [_number(props, key, "Float", default) / 100,
                     _number(props, "BendMiddleValue", "Float") / 50]
            if point != points[-1]:
                points.append(point)
    points.append([_number(props, "BendDestinationOffset", "Float", 100) / 100,
                   _number(props, "BendDestinationValue", "Float") / 50])
    return sorted(points)


def _merge_ties(events: list[dict]) -> list[dict]:
    merged, active = [], {}
    for event in sorted(events, key=lambda e: (e["start"], e["voice"], e["string_index"])):
        key = event["voice"], event["string_index"]
        prior = active.get(key)
        contiguous = prior is not None and abs(prior["start"] + prior["duration"] - event["start"]) < 1e-7
        if event["tie_destination"] and contiguous and prior["midi"] == event["midi"]:
            prior["duration"] = event["start"] + event["duration"] - prior["start"]
            prior["tied_note_ids"].append(event["note_id"])
            prior["tie_origin"] = event["tie_origin"]
            if not event["tie_origin"]:
                active.pop(key, None)
            continue
        active.pop(key, None)
        merged.append(event)
        if event["tie_origin"]:
            active[key] = event
    return merged


def _articulations(events: list[dict]) -> None:
    previous = {}
    for event in events:
        key = event["voice"], event["string_index"]
        prior = previous.get(key)
        event["articulation"] = "pick"
        if event["muted"]:
            event["articulation"] = "dead_note"
        elif event["bend_points"]:
            event["articulation"] = "prebend_release" if event["bend_origin"] > event["bend_destination"] else "bend"
        elif event["hopo_destination"] and prior is not None and abs(prior["start"] + prior["duration"] - event["start"]) < 1e-7:
            event["source_midi"] = prior["midi"]
            event["articulation"] = "hammer_on" if event["midi"] > prior["midi"] else "pull_off"
        elif event["slide"]:
            event["articulation"] = "slide"
        previous[key] = event


def load_gp_score(path: str | Path, track_indices=(0, 1), limit_s: float = 30.0) -> dict:
    """Return score metadata and selected guitar tracks with merged note events.

    The timeline begins at the score's first bar, including opening rests; notes
    are clipped at ``limit_s``. MIDI from the file is authoritative and its
    agreement with tuning + capo + fret is validated. Only selected tracks
    produce events. The original instrument program does not determine how a
    caller synthesizes those events.
    """
    if not math.isfinite(limit_s) or limit_s <= 0:
        raise ValueError("limit_s must be a finite positive duration")
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("Content/score.gpif"))
    tables = {kind: {x.get("id"): x for x in root.findall(f"./{kind}s/{kind}")}
              for kind in ("Bar", "Voice", "Beat", "Note", "Rhythm")}
    all_tracks = root.findall("./Tracks/Track")
    order = _text(root, "./MasterTrack/Tracks").split() or [t.get("id") for t in all_tracks]
    ordered_tracks = {track.get("id"): track for track in all_tracks}
    tracks = [ordered_tracks[track_id] for track_id in order]
    selected = tuple(dict.fromkeys(int(i) for i in track_indices))
    if not selected or any(i < 0 or i >= len(tracks) for i in selected):
        raise ValueError(f"track_indices must select indices in 0..{len(tracks)-1}")

    track_data, bar_offsets = [], []
    offset = 0
    for index, track in enumerate(tracks):
        staves = track.findall("./Staves/Staff")
        staff = staves[0] if staves else track
        props = _properties(staff)
        tuning_prop = props.get("Tuning")
        tuning = [] if tuning_prop is None else [int(x) for x in _text(tuning_prop, "Pitches").split()]
        data = {"track_index": index, "track_id": track.get("id"), "name": _text(track, "Name"),
                "instrument": _text(track, "./InstrumentSet/Name"),
                "program": int(_text(track, "./Sounds/Sound/MIDI/Program", "-1")),
                "tuning": tuning, "capo": int(_number(props, "CapoFret", "Fret")), "events": [],
                "rest_beats": 0}
        track_data.append(data)
        bar_offsets.append(offset)
        offset += max(1, len(staves))
        if index in selected and len(staves) > 1:
            raise ValueError(f"Multiple staves are not supported for guitar track {index}")

    changes = []
    for automation in root.findall("./MasterTrack/Automations/Automation"):
        if _text(automation, "Type") != "Tempo":
            continue
        values = _text(automation, "Value").split()
        bpm = float(values[0])
        if bpm <= 0 or not math.isfinite(bpm):
            raise ValueError("Tempo must be finite and positive")
        changes.append({"bar": int(_text(automation, "Bar", "0")),
                        "position": float(_text(automation, "Position", "0")),
                        "bpm": bpm, "linear": _text(automation, "Linear") == "true",
                        "tempo_unit": int(values[1]) if len(values) > 1 else 2})
    changes.sort(key=lambda c: (c["bar"], c["position"]))
    elapsed, tempo, played_bars = 0.0, 120.0, []
    masters = root.findall("./MasterBars/MasterBar")
    dynamics = {}
    for bar_index, master in enumerate(masters):
        if elapsed >= limit_s - 1e-9:
            break
        if any(master.find(tag) is not None for tag in ("Repeat", "AlternateEndings", "Directions")):
            raise ValueError(f"Repeat/direction navigation in bar {bar_index+1} is not supported")
        num, den = map(int, _text(master, "Time", "4/4").split("/"))
        bar_quarters = num * 4.0 / den
        bar_changes = [c for c in changes if c["bar"] == bar_index]
        segments = [(0.0, tempo)]
        for change in bar_changes:
            if change["linear"] or change["tempo_unit"] != 2:
                raise ValueError("Only step tempos in quarter-note BPM are supported")
            position = change["position"]
            if not 0 <= position <= 1:
                raise ValueError("GPIF tempo Position must be between 0 and 1")
            segments.append((position * bar_quarters, change["bpm"]))
        segments.sort(key=lambda item: item[0])

        def seconds_at(quarter):
            seconds, previous_q, bpm = 0.0, 0.0, segments[0][1]
            for q, next_bpm in segments[1:]:
                if q >= quarter:
                    break
                seconds += (q - previous_q) * 60 / bpm
                previous_q, bpm = q, next_bpm
            return seconds + (quarter - previous_q) * 60 / bpm

        bar_duration = seconds_at(bar_quarters)
        played_bars.append({"bar": bar_index + 1, "start": elapsed, "duration": bar_duration,
                            "time_signature": f"{num}/{den}"})
        refs = _text(master, "Bars").split()
        for index in selected:
            data = track_data[index]
            bar = tables["Bar"][refs[bar_offsets[index]]]
            for voice_index, voice_ref in enumerate(_text(bar, "Voices").split()):
                if voice_ref == "-1":
                    continue
                cursor = 0.0
                for beat_ref in _text(tables["Voice"][voice_ref], "Beats").split():
                    beat = tables["Beat"][beat_ref]
                    qdur = rhythm_quarters(tables["Rhythm"][beat.find("Rhythm").get("ref")])
                    start = elapsed + seconds_at(cursor)
                    end = elapsed + seconds_at(min(bar_quarters, cursor + qdur))
                    cursor += qdur
                    if start >= limit_s - 1e-9:
                        continue
                    dynamic = _text(beat, "Dynamic", dynamics.get((index, voice_index), "MF"))
                    dynamics[index, voice_index] = dynamic
                    note_refs = _text(beat, "Notes").split()
                    if not note_refs:
                        data["rest_beats"] += 1
                    for note_ref in note_refs:
                        note = tables["Note"][note_ref]
                        props = _properties(note)
                        if "String" not in props or "Fret" not in props:
                            raise ValueError(f"Note {note_ref} has no guitar tablature position")
                        string_index = int(_number(props, "String", "String"))
                        fret = int(_number(props, "Fret", "Fret"))
                        if not 0 <= string_index < len(data["tuning"]):
                            raise ValueError(f"Invalid string index on note {note_ref}")
                        expected = data["tuning"][string_index] + data["capo"] + fret
                        midi = _number(props, "Midi", "Number", expected)
                        muted = _enabled(props, "Muted")
                        if not muted and midi != expected:
                            raise ValueError(f"Note {note_ref}: MIDI {midi} disagrees with string/fret pitch {expected}")
                        accent = note.find("Accent") is not None
                        anti_accent = note.find("AntiAccent") is not None
                        velocity = max(1, min(127, DYNAMICS.get(dynamic, 96) + 10 * accent - 24 * anti_accent))
                        tie = note.find("Tie")
                        points = _bend_points(props)
                        data["events"].append({
                            "start": start, "duration": min(end, limit_s) - start,
                            "midi": midi, "velocity": velocity, "dynamic": dynamic,
                            "string_index": string_index, "fret": fret, "bar": bar_index + 1,
                            "voice": voice_index, "beat_id": beat_ref, "note_id": note_ref,
                            "tied_note_ids": [note_ref],
                            "tie_origin": tie is not None and tie.get("origin") == "true",
                            "tie_destination": tie is not None and tie.get("destination") == "true",
                            "hopo_origin": _enabled(props, "HopoOrigin"),
                            "hopo_destination": _enabled(props, "HopoDestination"),
                            "muted": muted, "palm_mute": _enabled(props, "PalmMuted"),
                            "staccato": note.find("Staccato") is not None,
                            "accent": accent, "anti_accent": anti_accent,
                            "vibrato": _text(note, "Vibrato"),
                            "slide": int(_number(props, "Slide", "Flags")),
                            "bend_points": points,
                            "bend_origin": _number(props, "BendOriginValue", "Float") / 50,
                            "bend_middle": _number(props, "BendMiddleValue", "Float") / 50,
                            "bend_destination": _number(props, "BendDestinationValue", "Float") / 50,
                        })
        elapsed += bar_duration
        tempo = segments[-1][1]

    result_tracks = []
    for index in selected:
        data = track_data[index]
        data["raw_notes"] = len(data["events"])
        data["events"] = _merge_ties(data["events"])
        _articulations(data["events"])
        data["sounding_events"] = len(data["events"])
        data["articulations"] = dict(Counter(e["articulation"] for e in data["events"]))
        data["first_note_s"] = data["events"][0]["start"] if data["events"] else None
        result_tracks.append(data)
    excluded = [{k: v for k, v in d.items() if k not in ("events", "rest_beats")}
                for i, d in enumerate(track_data) if i not in selected]
    return {"source": str(path.resolve()), "title": _text(root, "./Score/Title", path.stem),
            "artist": _text(root, "./Score/Artist"), "requested_seconds": limit_s,
            "score_duration_parsed_s": elapsed, "tempo_changes": changes,
            "total_notated_bars": len(masters), "played_bars": played_bars,
            "tracks": result_tracks, "excluded_tracks": excluded}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("score", type=Path)
    parser.add_argument("--tracks", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = load_gp_score(args.score, args.tracks, args.seconds)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({**report, "tracks": [{k: v for k, v in track.items() if k != "events"}
                                         for track in report["tracks"]]}, indent=2))


if __name__ == "__main__":
    main()
