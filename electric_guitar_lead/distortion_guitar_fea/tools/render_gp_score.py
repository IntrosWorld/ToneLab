"""Render the first section of a modern Guitar Pro (.gp/GPIF) lead track."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

from render_solo import Gesture, LeadGuitar, write_wav
import guitar_physics as phys

ROOT = Path(__file__).resolve().parents[1]
NOTE_VALUES = {"Whole": 4.0, "Half": 2.0, "Quarter": 1.0, "Eighth": .5,
               "16th": .25, "32nd": .125, "64th": .0625}
DYNAMICS = {"PPP": 48, "PP": 58, "P": 70, "MP": 82, "MF": 96,
            "F": 108, "FF": 118, "FFF": 124}


def rhythm_quarters(node):
    base = NOTE_VALUES[node.findtext("NoteValue")]
    value = base
    dot = node.find("AugmentationDot")
    if dot is not None:
        for power in range(1, int(dot.get("count", "0")) + 1):
            value += base / 2**power
    tuplet = node.find("PrimaryTuplet")
    if tuplet is not None:
        value *= int(tuplet.get("den")) / int(tuplet.get("num"))
    return value


def properties(note):
    return {p.get("name"): p for p in note.findall("./Properties/Property")}


def scalar(prop, child, default=0.0):
    if prop is None: return default
    text = prop.findtext(child)
    return default if text is None else float(text)


def parse_gpif(path: Path, track_index: int, limit_s: float):
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("Content/score.gpif"))
    tracks = root.findall("./Tracks/Track")
    if not 0 <= track_index < len(tracks):
        raise ValueError(f"track {track_index} outside 0..{len(tracks)-1}")
    track = tracks[track_index]
    bar_nodes = {x.get("id"): x for x in root.findall("./Bars/Bar")}
    voices = {x.get("id"): x for x in root.findall("./Voices/Voice")}
    beats = {x.get("id"): x for x in root.findall("./Beats/Beat")}
    notes = {x.get("id"): x for x in root.findall("./Notes/Note")}
    rhythms = {x.get("id"): x for x in root.findall("./Rhythms/Rhythm")}
    tempo_changes = {}
    for automation in root.findall("./MasterTrack/Automations/Automation"):
        if automation.findtext("Type") == "Tempo":
            tempo_changes[int(automation.findtext("Bar"))] = float(automation.findtext("Value").split()[0])
    tempo = tempo_changes.get(0, 120.0)
    events, elapsed = [], 0.0
    master_bars = root.findall("./MasterBars/MasterBar")
    for master_index, master in enumerate(master_bars):
        if elapsed >= limit_s: break
        tempo = tempo_changes.get(master_index, tempo)
        num, den = map(int, master.findtext("Time").split("/"))
        quarters_in_bar = num * 4.0 / den
        bar_refs = master.findtext("Bars").split()
        bar = bar_nodes[bar_refs[track_index]]
        for voice_ref in (bar.findtext("Voices") or "").split():
            if voice_ref == "-1": continue
            cursor = 0.0
            for beat_ref in voices[voice_ref].findtext("Beats").split():
                beat = beats[beat_ref]
                qdur = rhythm_quarters(rhythms[beat.find("Rhythm").get("ref")])
                start = elapsed + cursor * 60.0 / tempo
                duration = qdur * 60.0 / tempo
                cursor += qdur
                if start >= limit_s: continue
                for note_ref in (beat.findtext("Notes") or "").split():
                    note = notes[note_ref]; props = properties(note)
                    if "Midi" not in props or "String" not in props: continue
                    tie = note.find("Tie")
                    events.append({
                        "start": start, "duration": min(duration, limit_s-start),
                        "midi": scalar(props["Midi"], "Number"),
                        "string": int(scalar(props["String"], "String")),
                        "fret": scalar(props.get("Fret"), "Fret"),
                        "dynamic": beat.findtext("Dynamic") or "MF",
                        "accent": note.find("Accent") is not None,
                        "anti_accent": note.find("AntiAccent") is not None,
                        "vibrato": note.findtext("Vibrato"),
                        "slide": int(scalar(props.get("Slide"), "Flags")),
                        "bended": "Bended" in props,
                        "bend_origin": scalar(props.get("BendOriginValue"), "Float") / 50.0,
                        "bend_middle": scalar(props.get("BendMiddleValue"), "Float") / 50.0,
                        "bend_destination": scalar(props.get("BendDestinationValue"), "Float") / 50.0,
                        "hopo_origin": "HopoOrigin" in props,
                        "hopo_destination": "HopoDestination" in props,
                        "muted": "Muted" in props,
                        "tie_origin": tie is not None and tie.get("origin") == "true",
                        "tie_destination": tie is not None and tie.get("destination") == "true",
                        "bar": master_index + 1,
                    })
        elapsed += quarters_in_bar * 60.0 / tempo
    return track.findtext("Name").strip(), tempo_changes.get(0, tempo), events


def merge_ties(events):
    merged, active = [], {}
    for event in sorted(events, key=lambda x: (x["start"], x["string"], x["midi"])):
        key = event["string"]
        if event["tie_destination"] and key in active:
            prior = active[key]
            prior["duration"] = max(prior["duration"], event["start"] + event["duration"] - prior["start"])
            prior["vibrato"] = prior["vibrato"] or event["vibrato"]
            # GP can attach the slide-out to the last continuation of a tied
            # sustain. Carry that terminal gesture back to the sounding note.
            if event["slide"]:
                prior["terminal_slide"] = event["slide"]
            if not event["tie_origin"]: active.pop(key, None)
            continue
        merged.append(event)
        if event["tie_origin"]: active[key] = event
    return merged


def make_gestures(events):
    gestures, rows = [], []
    by_string = {}
    for i, event in enumerate(events): by_string.setdefault(event["string"], []).append(i)
    previous_on_string = {}
    for index, event in enumerate(events):
        articulation, midi, target = "pick", event["midi"], None
        same = by_string[event["string"]]; pos = same.index(index)
        next_event = events[same[pos+1]] if pos + 1 < len(same) else None
        previous = previous_on_string.get(event["string"])
        slide_flags = event.get("terminal_slide", event["slide"])
        slide_out = slide_flags & 0x3F
        slide_in = (slide_flags >> 6) & 0x03
        if event["muted"] and slide_in:
            # Both occurrences in tab1 are muted open strings: the slide mark
            # describes the hand scrape, not a pitched fret below the nut.
            target, articulation = midi, "dead_slide_in"
        elif event["muted"]:
            articulation = "dead_note"
        elif event["bended"]:
            values = [event["bend_origin"], event["bend_middle"], event["bend_destination"]]
            amount = max(values)
            if event["bend_origin"] > 0 and event["bend_destination"] < event["bend_origin"]:
                midi, target, articulation = midi + event["bend_origin"], midi, "prebend_release"
            elif event["bend_destination"] < amount:
                target, articulation = midi + amount, "bend_release"
            else:
                target, articulation = midi + amount, "bend_up"
        elif slide_in:
            target, midi, articulation = midi, midi + (-3 if slide_in == 1 else 3), "slide"
        elif slide_out in (1, 2) and next_event is not None:
            target = next_event["midi"]
            articulation = "legato_slide" if slide_out == 2 else "slide"
        elif slide_out in (3, 4):
            target = midi + (3 if slide_out == 3 else -3)
            articulation = "slide_out_up" if slide_out == 3 else "slide_out_down"
        elif event["hopo_destination"] and previous is not None:
            midi, target = previous["midi"], event["midi"]
            articulation = "hammer_on" if target >= midi else "pull_off"
        elif event["accent"]:
            articulation = "accent"
        velocity = DYNAMICS.get(event["dynamic"], 96)
        if event["accent"]: velocity = min(124, velocity + 10)
        if event["anti_accent"]: velocity = max(45, velocity - 24)
        vibrato = 18.0 if event["vibrato"] else (13.0 if event["bended"] else 0.0)
        duration = max(.075, event["duration"] * .96)
        gesture = Gesture(event["start"], duration, midi, target, velocity,
                          articulation, vibrato, event["string"])
        gestures.append(gesture)
        rows.append({**event, "render_midi": midi, "target_midi": target,
                     "articulation": articulation, "velocity": velocity})
        previous_on_string[event["string"]] = event
    return gestures, rows


def main():
    p = argparse.ArgumentParser(); p.add_argument("score", type=Path)
    p.add_argument("--track", type=int, default=2); p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--delay-level", type=float, default=1.0)
    p.add_argument("--out", type=Path, default=ROOT/"out"/"cfy_first60_fea.wav")
    args = p.parse_args()
    track_name, tempo, raw = parse_gpif(args.score, args.track, args.seconds)
    merged = merge_ties(raw); gestures, rows = make_gestures(merged)
    with zipfile.ZipFile(args.score) as archive:
        metadata_root = ET.fromstring(archive.read("Content/score.gpif"))
    title = (metadata_root.findtext("./Score/Title") or args.score.stem).strip()
    metadata_track = metadata_root.findall("./Tracks/Track")[args.track]
    tuning_text = metadata_track.findtext(".//Property[@name='Tuning']/Pitches") or "40 45 50 55 59 64"
    tuning = [int(value) for value in tuning_text.split()]
    phys.configure_tuning(tuning)
    # Rebuild after configuring tuning; LeadGuitar reads the physics module at render time.
    audio = LeadGuitar(tempo_bpm=tempo, delay_level=args.delay_level).render(gestures)[:int(args.seconds*48_000)]
    write_wav(args.out, audio, 48_000)
    sound_path = (metadata_track.findtext("./Sounds/Sound/Path") or "").strip()
    strip_parameters = (metadata_track.findtext("./RSE/ChannelStrip/Parameters") or "").strip()
    report = {
        "source": str(args.score), "title": title,
        "track_index": args.track, "track_name": track_name, "tempo_bpm": tempo,
        "tuning_open_midi_low_to_high": tuning,
        "gp_audio_engine": (metadata_track.findtext("AudioEngineState") or "").strip(),
        "gp_sound_path": sound_path,
        "gp_channel_strip_parameters": [float(v) for v in strip_parameters.split()] if strip_parameters else [],
        "tone_implementation": "white-box four-stage high-gain preamp plus FEA cabinet",
        "delay_level": args.delay_level, "delay_time_s": .75*60/tempo,
        "rendered_seconds": len(audio)/48_000, "requested_limit_seconds": args.seconds,
        "raw_notes": len(raw), "sounding_events": len(gestures),
        "articulations": dict(Counter(g.articulation for g in gestures)), "events": rows,
    }
    report_path = args.out.with_name(args.out.stem + "_score.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "events"}, indent=2))
    print(f"rendered {len(audio)/48000:.2f} s -> {args.out}")


if __name__ == "__main__": main()
