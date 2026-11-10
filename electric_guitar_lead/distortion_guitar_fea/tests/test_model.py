import json
import sys
from pathlib import Path
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import guitar_physics as physics
from render_gp_score import make_gestures, merge_ties, parse_gpif


def test_model_tables_and_playable_range():
    structure = json.loads((ROOT / "model" / "structure_modes.json").read_text())
    cabinet = json.loads((ROOT / "model" / "amp_cabinet_modes.json").read_text())
    assert structure["modes"] and cabinet["modes"]
    assert physics.choose_string(69) == 5
    assert physics.fret_for_midi(69, 5) == 5


def test_distortion_render_manifest_selects_only_lead_track():
    score = json.loads((ROOT / "out" / "tab1_distortion_guitar_fea_score.json").read_text())
    assert score["track_index"] == 0
    assert score["track_name"] == "Lead Guitar"
    assert score["sounding_events"] == 47


def test_tab1_playing_effects_survive_gpif_mapping():
    _, _, raw = parse_gpif(ROOT / "ref" / "tab1.gp", 0, 20)
    gestures, _ = make_gestures(merge_ties(raw))
    articulations = [gesture.articulation for gesture in gestures]
    assert articulations.count("dead_slide_in") == 2
    assert articulations.count("bend_up") == 8
    assert articulations.count("prebend_release") == 1
    assert articulations.count("slide_out_down") == 1
    assert articulations.count("legato_slide") == 1
    assert articulations.count("pull_off") == 4
    assert sum(bool(event["vibrato"]) for event in raw) == 2  # one sustain across a tied continuation
    assert sum(event["accent"] for event in raw) == 1
    assert sum(event["anti_accent"] for event in raw) == 1


def test_tab2_selects_both_guitars_and_preserves_effects():
    expected = {
        0: {"bend_up": 2, "bend_release": 2, "hammer_on": 8,
            "pull_off": 4, "legato_slide": 4, "pick": 54},
        1: {"pick": 18},
    }
    for track, counts in expected.items():
        name, tempo, raw = parse_gpif(ROOT / "ref" / "tab2.gp", track, 25)
        gestures, _ = make_gestures(merge_ties(raw))
        actual = {kind: sum(g.articulation == kind for g in gestures) for kind in counts}
        assert actual == counts
        assert tempo == 163
        assert name in ("Lead Guitar", "Rhythm Guitar")
    manifest = json.loads((ROOT / "out" / "tab2_guitar_only_manifest.json").read_text())
    assert [track["index"] for track in manifest["included_tracks"]] == [0, 1]
    assert [track["index"] for track in manifest["excluded_tracks"]] == [2, 3]


def test_tab2_exact_bass_and_drum_tracks_are_rendered():
    bass = json.loads((ROOT / "bass_fea" / "tab2_bass_score.json").read_text())
    drums = json.loads((ROOT / "drum_set_fea" / "tab2_drum_score.json").read_text())
    assert bass["track_index"] == 2
    assert bass["sounding_events"] == 41
    assert bass["pitch_range_midi"] == [29, 43]
    assert drums["track_index"] == 3
    assert drums["hits"] == 103
    assert drums["pieces"] == {"kick": 36, "crash": 3, "hihat": 48, "snare": 16}


def test_tab2_full_band_manifest_and_audio():
    manifest = json.loads((ROOT / "out" / "tab2_full_band_manifest.json").read_text())
    assert [track["index"] for track in manifest["included_tracks"]] == [0, 1, 2, 3]
    assert manifest["peak"] <= .94
    with wave.open(str(ROOT / "out" / "tab2_full_band_fea.wav"), "rb") as audio:
        assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (48_000, 2, 3)
        assert audio.getnframes() == 1_141_633


def test_legato_slide_holds_pitch_then_connects_late():
    import numpy as np
    time = np.array([0., .50, .75, .90, 1.0])
    pitch = physics.pitch_curve(74., 72., "legato_slide", time, 1.0)
    assert pitch[1] == 74.
    assert 72. < pitch[2] < 74.
    assert pitch[3] == 72.
    assert pitch[4] == 72.


def test_tab3_all_gp_tracks_and_kit_voices_are_rendered():
    lead = json.loads((ROOT / "out" / "tab3_lead_guitar_fea_score.json").read_text())
    rhythm = json.loads((ROOT / "out" / "tab3_rhythm_guitar_fea_score.json").read_text())
    bass = json.loads((ROOT / "bass_fea" / "tab3_bass_fea_score.json").read_text())
    drums = json.loads((ROOT / "drum_set_fea" / "tab3_drums_fea_score.json").read_text())
    assert (lead["sounding_events"], rhythm["sounding_events"]) == (505, 1025)
    assert rhythm["articulations"]["dead_note"] == 72
    assert bass["sounding_events"] == 335
    assert bass["tuning_open_midi_low_to_high"] == [26, 33, 38, 43]
    assert drums["hits"] == 996
    assert drums["pieces"] == {
        "snare": 166, "tom_low": 22, "kick": 274, "crash": 30,
        "hihat": 485, "tom_floor": 8, "tom_high": 9, "tom_mid": 2,
    }


def test_tab3_full_band_audio_and_manifest():
    manifest = json.loads((ROOT / "out" / "tab3_full_band_fea_manifest.json").read_text())
    assert [track["index"] for track in manifest["included_tracks"]] == [0, 1, 2, 3]
    assert manifest["tempo_bpm"] == 95
    with wave.open(str(ROOT / "out" / "tab3_full_band_fea.wav"), "rb") as audio:
        assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (48_000, 2, 3)
        assert abs(audio.getnframes()/48_000 - 178.778947) < 1/48_000
