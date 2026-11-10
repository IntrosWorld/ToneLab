"""Protect notation while interpreting the physical timing of chord strokes."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from guitar_performance import prepare_performance_events  # noqa: E402
from gp_score import load_gp_score  # noqa: E402


def _event(string, start=0., duration=1., **extra):
    return dict(string_index=string, start=start, duration=duration, midi=60 + string,
                velocity=108, voice=0, articulation="pick", **extra)


def test_legato_grid_rests_and_source_data_survive_preparation():
    events = [_event(4, start=1., duration=.25),
              _event(4, start=1.25, duration=.125),
              _event(4, start=1.375, duration=.125),
              _event(5, start=3., duration=.5)]
    events[1]["articulation"] = "pull_off"
    events[1]["source_midi"] = 66
    before = deepcopy(events)
    rendered = prepare_performance_events(events, "lead", 1946)
    assert events == before
    for source, actual in zip(events, rendered):
        for key in ("midi", "start", "duration", "string_index", "articulation"):
            assert actual[key] == source[key]
        assert abs(actual["velocity"] - source["velocity"]) <= 8
    assert rendered[1]["source_midi"] == 66
    assert rendered[0]["phrase_index"] == rendered[2]["phrase_index"]
    assert rendered[-1]["phrase_index"] != rendered[0]["phrase_index"]


def test_physical_strum_direction_and_string_gaps_preserve_note_ends():
    chord = [_event(i, start=2., duration=.5, stroke_direction="up")
             for i in (0, 2, 5)]
    rendered = prepare_performance_events(chord, "rhythm", 7251)
    assert [e["string_index"] for e in rendered] == [5, 2, 0]
    assert rendered[0]["start"] == 2.
    assert 0 < rendered[-1]["start"] - 2. <= .05
    # Three crossed string spacings take longer than the remaining two.
    assert rendered[1]["start"] - rendered[0]["start"] > (
        rendered[2]["start"] - rendered[1]["start"])
    for event in rendered:
        assert event["start"] + event["duration"] == pytest.approx(2.5)
        assert event["score_start"] == 2.
        assert event["score_duration"] == .5


def test_short_notes_and_repeated_downstrokes_are_not_recomposed():
    chords = [_event(i, start=t, duration=.02) for t in (0., .5) for i in range(6)]
    output = prepare_performance_events(chords, "rhythm", 21)
    assert [e["string_index"] for e in output] == list(range(6)) * 2
    assert all(e["duration"] > 0. for e in output)
    assert all(e["stroke_direction"] == "down" for e in output)
    assert max(e["stroke_delay"] for e in output) <= .004
    assert output == prepare_performance_events(chords, "rhythm", 21)
    assert output != prepare_performance_events(chords, "rhythm", 22)
    assert prepare_performance_events([], "lead", 1) == []


def test_assamj_interpretation_retains_selected_attacks_and_silent_regions():
    path = Path("C:/Users/Micro/Downloads/assamj.gp")
    if not path.exists():
        pytest.skip("user-provided score is not installed")
    score = load_gp_score(path)
    for track, role, seed in zip(score["tracks"], ("lead", "rhythm"), (1946, 7251)):
        events = prepare_performance_events(track["events"], role, seed)
        assert len(events) == track["sounding_events"]
        source_pitches = sorted((e["start"], e["midi"], e["string_index"])
                                for e in track["events"])
        actual_pitches = sorted((e["score_start"], e["midi"], e["string_index"])
                                for e in events)
        assert source_pitches == actual_pitches
        assert all(e["start"] >= e["score_start"] for e in events)
        assert all(e["start"] + e["duration"] <= 30. + 1e-8 for e in events)
        if role == "lead":
            assert min(e["start"] for e in events) == pytest.approx(8 * 60 / 107)
            assert sum(e["articulation"] in ("hammer_on", "pull_off") for e in events) == 6
        else:
            assert not any(8.972 <= e["start"] < 17.943 for e in events)
        assert np.ptp([e["velocity"] for e in events]) >= 4
