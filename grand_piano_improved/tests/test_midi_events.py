"""Regression cases for lost strikes, channel pedals and hardware timing."""
from __future__ import annotations

import math
from pathlib import Path
import subprocess
import sys

import mido
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from midi_events import clip_events, hardware_events, load_events, load_performance


def midi_file(tmp_path, messages, *, tracks=None, midi_type=1):
    midi = mido.MidiFile(type=midi_type, ticks_per_beat=480)
    for messages_in_track in (tracks if tracks is not None else [messages]):
        midi.tracks.append(mido.MidiTrack(messages_in_track))
    path = tmp_path / "case.mid"
    midi.save(path)
    return path


def on(pitch=60, velocity=80, channel=0, time=0):
    return mido.Message("note_on", note=pitch, velocity=velocity, channel=channel, time=time)


def off(pitch=60, channel=0, time=0):
    return mido.Message("note_off", note=pitch, channel=channel, time=time)


def pedal(down, channel=0, time=0):
    return mido.Message("control_change", control=64, value=127 if down else 0,
                        channel=channel, time=time)


def test_channels_do_not_overwrite_same_pitch(tmp_path):
    path = midi_file(tmp_path, [on(velocity=70), on(velocity=100, channel=1, time=120),
                                off(time=120), off(channel=1, time=240)])
    assert load_events(path) == ([(60, 70, 0.0, 0.25), (60, 100, 0.125, 0.5)], [])


def test_repeated_overlapping_notes_are_paired_fifo(tmp_path):
    path = midi_file(tmp_path, [on(velocity=50), on(velocity=110, time=120),
                                off(time=120), off(time=240)])
    assert load_events(path)[0] == [(60, 50, 0.0, 0.25), (60, 110, 0.125, 0.5)]


def test_pedal_is_channel_local_and_spans_form_union(tmp_path):
    path = midi_file(tmp_path, [pedal(True), on(), on(channel=1), off(time=120),
                                off(channel=1), pedal(True, channel=1, time=120),
                                pedal(False, time=240), pedal(False, channel=1, time=240)])
    events, spans = load_events(path)
    assert [e[3] for e in events] == [0.5, 0.125]
    assert spans == [(0.0, 0.75)]


def test_pedal_lift_does_not_release_a_held_key(tmp_path):
    path = midi_file(tmp_path, [pedal(True), on(), pedal(False, time=240), off(time=240)])
    assert load_events(path) == ([(60, 80, 0.0, 0.5)], [(0.0, 0.25)])


def test_velocity_zero_is_a_release_and_events_keep_onset_order(tmp_path):
    path = midi_file(tmp_path, [on(48), on(72, time=120), on(72, velocity=0, time=120),
                                off(48, time=240)])
    assert load_events(path)[0] == [(48, 80, 0.0, 0.5), (72, 80, 0.125, 0.25)]


def test_tempo_map_and_multitrack_merge(tmp_path):
    path = midi_file(tmp_path, [], tracks=[
        [mido.MetaMessage("set_tempo", tempo=500000),
         mido.MetaMessage("set_tempo", tempo=1000000, time=480)],
        [on(), off(time=960)],
    ])
    assert load_events(path)[0] == [(60, 80, 0.0, 1.5)]


def test_dangling_notes_and_pedal_close_after_end_track_silence(tmp_path):
    path = midi_file(tmp_path, [off(70), pedal(True), on(),
                                mido.MetaMessage("end_of_track", time=960)])
    performance = load_performance(path)
    assert performance.events == [(60, 80, 0.0, 1.0)]
    assert performance.notes[0].key_release == 1.0
    assert performance.pedal_spans == ((0.0, 1.0),)
    assert len(performance.warnings) == 2


def test_sustain_all_overrides_pedal_up_on_every_channel(tmp_path):
    path = midi_file(tmp_path, [on(), on(channel=1), off(time=120), off(channel=1),
                                pedal(False), mido.MetaMessage("end_of_track", time=360)])
    performance = load_performance(path, sustain_all=True)
    assert [n.release for n in performance.notes] == [0.5, 0.5]
    assert performance.pedal_spans == ((0.0, 0.5),)
    assert hardware_events(performance) == [
        ("ped", 0.0, 1, 0), ("on", 0.0, 60, 80), ("on", 0.0, 60, 80),
        ("off", 0.125, 60, 0), ("ped", 0.5, 0, 0)]


@pytest.mark.parametrize("controller,release", [(120, 0.25), (123, 0.5)])
def test_all_notes_off_respects_pedal_but_all_sound_off_ends_notes(tmp_path, controller, release):
    path = midi_file(tmp_path, [pedal(True), on(),
                                mido.Message("control_change", control=controller, value=0, time=240),
                                pedal(False, time=240)])
    note = load_performance(path).notes[0]
    assert note.key_release == 0.25
    assert note.release == release


def test_reset_controllers_releases_pedal_but_preserves_held_keys(tmp_path):
    path = midi_file(tmp_path, [pedal(True), on(48), on(60), off(48, time=120),
                                mido.Message("control_change", control=121, value=0, time=120),
                                off(60, time=240)])
    assert [n.release for n in load_performance(path).notes] == [0.25, 0.5]


def test_hardware_off_occurs_at_key_release_not_pedal_lift(tmp_path):
    path = midi_file(tmp_path, [pedal(True), on(), off(time=240), pedal(False, time=240)])
    performance = load_performance(path)
    assert performance.notes[0].release == 0.5
    assert hardware_events(performance) == [
        ("ped", 0.0, 1, 0), ("on", 0.0, 60, 80),
        ("off", 0.25, 60, 0), ("ped", 0.5, 0, 0)]


def test_hardware_equal_timestamp_retrigger_preserves_source_order(tmp_path):
    path = midi_file(tmp_path, [on(), off(time=240), on(velocity=100), off(time=240)])
    assert hardware_events(load_performance(path))[1:3] == [
        ("off", 0.25, 60, 0), ("on", 0.25, 60, 100)]


def test_hardware_holds_pitch_until_last_channel_releases(tmp_path):
    path = midi_file(tmp_path, [on(), on(channel=1), off(time=120), off(channel=1, time=120)])
    assert hardware_events(load_performance(path)) == [
        ("on", 0.0, 60, 80), ("on", 0.0, 60, 80), ("off", 0.25, 60, 0)]


def test_hardware_clipping_closes_held_notes_and_pedal(tmp_path):
    path = midi_file(tmp_path, [pedal(True), on(), off(time=960), pedal(False)])
    performance = load_performance(path)
    assert hardware_events(performance, 0.25) == [
        ("ped", 0.0, 1, 0), ("on", 0.0, 60, 80),
        ("off", 0.25, 60, 0), ("ped", 0.25, 0, 0)]
    assert hardware_events(performance, 0.0) == []
    assert clip_events(performance.events, performance.pedal_spans, 0.25) == (
        [(60, 80, 0.0, 0.25)], [(0.0, 0.25)])
    assert clip_events(performance.events, performance.pedal_spans, 0.0) == ([], [])


def test_percussion_and_unplayable_notes_are_skipped_with_counts(tmp_path):
    path = midi_file(tmp_path, [pedal(True, channel=9), on(35, channel=9), on(12),
                                on(60), off(35, channel=9, time=480), off(12), off(60)])
    performance = load_performance(path)
    assert performance.events == [(60, 80, 0.0, 0.5)]
    assert performance.pedal_spans == ()
    assert len(performance.warnings) == 2
    assert "percussion" in performance.warnings[0]
    assert "outside piano" in performance.warnings[1]


@pytest.mark.parametrize("seconds", [-1.0, math.inf, math.nan])
def test_invalid_clip_duration_is_rejected(seconds):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        clip_events([], [], seconds)


def test_bad_midi_has_contextual_error(tmp_path):
    path = tmp_path / "broken.mid"
    path.write_bytes(b"not a midi file")
    with pytest.raises(ValueError, match="Cannot read MIDI"):
        load_events(path)


def test_type_two_is_rejected_explicitly(tmp_path):
    path = midi_file(tmp_path, [on(), off(time=480)], midi_type=2)
    with pytest.raises(ValueError, match="type 2"):
        load_events(path)


def test_export_cli_preserves_raw_sustain_and_empty_clips(tmp_path):
    path = midi_file(tmp_path, [pedal(True), on(), off(time=240), pedal(False, time=240)])
    output = tmp_path / "events.txt"
    command = [sys.executable, str(ROOT / "tools" / "export_events.py"),
               "--midi", str(path), "--out", str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert output.read_text().splitlines()[2] == "off 0.250000000 60 0"
    result = subprocess.run(command + ["--seconds", "0"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert output.read_text() == ""


@pytest.mark.parametrize("number", range(9, 16))
def test_supplied_midi_retains_every_strike(number):
    path = ROOT / "midi" / f"demo{number}.mid"
    if not path.exists():
        pytest.skip("optional supplied performance is absent")
    midi = mido.MidiFile(path)
    expected = sum(msg.type == "note_on" and msg.velocity > 0 and msg.channel != 9
                   and 21 <= msg.note <= 108 for msg in midi)
    performance = load_performance(path)
    assert len(performance.notes) == expected
    assert all(n.start <= n.key_release <= n.release <= performance.duration
               for n in performance.notes)
    assert [n.start for n in performance.notes] == sorted(n.start for n in performance.notes)
    transitions = hardware_events(performance, seconds=15)
    assert [e[1] for e in transitions] == sorted(e[1] for e in transitions)
    assert all(e[1] <= 15 for e in transitions)
