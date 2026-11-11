"""Audio behavior and numeric regression tests for the improved renderer."""
from pathlib import Path
import sys
import wave

import numpy as np
import pytest
from scipy.signal import lfilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from audio_dsp import allpass, damped_comb, causal_smooth, pedal_release_envelope
from audio_io import write_wav
from render_model import GrandPiano, RenderCancelled


@pytest.mark.parametrize("length,delay", [(1, 5), (23, 11), (10000, 113)])
def test_fast_room_matches_dense_reference(length, delay):
    source = np.random.default_rng(29).normal(size=length)
    feedback, damping = .86, .18
    denominator = np.zeros(delay + 1)
    denominator[0], denominator[1], denominator[delay] = 1, -damping, -feedback * (1 - damping)
    expected = lfilter([1, -damping], denominator, source)
    np.testing.assert_allclose(damped_comb(source, delay, feedback, damping), expected, atol=1e-13)
    numerator = np.zeros(delay + 1)
    denominator = np.zeros(delay + 1)
    numerator[0], numerator[delay] = -.5, 1
    denominator[0], denominator[delay] = 1, -.5
    np.testing.assert_allclose(allpass(source, delay), lfilter(numerator, denominator, source), atol=1e-13)


def test_pedal_gate_does_not_anticipate_events():
    gate = np.zeros(40)
    gate[20:30] = 1
    smoothed = causal_smooth(gate, 8)
    assert np.all(smoothed[:20] == 0)
    assert smoothed[27] == 1
    assert smoothed[37] == 0


def test_sympathetic_damper_releases_and_recatches():
    gate = pedal_release_envelope(2000, 1000, [(0.1, .5), (1.2, 1.4)], .2)
    assert np.all(gate[:100] == 0)
    assert gate[300] == 1
    assert gate[700] == pytest.approx(.001)
    assert gate[1100] < 1e-8
    assert gate[1200] < 1e-9
    assert gate[1250] == 1


@pytest.mark.parametrize("velocity", [7, 8, 15, 16, 31, 32, 63, 64, 95, 96, 119, 120, 127])
def test_velocity_preserves_requested_strike_level(velocity):
    piano = GrandPiano(8000, room_mix=0, body_mix=0, normalize=False)
    piano.SYMPATHETIC_LEVEL = 0
    audio = piano.render([(60, velocity, 0, .4)], tail=.2)
    # Recover mono gain from pickup maximum and fixed master gain.
    from piano_physics import stereo_arrival
    gain = max(stereo_arrival(60)[2:])
    assert np.max(np.abs(audio)) == pytest.approx(piano._note_loudness(60, velocity) * gain * .25, rel=2e-5)


def test_render_repeatability_tail_and_cache_budget():
    piano = GrandPiano(8000, room_mix=.2, max_cache_mb=.02)
    piano.SYMPATHETIC_LEVEL = 0
    events = [(60, 90, 0, .25), (64, 64, .1, .3)]
    first = piano.render(events, tail=.2)
    second = piano.render(events, tail=.2)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (4001, 2)
    assert np.isfinite(first).all()
    assert np.max(np.abs(first)) <= .89 + 1e-12
    assert np.all(first[-1] == 0)
    assert piano.cache_bytes <= piano.max_cache_bytes


def test_template_prefix_does_not_depend_on_requested_sustain():
    piano = GrandPiano(8000)
    short = piano.note_template(60, 5, .5).copy()
    longer = piano.note_template(60, 5, 1.5)
    np.testing.assert_allclose(short, longer[:len(short)], atol=1e-7)


def test_zero_duration_note_is_damped():
    piano = GrandPiano(8000, room_mix=0, body_mix=0, normalize=False)
    piano.SYMPATHETIC_LEVEL = 0
    released = piano.render([(60, 96, 0, 0)], tail=1)
    held = piano.render([(60, 96, 0, .8)], tail=.2)
    assert np.linalg.norm(released[4000:6000]) < np.linalg.norm(held[4000:6000]) * .01


def test_invalid_events_and_cancellation_fail_cleanly():
    piano = GrandPiano(8000)
    for event in [(60, 80, -1, 1), (60, 80, 1, 0), (0, 80, 0, 1), (60, 0, 0, 1)]:
        with pytest.raises(ValueError):
            piano.render([event])
    with pytest.raises(ValueError):
        piano.render([], tail=-1)
    piano.cancel = lambda: True
    with pytest.raises(RenderCancelled):
        piano.render([(60, 80, 0, 1)])


def test_wav_exact_24bit_packing_and_failed_export_preserves_file(tmp_path):
    path = tmp_path / "roundtrip.wav"
    audio = np.array([[-1, 1], [0, .5], [-.25, 0]], dtype=float)
    write_wav(path, audio, 48000)
    before = path.read_bytes()
    with wave.open(str(path)) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()) == (2, 3, 48000, 3)
        packed = np.frombuffer(wav.readframes(3), dtype=np.uint8).reshape(-1, 3).astype(np.int32)
    decoded = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
    decoded = np.where(decoded & 0x800000, decoded - 0x1000000, decoded)
    np.testing.assert_array_equal(decoded, np.rint(audio.ravel() * 8388607).astype(np.int32))
    with pytest.raises(ValueError, match="non-finite"):
        write_wav(path, np.array([[np.nan, 0]]), 48000)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))
