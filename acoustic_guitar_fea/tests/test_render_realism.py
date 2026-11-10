"""Regressions for dynamics, physical strings, and causal acoustic radiation."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from render_model import AcousticGuitar, GuitarBody  # noqa: E402

RATE = 12_000


@pytest.fixture
def body():
    path = ROOT / "model" / "body_modes.json"
    if not path.exists():
        pytest.skip("run tools/soundboard_fea.py first")
    return GuitarBody(RATE, path)


def guitar(seed=1946):
    return AcousticGuitar(sample_rate=RATE, technique="pick", seed=seed)


def energy(audio):
    return float(np.linalg.norm(audio))


def tab_note(midi, start, duration, string_index, velocity=85):
    open_midi = (40, 45, 50, 55, 59, 64)[string_index]
    return dict(midi=midi, velocity=velocity, start=start, duration=duration,
                string_index=string_index, fret=midi - open_midi)


def test_body_preserves_input_gain_and_superposition(body):
    rng = np.random.default_rng(321)
    a = rng.normal(0.0, 0.03, 4096)
    b = rng.normal(0.0, 0.02, 4096)
    response_a = body.process(a)
    response_b = body.process(b)
    assert energy(response_a) > 0.0
    np.testing.assert_allclose(body.process(0.15 * a), 0.15 * response_a,
                               rtol=2e-10, atol=1e-12)
    np.testing.assert_allclose(body.process(a + b), response_a + response_b,
                               rtol=2e-10, atol=1e-12)


def test_body_does_not_emit_sound_before_input(body):
    impulse = np.zeros(4096)
    impulse[2048] = 1.0
    rendered = body.process(impulse)
    assert rendered.shape == (4096, 2)
    assert np.count_nonzero(rendered[:2048]) == 0
    assert energy(rendered[2048:]) > 0.0


def test_upper_body_radiation_retains_treble_when_folded_to_mono():
    body = GuitarBody(48_000)
    impulse = np.zeros(8192)
    impulse[0] = 1.0
    spectrum = np.fft.rfft(body.upper_radiation(impulse), axis=0)
    frequencies = np.fft.rfftfreq(len(impulse), 1 / 48_000)
    treble = (frequencies >= 1000) & (frequencies <= 6000)
    stereo_power = np.mean(np.abs(spectrum[treble])**2, axis=1)
    mono_power = np.abs(spectrum[treble].mean(axis=1))**2
    # Close microphones must not cancel the attack's upper harmonics in mono.
    assert np.min(mono_power / stereo_power) > 0.85


def test_unmastered_note_and_render_preserve_performance_dynamics():
    soft = guitar().note_bridge_force(64, 35, 0.4, tail=0.15)
    loud = guitar().note_bridge_force(64, 105, 0.4, tail=0.15)
    assert energy(loud) > 3.0 * energy(soft)
    soft_mix = guitar().render([(64, 35, 0.0, 0.4)], tail=0.15,
                              normalize=False, room=False)
    loud_mix = guitar().render([(64, 105, 0.0, 0.4)], tail=0.15,
                              normalize=False, room=False)
    assert energy(loud_mix) > 3.0 * energy(soft_mix)


def test_same_seed_reproduces_audio_and_new_seed_varies_pluck():
    events = [tab_note(64, 0.0, 0.4, 5), tab_note(60, 0.1, 0.3, 4)]
    a = guitar(42).render(events, tail=0.1, normalize=False)
    b = guitar(42).render(events, tail=0.1, normalize=False)
    c = guitar(43).render(events, tail=0.1, normalize=False)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.isfinite(a).all()


@pytest.mark.parametrize("room", [False, True])
def test_delayed_note_has_exact_silence_before_onset(room):
    # Ending while the string still rings catches circular-delay wraparound.
    audio = guitar().render([(52, 100, 0.2, 0.4)], tail=0.0,
                            normalize=True, room=room)
    assert np.count_nonzero(audio[:int(0.2 * RATE)]) == 0
    assert energy(audio[int(0.2 * RATE):]) > 0.0


def test_string_assignment_changes_spectrum_at_the_same_pitch():
    open_e = guitar().note_bridge_force(64, 90, 0.5, tail=0.1,
                                       string_index=5)
    fretted_e = guitar().note_bridge_force(64, 90, 0.5, tail=0.1,
                                          string_index=2)
    # Discard the noisy attack; compare normalized sustain power spectra.
    start, end = int(0.04 * RATE), int(0.4 * RATE)
    spectra = []
    for note in (open_e, fretted_e):
        section = note[start:end] * np.hanning(end - start)
        spectrum = np.abs(np.fft.rfft(section)) ** 2
        spectra.append(spectrum / np.sum(spectrum))
    assert np.linalg.norm(spectra[0] - spectra[1]) > 0.04


def test_legato_slide_starts_near_source_pitch_and_reaches_destination():
    from scipy.signal import butter, hilbert, sosfiltfilt

    def pitch_in_window(note, start, end):
        # Isolate the fundamental from the pluck's higher string modes.
        fundamental = sosfiltfilt(butter(5, (240, 530), btype="bandpass",
                                       fs=RATE, output="sos"), note)
        phase = np.unwrap(np.angle(hilbert(fundamental)))
        selection = slice(round(start * RATE), round(end * RATE))
        t = np.arange(len(note)) / RATE
        return np.polyfit(t[selection], phase[selection], 1)[0] / (2 * np.pi)

    sliding = guitar().note_bridge_force(69, 90, 0.4, tail=0.1,
                                        string_index=5,
                                        articulation="legato_slide",
                                        source_midi=64)
    destination = guitar().note_bridge_force(69, 90, 0.4, tail=0.1,
                                            string_index=5,
                                            articulation="legato_slide",
                                            source_midi=69)
    early_slide = pitch_in_window(sliding, .015, .035)
    early_destination = pitch_in_window(destination, .015, .035)
    assert 325.0 < early_slide < 375.0
    assert early_destination - early_slide > 60.0
    assert pitch_in_window(sliding, .20, .30) == pytest.approx(440.0, rel=.01)


def test_same_string_retrigger_cuts_previous_sustain():
    # A third note fixes the output length for both event lists. The old first
    # note may ring acoustically after release, but cannot keep vibrating as a
    # separate physical string after that string is fretted/plucked again.
    sustained = [tab_note(64, 0.0, 1.4, 5),
                 tab_note(67, 0.2, 0.2, 5),
                 tab_note(52, 1.5, 0.1, 2)]
    explicitly_stopped = [dict(event) for event in sustained]
    explicitly_stopped[0]["duration"] = 0.2
    a = guitar().render(sustained, tail=0.1, normalize=False, room=False)
    b = guitar().render(explicitly_stopped, tail=0.1,
                        normalize=False, room=False)
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)


def test_tuple_events_keep_polyphony_without_tab_string_assignment():
    overlapping = [(64, 85, 0.0, 0.7), (67, 85, 0.2, 0.2),
                   (52, 85, 0.8, 0.1)]
    stopped = [(64, 85, 0.0, 0.2), *overlapping[1:]]
    a = guitar().render(overlapping, tail=0.1, normalize=False, room=False)
    b = guitar().render(stopped, tail=0.1, normalize=False, room=False)
    assert energy(a[int(0.5 * RATE):int(0.7 * RATE)]) > 2.0 * energy(
        b[int(0.5 * RATE):int(0.7 * RATE)]
    )


def test_empty_and_zero_velocity_inputs_remain_finite_silence(body):
    empty = guitar().render([], tail=0.0)
    assert empty.shape == (1, 2)
    assert np.count_nonzero(empty) == 0
    empty_note = guitar().note_bridge_force(64, 85, 0.0, tail=0.0)
    assert np.isfinite(empty_note).all()
    assert empty_note.size <= 1
    assert np.count_nonzero(empty_note) == 0
    silent = guitar().note_bridge_force(64, 0, 0.3, tail=0.1)
    assert np.isfinite(silent).all()
    assert np.count_nonzero(silent) == 0
    response = body.process(np.zeros(128))
    assert response.shape == (128, 2)
    assert np.count_nonzero(response) == 0
