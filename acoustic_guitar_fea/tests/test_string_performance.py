"""Behavioral checks for one continuously vibrating bank per guitar string."""
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.signal import butter, hilbert, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from string_performance import _pitch_ratios, render_string_performance


def note(midi=64, start=0., duration=.8, velocity=96, string_index=5, **kwargs):
    return dict(midi=midi, start=start, duration=duration, velocity=velocity,
                string_index=string_index, **kwargs)


def measured_pitch(signal, rate, target, start, end):
    filtered = sosfiltfilt(butter(4, (target * .85, target * 1.15),
                                fs=rate, btype="bandpass", output="sos"), signal)
    phase = np.unwrap(np.angle(hilbert(filtered)))
    left, right = round(start * rate), round(end * rate)
    return np.polyfit(np.arange(left, right) / rate, phase[left:right], 1)[0] / (2. * np.pi)


@pytest.mark.parametrize("midi,idx,rate", [(40, 0, 12000), (52, 2, 48000), (75, 5, 12000)])
def test_pitch_is_exact_after_settling(midi, idx, rate):
    signal = render_string_performance([note(midi, string_index=idx, duration=1.)], rate, polarization=False)
    target = 440. * 2. ** ((midi - 69) / 12.)
    pitch = measured_pitch(signal, rate, target, .18, .72)
    assert abs(1200. * np.log2(pitch / target)) < 1.5


def test_exact_length_silent_prefix_and_repeatable_seed():
    events = [note(start=.15, duration=.25)]
    signal = render_string_performance(events, 12000, duration=1., seed=52)
    assert signal.shape == (12000,)
    assert np.count_nonzero(signal[:1800]) == 0
    assert np.isfinite(signal).all()
    np.testing.assert_array_equal(signal, render_string_performance(events, 12000, duration=1., seed=52))
    assert not np.array_equal(signal, render_string_performance(events, 12000, duration=1., seed=53))


def test_hammer_inherits_previous_pluck_energy_and_retunes():
    rate = 12000
    follow = note(66, start=.40, duration=.65, articulation="hammer_on", source_midi=64)
    quiet = render_string_performance([note(duration=.4, velocity=32), follow], rate, polarization=False)
    loud, diagnostics = render_string_performance([note(duration=.4, velocity=110), follow], rate,
                                                   polarization=False, return_diagnostics=True)
    # A separate replacement note would have identical second-note energy.
    assert np.linalg.norm(loud[6000:11000]) > 3. * np.linalg.norm(quiet[6000:11000])
    assert diagnostics["events"][1]["inherits_state"]
    assert diagnostics["events"][1]["excitation_energy"] < .03 * diagnostics["events"][0]["excitation_energy"]
    target = 440. * 2. ** ((66 - 69) / 12.)
    pitch = measured_pitch(loud, rate, target, .55, .91)
    assert abs(1200. * np.log2(pitch / target)) < 1.5


def test_legato_contact_starts_from_identical_instantaneous_vibration():
    rate = 12000
    split = .4
    sustained = render_string_performance([note(duration=.9)], rate, polarization=False)
    hammer = render_string_performance([note(duration=split),
        note(66, start=split, duration=.5, articulation="hammer_on", source_midi=64)], rate, polarization=False)
    # At the contact boundary, output still comes from the preceding state.
    np.testing.assert_allclose(hammer[:round(split * rate) + 1], sustained[:round(split * rate) + 1],
                               rtol=0., atol=2e-15)


def test_different_strings_remain_independent():
    options = dict(sample_rate=12000, duration=.9, seed=90)
    first = note(52, string_index=2, duration=.5)
    second = note(67, start=.15, duration=.5)
    together = render_string_performance([first, second], **options)
    apart = render_string_performance([first], **options) + render_string_performance([second], **options)
    np.testing.assert_array_equal(together, apart)


def test_release_obeys_score_off_even_when_next_note_is_later():
    rate = 12000
    events = [note(duration=.15), note(67, start=.75, duration=.15)]
    signal = render_string_performance(events, rate, duration=1.2)
    assert np.linalg.norm(signal[6000:8400]) < 1e-7 * np.linalg.norm(signal[600:1500])
    assert np.linalg.norm(signal[9600:10300]) > .01


def test_overlapping_restrikes_do_not_accumulate_old_voices():
    rate = 12000
    events = [note(start=.012 * index, duration=.6) for index in range(30)]
    signal = render_string_performance(events, rate, duration=.7)
    single = render_string_performance([note(duration=.6)], rate, duration=.7)
    assert np.max(np.abs(signal)) < 2.2 * np.max(np.abs(single))


def test_bend_reaches_notated_pitch():
    rate = 12000
    signal = render_string_performance([note(duration=1.1, articulation="bend",
        bend_points=[[0., 0.], [.4, 2.], [1., 2.]])], rate, polarization=False)
    target = 440. * 2. ** ((66 - 69) / 12.)
    pitch = measured_pitch(signal, rate, target, .6, .9)
    assert abs(1200. * np.log2(pitch / target)) < 1.5


def test_standalone_legato_slide_uses_explicit_source_pitch():
    event = note(69, duration=.8, articulation="legato_slide", source_midi=64)
    ratios = _pitch_ratios(event, 12000, 12000, None, False, False)
    assert ratios[0] == pytest.approx(2. ** (-5. / 12.))
    assert ratios[1200] == pytest.approx(1.)


def test_finger_release_does_not_cancel_high_string_partials():
    rate = 48000
    event = note(80, duration=.8, velocity=85)
    signal = render_string_performance([event], rate)
    section = signal[2400:7200]
    power = np.abs(np.fft.rfft(section * np.hanning(len(section)))) ** 2
    frequencies = np.fft.rfftfreq(len(section), 1. / rate)
    fundamental = 440. * 2. ** ((80 - 69) / 12.)
    # Reject the extra spectral cancellation produced by fixed-phase modal
    # injection over a finite release pulse, especially on the high E string.
    assert power[frequencies > 2.5 * fundamental].sum() / power.sum() > .025


def test_input_validation():
    with pytest.raises(ValueError, match="string/fret"):
        render_string_performance([note(fret=2)])
    with pytest.raises(ValueError, match="nonnegative"):
        render_string_performance([note(start=-1.)])
    with pytest.raises(ValueError, match="bend_points"):
        render_string_performance([note(bend_points=[[1., 0.], [0., 1.]])])
    assert not np.count_nonzero(render_string_performance([], 12000, duration=.1))


def test_body_loading_is_passive_and_preserves_tuning():
    rate = 12000
    events = [note(duration=1.)]
    options = dict(sample_rate=rate, polarization=False)
    dry = render_string_performance(events, **options)
    loaded = render_string_performance(events, body_loss=lambda frequencies, idx: np.full_like(frequencies, 2.), **options)
    assert np.linalg.norm(loaded[6000:10000]) < .5 * np.linalg.norm(dry[6000:10000])
    target = 440. * 2. ** ((64 - 69) / 12.)
    shifted = render_string_performance(events,
        frequency_shift=lambda frequencies, idx: -1. - .0001 * frequencies, **options)
    assert abs(1200. * np.log2(measured_pitch(shifted, rate, target, .18, .72) / target)) < 1.5
    with pytest.raises(ValueError, match="nonnegative decay"):
        render_string_performance(events, body_loss=lambda frequencies, idx: -1., **options)
