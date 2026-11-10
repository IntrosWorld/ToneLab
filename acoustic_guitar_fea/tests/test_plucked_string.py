"""Audible invariants of the traveling-wave string."""
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.signal import butter, hilbert, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from plucked_string import render_plucked_string


def pitch(signal, rate, target):
    filtered = sosfiltfilt(butter(4, (target * .85, target * 1.15),
                                fs=rate, btype="bandpass", output="sos"), signal)
    phase = np.unwrap(np.angle(hilbert(filtered)))
    start, end = round(.12 * rate), round(.7 * rate)
    return np.polyfit(np.arange(start, end) / rate, phase[start:end], 1)[0] / (2 * np.pi)


@pytest.mark.parametrize("rate,freq,stiffness", [(12000,82.4069,.0001),
    (48000,164.8138,.0002), (48000,622.254,.0004), (12000,622.254,.0004)])
def test_string_is_tuned_including_loss_and_dispersion(rate, freq, stiffness):
    signal = render_plucked_string(freq, 1., rate, stiffness=stiffness)
    measured = pitch(signal, rate, freq)
    assert abs(1200 * np.log2(measured / freq)) < 3.


def test_pluck_reproducibility_and_performance_dynamics():
    quiet = render_plucked_string(164.8138, .5, velocity=35, seed=5)
    loud = render_plucked_string(164.8138, .5, velocity=105, seed=5)
    np.testing.assert_array_equal(loud, render_plucked_string(164.8138, .5, velocity=105, seed=5))
    assert np.linalg.norm(loud) > 3 * np.linalg.norm(quiet)
    assert not np.array_equal(loud, render_plucked_string(164.8138, .5, velocity=105, seed=6))


def test_high_partial_energy_decays_faster_than_fundamental():
    rate = 48000
    signal = render_plucked_string(164.8138, 2., rate)
    low = sosfiltfilt(butter(4, (145.,190.), fs=rate, btype="bandpass", output="sos"), signal)
    high = sosfiltfilt(butter(4, (1800.,6500.), fs=rate, btype="bandpass", output="sos"), signal)
    # Fit decay over several sustain windows, away from the contact impulse.
    # A fixed 5x early/late ratio required unnecessarily dead upper partials.
    centers = np.linspace(.15, .9, 8)
    def decay_slope(band):
        levels = []
        for center in centers:
            part = band[round((center - .04) * rate):round((center + .04) * rate)]
            levels.append(20 * np.log10(np.sqrt(np.mean(part * part))))
        return np.polyfit(centers, levels, 1)[0]
    fundamental_slope = decay_slope(low)
    upper_slope = decay_slope(high)
    assert fundamental_slope < 0
    assert upper_slope < fundamental_slope - 3.0


def test_release_damps_without_truncating_or_producing_noise():
    signal = render_plucked_string(164.8138, .3, 12000, tail=.3, release_t60=.08)
    assert np.isfinite(signal).all()
    assert np.linalg.norm(signal[6600:]) < 1e-7 * np.linalg.norm(signal[2400:3000])
    assert not np.count_nonzero(render_plucked_string(164.8138,.3,velocity=0))


def test_frequency_control_changes_pitch_and_preserves_output_time():
    rate = 12000
    ratio = np.full(round(1.17 * rate), 2.**(3/12))
    signal = render_plucked_string(164.8138, 1., rate, frequency_ratio=ratio)
    assert len(signal) == len(ratio)
    assert abs(1200 * np.log2(pitch(signal, rate, 196.0) / (164.8138 * ratio[0]))) < 3.


def test_polarization_decay_scale_preserves_attack_and_changes_sustain():
    main = render_plucked_string(164.8138, 2., 12000, decay_scale=1.)
    other = render_plucked_string(164.8138, 2., 12000, decay_scale=1.4)
    # Before the first round trip, loss cannot affect the contact packet.
    np.testing.assert_array_equal(main[:55], other[:55])
    assert np.linalg.norm(other[18000:24000]) > 1.3 * np.linalg.norm(main[18000:24000])
