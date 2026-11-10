"""A tuned, lossy traveling-wave steel string, excited only at the pluck.

The initial bridge-slope wave is recirculated through a fractional-delay string,
with small contact irregularities. Loop losses progressively remove upper
partials. A dispersive allpass can approximate the physical string's stiffness.
The fixed excitation gain preserves MIDI dynamics between separate calls.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import brentq
from scipy.signal import butter, sosfilt

try:
    from numba import njit
except ImportError:  # Pure Python remains functional for small standalone uses.
    def njit(*args, **kwargs):
        return lambda fn: fn


@njit(cache=True)
def _circulate(excitation, count, delay, fractional_a, dispersion_a, loss, gain):
    output = np.zeros(count, dtype=np.float64)
    previous_delayed = 0.0
    dispersive_input = 0.0
    dispersive_output = 0.0
    fractional_input = 0.0
    fractional_output = 0.0
    for frame in range(count):
        delayed = output[frame - delay] if frame >= delay else 0.0
        filtered = (1.0 - loss) * delayed + loss * previous_delayed
        previous_delayed = delayed
        dispersed = dispersion_a * filtered + dispersive_input - dispersion_a * dispersive_output
        dispersive_input = filtered
        dispersive_output = dispersed
        fractional = fractional_a * dispersed + fractional_input - fractional_a * fractional_output
        fractional_input = dispersed
        fractional_output = fractional
        output[frame] = gain * fractional
        if frame < len(excitation):
            output[frame] += excitation[frame]
    return output


def _allpass_phase(omega, delay):
    return -2.0 * np.arctan(delay * np.tan(omega / 2.0))


def _loop_design(freq_hz, sample_rate, stiffness, loss, fundamental_t60):
    """Tune the actual loop phase at f0, including both allpass sections."""
    period = sample_rate / freq_hz
    omega = 2.0 * math.pi / period
    loss_phase = float(np.angle((1.0 - loss) + loss * np.exp(-1j * omega)))

    def design(dispersive_delay):
        ap_phase = float(_allpass_phase(omega, dispersive_delay))
        remaining = period + (loss_phase + ap_phase) / omega
        integer = max(1, int(math.floor(remaining)))
        remainder = remaining - integer
        frac_delay = math.tan(omega * remainder / 2.0) / math.tan(omega / 2.0)
        return integer, (1.0 - frac_delay) / (1.0 + frac_delay)

    # B is the stiff-string coefficient; match the first two resonances. A
    # single allpass is an approximation at the top of the audible spectrum.
    dispersion_delay = 1.0
    if stiffness > 0.0:
        target_second = 2.0 * freq_hz * math.sqrt((1.0 + 4.0 * stiffness) / (1.0 + stiffness))
        second_omega = 2.0 * math.pi * target_second / sample_rate

        def detuning(d):
            integer, a = design(d)
            second_loss_phase = float(np.angle((1.0 - loss) + loss * np.exp(-1j * second_omega)))
            fractional_delay = (1.0 - a) / (1.0 + a)
            phase = (-integer * second_omega + second_loss_phase
                     + _allpass_phase(second_omega, d)
                     + _allpass_phase(second_omega, fractional_delay))
            return float(phase + 4.0 * math.pi)

        maximum = min(32.0, .28 * period)
        if maximum > 1.0 and detuning(1.0) * detuning(maximum) < 0.0:
            dispersion_delay = brentq(detuning, 1.0, maximum, xtol=1e-10)
    integer, fractional_a = design(dispersion_delay)
    dispersion_a = (1.0 - dispersion_delay) / (1.0 + dispersion_delay)
    fundamental_loss = abs((1.0 - loss) + loss * np.exp(-1j * omega))
    # Never boost the loop above unity, even on high strings at a low rate.
    gain = min(.99998, math.exp(-math.log(1000.) / (freq_hz * fundamental_t60)) / fundamental_loss)
    return integer, fractional_a, dispersion_a, gain


def render_plucked_string(freq_hz, duration, sample_rate=48_000, velocity=96,
                          pluck_fraction=.18, *, string_index=2, seed=1946,
                          release_t60=.10, tail=.17, fret=0, stiffness=0.,
                          legato=False, frequency_ratio=None, decay_scale=1.0,
                          brightness=1.0):
    """Return mono bridge force; no body, room, peak normalization or samples.

    ``frequency_ratio`` is an optional per-output-sample instantaneous pitch
    ratio (for slides, bends and pitch relaxation). It is integrated before
    resampling the string; left-hand damping remains in output time.
    """
    if sample_rate < 8000 or not 20.0 <= freq_hz < .20 * sample_rate:
        raise ValueError("sample rate/pitch outside supported string range")
    if duration < 0.0 or tail < 0.0 or release_t60 <= 0.0:
        raise ValueError("duration and tail must be nonnegative; release_t60 positive")
    if not .02 <= pluck_fraction <= .48:
        raise ValueError("pluck_fraction must lie between .02 and .48")
    if not 0 <= string_index <= 5 or stiffness < 0:
        raise ValueError("string index must be 0..5 and stiffness nonnegative")
    if not math.isfinite(decay_scale) or decay_scale <= 0:
        raise ValueError("decay_scale must be finite and positive")
    if not math.isfinite(brightness) or brightness <= 0:
        raise ValueError("brightness must be finite and positive")
    count = max(1, round((duration + tail) * sample_rate))
    if velocity <= 0 or duration == 0:
        return np.zeros(count)
    velocity_norm = float(np.clip(velocity / 127.0, 0.0, 1.0))
    coordinates = None
    synthesis_count = count
    maximum_ratio = 1.0
    if frequency_ratio is not None:
        ratios = np.asarray(frequency_ratio, dtype=np.float64)
        if ratios.shape != (count,) or not np.isfinite(ratios).all() or np.any(ratios <= 0):
            raise ValueError("frequency_ratio must contain one finite positive ratio per output sample")
        coordinates = np.concatenate(([0.0], np.cumsum(ratios[:-1])))
        synthesis_count = max(count, int(math.ceil(coordinates[-1])) + 4)
        maximum_ratio = max(1.0, float(np.max(ratios)))

    period = sample_rate / freq_hz
    rng = np.random.default_rng(seed)
    # Initialize one complete traveling bridge-force wave. The derivative of
    # a triangular displacement has signed sin(n*pi*p)/n coefficients. The
    # former noise-packet/differencing comb omitted 1/n, hollowing out bass
    # and low harmonics while emphasizing a short, synthetic-sounding click.
    excitation_length = max(8, math.ceil(period))
    time = np.arange(excitation_length) / sample_rate
    upper_limit = min(12000., .40 * sample_rate / maximum_ratio)
    modes = np.arange(1, max(2, min(96, int(upper_limit / freq_hz) + 1)))
    contact_width = .008 / brightness
    amplitudes = .50 * np.sin(np.pi * modes * pluck_fraction)
    amplitudes /= np.pi * modes * pluck_fraction * (1. - pluck_fraction)
    cutoff = (2800. + 5700. * velocity_norm) * brightness * (.72 if legato else 1.)
    amplitudes *= np.sinc(modes * contact_width) * np.exp(-(modes * freq_hz / cutoff)**1.25)
    # Small smooth variations of contact, not random removal of entire partials.
    variation = rng.normal(0., .035, len(modes))
    variation = np.convolve(np.pad(variation, (1, 1), mode="edge"), [.25, .50, .25], mode="valid")
    amplitudes *= 1. + variation
    phases = rng.normal(0., .018, len(modes)) * np.minimum(modes / 5., 1.)
    excitation = np.sum(amplitudes[:, None] * np.cos(
        2. * np.pi * modes[:, None] * freq_hz * time[None, :] + phases[:, None]), axis=0)
    # A tiny finite contact component roughens the edge without dominating the
    # sustained spectrum or becoming an independent noise bed.
    contact_noise = rng.standard_normal(excitation_length)
    contact_noise = sosfilt(butter(1, (700., min(5400., .38 * sample_rate)),
                                  btype="bandpass", fs=sample_rate, output="sos"), contact_noise)
    excitation += (.004 if legato else .012) * brightness * contact_noise
    excitation -= np.mean(excitation)
    excitation *= velocity_norm**1.45 * (.78 if legato else 1.)

    reference_loss = .43 if string_index < 4 else .36
    # Preserve low/mid-frequency damping at lower export/test rates: a fixed
    # averaging coefficient otherwise kills harmonics 16x faster at 12 kHz.
    # A high note makes more round trips per second. Compensate for that so
    # loss at (say) 3 kHz does not become several times faster just because
    # the same string is fretted higher. The previous loop became a dull,
    # short pluck on upper frets even while bass harmonics rang too long.
    variance = min(.2499, reference_loss * (1.0 - reference_loss)
                   * (165. / freq_hz) * (sample_rate / 48000.)**2 / decay_scale)
    loss = .5 * (1.0 - math.sqrt(1.0 - 4.0 * variance))
    fundamental_t60 = float(np.clip(7.0 * (165.0 / freq_hz)**.35 / (1.0 + .016 * fret), 1.5, 10.0)) * decay_scale
    delay, fractional_a, dispersion_a, gain = _loop_design(
        freq_hz, sample_rate, stiffness, loss, fundamental_t60)
    force = _circulate(excitation, synthesis_count, delay, fractional_a, dispersion_a, loss, gain)
    if coordinates is not None:
        force = map_coordinates(force, [coordinates], order=3, mode="constant", prefilter=True)
    else:
        force = force[:count]
    # Smooth onset and damp the physical string after note-off. No independent
    # broadband bed or continued attack noise is added during sustain.
    attack = min(count, max(2, round(.00035 * sample_rate)))
    force[:attack] *= .5 - .5 * np.cos(math.pi * np.arange(attack) / attack)
    release = min(count, round(duration * sample_rate))
    if release < count:
        force[release:] *= np.exp(-math.log(1000.) * np.arange(count - release) / (sample_rate * release_t60))
    return force
