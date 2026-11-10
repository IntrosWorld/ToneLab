"""Delay-line filters with work proportional to audio length.

Delay-sized blocks have no feedback dependency within the block. SciPy steps
the one-pole damping filter, while NumPy handles the delayed feedback. This
avoids treating a mostly-zero, 1,000-tap denominator as a dense IIR filter.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter


def damped_comb(source, delay, feedback, damping):
    """y[n] = x[n] + feedback * lowpass(y[n-delay])."""
    if delay < 1 or not 0 <= feedback < 1 or not 0 <= damping < 1:
        raise ValueError("invalid comb delay, feedback, or damping")
    source = np.asarray(source, dtype=np.float64)
    out = source.copy()
    state = np.zeros(1)
    for begin in range(delay, len(source), delay):
        end = min(begin + delay, len(source))
        filtered, state = lfilter(
            [1.0 - damping], [1.0, -damping],
            out[begin - delay:end - delay], zi=state,
        )
        out[begin:end] += feedback * filtered
    return out


def allpass(source, delay, gain=0.5):
    """H(z) = (-gain + z^-delay) / (1 - gain*z^-delay)."""
    if delay < 1 or not abs(gain) < 1:
        raise ValueError("invalid allpass delay or gain")
    source = np.asarray(source, dtype=np.float64)
    out = -gain * source
    for begin in range(delay, len(source), delay):
        end = min(begin + delay, len(source))
        out[begin:end] += (source[begin - delay:end - delay]
                           + gain * out[begin - delay:end - delay])
    return out


def causal_smooth(values, samples):
    """Trailing moving average; transitions never anticipate a MIDI event."""
    values = np.asarray(values, dtype=np.float64)
    samples = max(1, int(samples))
    sums = np.concatenate(([0.0], np.cumsum(values)))
    result = sums[1:].copy()
    if samples < len(values):
        result[samples:] -= sums[1:-samples]
    return result / samples


def pedal_release_envelope(length, sample_rate, spans, t60):
    """Unity under pedal, felt damping after a lift, with a short catch ramp."""
    envelope = np.zeros(length, dtype=np.float64)
    cursor = 0
    level = 0.0
    decay = np.log(1000.0) / max(t60 * sample_rate, 1.0)
    for down, up in sorted(spans):
        begin = max(cursor, min(length, int(down * sample_rate)))
        end = max(begin, min(length, int(up * sample_rate)))
        if begin > cursor:
            envelope[cursor:begin] = level * np.exp(-decay * np.arange(begin - cursor))
            level *= np.exp(-decay * (begin - cursor))
        if end > begin:
            envelope[begin:end] = 1.0
            ramp = min(end - begin, max(2, int(0.005 * sample_rate)))
            envelope[begin:begin + ramp] = np.linspace(level, 1.0, ramp)
            level = 1.0
        cursor = end
    if cursor < length:
        envelope[cursor:] = level * np.exp(-decay * np.arange(length - cursor))
    return envelope


def pedal_resonator(source, sample_rate, spans, frequency, t60, damper_t60, gain=1.0):
    """A two-pole string whose *state* decays faster while the pedal is up.

    Filter whole pedal intervals with SciPy, rather than looping over samples.
    At each pole change reconstruct the filter delay state from its last two
    outputs. Multiplying an undamped output by an envelope instead would leave
    hidden energy in the filter and resurrect old notes on the next pedal-down.
    ``gain`` retains the undamped resonator's excitation calibration.
    """
    source = np.asarray(source, dtype=np.float64)
    if (source.ndim != 1 or not np.isfinite([sample_rate, frequency, t60, damper_t60, gain]).all()
            or sample_rate <= 0 or not 0 < frequency < sample_rate / 2
            or t60 <= 0 or damper_t60 <= 0):
        raise ValueError("invalid pedal resonator parameters")
    length = len(source)
    out = np.zeros(length, dtype=np.float64)
    intervals = []
    for down, up in sorted(spans):
        if not np.isfinite([down, up]).all() or down < 0 or up < down:
            raise ValueError("invalid pedal interval")
        begin, end = min(length, int(down * sample_rate)), min(length, int(up * sample_rate))
        if end <= begin:
            continue
        if intervals and begin <= intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], max(end, intervals[-1][1]))
        else:
            intervals.append((begin, end))

    angle = 2 * np.pi * frequency / sample_rate
    radius = np.exp(-np.log(1000.0) / (t60 * sample_rate))
    damped_radius = radius * np.exp(-np.log(1000.0) / (damper_t60 * sample_rate))
    numerator = [gain * (1.0 - radius) * np.sin(angle)]

    def run(begin, end, pole_radius):
        if end <= begin:
            return
        a1, a2 = -2 * pole_radius * np.cos(angle), pole_radius * pole_radius
        previous = out[begin - 1] if begin else 0.0
        previous2 = out[begin - 2] if begin >= 2 else 0.0
        state = [-a1 * previous - a2 * previous2, -a2 * previous]
        out[begin:end], _ = lfilter(numerator, [1.0, a1, a2],
                                  source[begin:end], zi=state)

    cursor = 0
    for begin, end in intervals:
        run(cursor, begin, damped_radius)
        run(begin, end, radius)
        cursor = end
    run(cursor, length, damped_radius)
    return out
