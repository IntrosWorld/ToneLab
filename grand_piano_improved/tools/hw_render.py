#!/usr/bin/env python3
"""Fast model of what the RTL actually produces, for a whole performance.

`hw_model.simulate_note` is bit-accurate but covers only the string bank, and
only one note.  `render_model.py` covers the whole instrument but is the *ideal*
design, with floats, unlimited polyphony and -- the part that matters here --
peak normalisation of the sympathetic and reverb paths against the dry signal.
Hardware cannot look ahead to do that, so it uses fixed gains, and that
difference is exactly what makes the RTL sound drier than the approved render.

This module fills the gap: it reproduces the RTL's *structure and gain staging*
for a full piece, fast enough to tune against.

  * the same 28 voices, allocation and stealing rules
  * the same key-down / pedal logic
  * the same fixed, content-independent gains everywhere
  * the same order of stages

It is float rather than integer, and it renders each resonator in closed form
rather than stepping it, so it is not bit-accurate the way `simulate_note` is.
For choosing a *level* that should not matter -- a gain is a gain -- and it
turns a ten-minute-per-second RTL render into a few seconds.

STATUS: NOT YET AGREEING WITH THE RTL -- do not tune against it.

Measured against a real RTL render of the same 24 s passage, the envelopes
differ by 6.5 dB mean (correlation 0.86), and on a single sustained C4 the dry
string output differs by 13 dB after one second.  What has been ruled out:

  * the RTL itself.  It reproduces `hw_model.simulate_note` to 0.01 dB over a
    full second, so the hardware faithfully implements the intended arithmetic.
  * the fixed-point resonator.  A single resonator tracks the ideal exponential
    to 0.01 dB for twelve seconds, and individual components match the closed
    form to better than 0.8 dB.
  * truncation bias in the loss step -- rounding instead of truncating changes
    nothing.
  * the decay dead zone, where `loss * x` would shift to zero: it exists in
    principle below about -70 dB but the resonator does not stall there.

So single resonators agree and whole notes do not, which points at the summed
beat structure: the ~100 components of a note include unisons detuned by about
a cent (0.14 Hz at C4, a seven-second beat period) and polarisations detuned by
4e-4.  Neither a 30 ms nor a 500 ms analysis window averages a seven-second
beat, so the two renderings may simply be caught in different beat phases and
the comparison itself may be measuring nothing.  That has to be settled --
with long-window energy over many seconds -- before this model is trusted.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter

import hw_model as hw
import piano_physics as phys
import render_model as rm


# ---------------------------------------------------------------------------
# Driven magic-circle resonator, as a transfer function
# ---------------------------------------------------------------------------
def resonator_taps(k, loss, gain):
    """Biquad equivalent of the RTL's driven resonator.

    The RTL steps

        x <- (x + k*y + g*u) * (1-lambda)
        y <- (y - k*x)       * (1-lambda)

    and takes y as the output.  Solving that recurrence gives a two-pole
    section, so the whole bank can run through `lfilter` at C speed instead of
    a Python loop over a million samples.
    """
    a = 1.0 - loss
    b = [-(a ** 2) * k * gain]
    den = [1.0, -(2.0 * a - (a ** 2) * k * k), a ** 2]
    return b, den


def _check_resonator_form():
    """Confirm the closed form matches the RTL recurrence step for step."""
    k, loss, gain = 0.05, 1e-4, 0.01
    n = 2000
    u = np.zeros(n)
    u[0] = 1.0
    x = y = 0.0
    direct = np.zeros(n)
    for i in range(n):
        x = (x + k * y + gain * u[i]) * (1.0 - loss)
        y = (y - k * x) * (1.0 - loss)
        direct[i] = y
    b, a = resonator_taps(k, loss, gain)
    closed = lfilter(b, a, u)
    err = np.abs(direct - closed).max() / max(np.abs(direct).max(), 1e-18)
    assert err < 1e-6, f"resonator form mismatch: {err:.2e}"


# ---------------------------------------------------------------------------
# Voices: the RTL's allocation rules
# ---------------------------------------------------------------------------
class VoicePool:
    """Mirrors grand_piano_core's allocation, stealing and pedal behaviour."""

    def __init__(self, count=hw.NUM_VOICES):
        self.count = count
        self.pitch = [None] * count
        self.key_down = [False] * count
        self.released = [False] * count
        self.undamped = [False] * count
        self.age = [0] * count
        self.free_at = [-1.0] * count   # when the voice becomes reusable

    def allocate(self, pitch, time, clock):
        # Free voice, then a released one, then the oldest -- as in the RTL.
        for v in range(self.count):
            if self.pitch[v] is None or (self.free_at[v] >= 0.0
                                         and self.free_at[v] <= time):
                return v, False
        for v in range(self.count):
            if self.released[v]:
                return v, True
        return min(range(self.count), key=lambda v: self.age[v]), True


def render_events(events, spans, sample_rate=hw.SAMPLE_RATE, tail=3.0,
                  voices=hw.NUM_VOICES, verbose=False):
    """Render the string bank exactly as the RTL voices it, into dry L/R."""
    end = max((e[3] for e in events), default=0.0)
    total = int(sample_rate * (end + tail)) + 1
    dry = np.zeros((total, 2), dtype=np.float64)
    mono = np.zeros(total, dtype=np.float64)

    stereo = hw.stereo_rom(sample_rate)

    def pedal_down(t):
        return any(a <= t < b for a, b in spans)

    pool = VoicePool(voices)
    order = sorted(range(len(events)), key=lambda i: events[i][2])
    stolen = 0

    for clock, idx in enumerate(order):
        pitch, velocity, start, release = events[idx]
        v, was_stolen = pool.allocate(pitch, start, clock)
        stolen += 1 if was_stolen else 0

        comps = hw.components(pitch, sample_rate)
        layer = int(np.clip(velocity * hw.VELOCITY_LAYERS / 128.0,
                            0, hw.VELOCITY_LAYERS - 1))
        amps = hw.normalised_amplitudes(pitch, hw.velocity_for_layer(layer),
                                        comps, sample_rate)

        # The RTL damps a note when its key is up and the pedal is not holding
        # it.  A note held through a pedal lift keeps ringing.
        damp_t60 = phys.damper_t60(pitch)
        if damp_t60 is None:
            damp_at = None                      # top octave has no damper
        else:
            damp_at = release
            for a, b in spans:
                if a <= release < b:
                    damp_at = b                 # pedal holds it until the lift
                    break

        span_len = min(total - int(start * sample_rate),
                       int(sample_rate * hw.MAX_NOTE_SECONDS))
        if span_len <= 0:
            continue
        buf = np.zeros(span_len, dtype=np.float64)
        t = np.arange(span_len) / sample_rate

        contact = float(phys.contact_time_s(pitch, velocity)) / 3.0
        attack = 1.0 - np.exp(-t / contact)

        for (freq, rate, phase, _, _, _), amp in zip(comps, amps):
            if amp <= 0.0:
                continue
            lifetime = int(sample_rate * math.log(max(amp, 1e-12) / 1e-6)
                           / max(rate, 1e-6)) + 1
            n = min(span_len, lifetime)
            if n <= 1:
                continue
            buf[:n] += amp * np.exp(-rate * t[:n]) * np.sin(
                2.0 * np.pi * freq * t[:n] + phase)
        buf *= attack

        if damp_at is not None and damp_at > start:
            off = int((damp_at - start) * sample_rate)
            if off < span_len:
                tt = np.arange(span_len - off) / sample_rate
                buf[off:] *= np.exp(-tt * math.log(1000.0) / damp_t60)

        begin = int(start * sample_rate)
        dl, dr, pan_l, pan_r = stereo[pitch - phys.LOWEST_MIDI]
        gl = pan_l / (1 << hw.GAIN_Q)
        gr = pan_r / (1 << hw.GAIN_Q)
        for channel, delay, gain in ((0, dl, gl), (1, dr, gr)):
            s = begin + delay
            n = min(span_len, total - s)
            if n > 0:
                dry[s:s + n, channel] += buf[:n] * gain
        n = min(span_len, total - begin)
        mono[begin:begin + n] += buf[:n]

        pool.pitch[v] = pitch
        pool.age[v] = clock
        pool.released[v] = False
        pool.free_at[v] = (damp_at + 4.0 * damp_t60) if damp_at is not None else -1.0

    if verbose:
        print(f"  voices stolen: {stolen} of {len(events)} notes")
    return dry, mono, total


# ---------------------------------------------------------------------------
# The stages after the strings
# ---------------------------------------------------------------------------
def soundboard(dry, sample_rate=hw.SAMPLE_RATE, mix=hw.BODY_MIX_Q):
    rows = hw.body_rom(sample_rate=sample_rate)
    out = np.zeros_like(dry)
    for channel, gain_index in ((0, 2), (1, 3)):
        acc = np.zeros(len(dry))
        for row in rows:
            k = row[0] / (1 << hw.COEF_Q)
            loss = row[1] / (1 << hw.LOSS_Q)
            g = row[gain_index] / (1 << hw.COEF_Q)
            if k == 0.0 or g == 0.0:
                continue
            b, a = resonator_taps(k, loss, g)
            acc += lfilter(b, a, dry[:, channel])
        out[:, channel] = (1.0 - mix) * dry[:, channel] + mix * acc
    return out


def sympathetic(mixed, spans, sample_rate=hw.SAMPLE_RATE,
                level=hw.SYMP_LEVEL_Q):
    rows = hw.sympathetic_rom(sample_rate)
    total = len(mixed)

    gate = np.zeros(total)
    for a, b in spans:
        lo, hi = max(0, int(a * sample_rate)), min(total, int(b * sample_rate))
        if hi > lo:
            gate[lo:hi] = 1.0

    drive = mixed.mean(axis=1)
    drive = lfilter(*butter(2, hw.SYMP_HIGHPASS_HZ / (sample_rate * 0.5),
                            btype="high"), drive)
    pedalled = drive * gate

    out = np.zeros_like(mixed)
    for k_i, loss_i, gain_i, pl, pr, undamped in rows:
        if k_i == 0 or gain_i == 0:
            continue
        k = k_i / (1 << hw.COEF_Q)
        loss = loss_i / (1 << hw.LOSS_Q)
        g = gain_i / (1 << hw.COEF_Q)
        b, a = resonator_taps(k, loss, g)
        resp = lfilter(b, a, drive if undamped else pedalled)
        out[:, 0] += resp * (pl / (1 << hw.GAIN_Q))
        out[:, 1] += resp * (pr / (1 << hw.GAIN_Q))
    return mixed + level * out


def room(pre, sample_rate=hw.SAMPLE_RATE, mix=hw.ROOM_MIX_Q,
         feedback=hw.ROOM_SIZE_Q, damping=hw.ROOM_DAMPING_Q,
         wet_gain=None):
    if wet_gain is None:
        wet_gain = (1.0 - feedback) * hw.ROOM_GAIN_Q
    wet = np.zeros_like(pre)
    for channel in range(2):
        offset = 0 if channel == 0 else hw.REVERB_STEREO_OFFSET
        acc = np.zeros(len(pre))
        for length in hw.REVERB_COMBS:
            d = length + offset
            # Damped comb, folded into one recursion:
            #   (1 - damp z^-1) y - fb(1-damp) z^-d y = (1 - damp z^-1) x
            a = np.zeros(d + 1)
            a[0] = 1.0
            a[1] = -damping
            a[d] = -feedback * (1.0 - damping)
            acc += lfilter([1.0, -damping], a, pre[:, channel])
        acc /= len(hw.REVERB_COMBS)
        for length in hw.REVERB_ALLPASS:
            d = length + offset
            g = 0.5
            b = np.zeros(d + 1)
            b[0] = -g
            b[d] = 1.0
            a = np.zeros(d + 1)
            a[0] = 1.0
            a[d] = -g
            acc = lfilter(b, a, acc)
        wet[:, channel] = acc
    return (1.0 - mix) * pre + mix * (wet * wet_gain)


def render(events, spans, sample_rate=hw.SAMPLE_RATE, symp_level=None,
           room_mix=None, master=hw.MASTER_GAIN_Q, verbose=False):
    """Full chain, in the RTL's order and with the RTL's gain staging."""
    dry, _, _ = render_events(events, spans, sample_rate, verbose=verbose)
    mixed = soundboard(dry, sample_rate)
    pre = sympathetic(mixed, spans, sample_rate,
                      level=hw.SYMP_LEVEL_Q if symp_level is None else symp_level)
    wet = room(pre, sample_rate,
               mix=hw.ROOM_MIX_Q if room_mix is None else room_mix)
    return wet * master


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--midi", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--symp-level", type=float)
    parser.add_argument("--room-mix", type=float)
    args = parser.parse_args()

    _check_resonator_form()

    events, spans = rm.load_events(args.midi)
    if args.seconds:
        events = [(p, v, s, min(r, args.seconds))
                  for p, v, s, r in events if s < args.seconds]
        spans = [(a, min(b, args.seconds)) for a, b in spans if a < args.seconds]

    print(f"{len(events)} notes, {len(spans)} pedal spans")
    audio = render(events, spans, symp_level=args.symp_level,
                   room_mix=args.room_mix, verbose=True)
    peak = float(np.abs(audio).max())
    print(f"peak {20 * np.log10(max(peak, 1e-12)):.2f} dBFS"
          f"{'   CLIPPING' if peak > 1.0 else ''}")
    rm.write_wav(args.out, np.clip(audio, -1.0, 1.0), hw.SAMPLE_RATE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
