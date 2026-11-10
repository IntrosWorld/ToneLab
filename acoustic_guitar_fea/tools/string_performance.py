"""Continuous, stateful steel-string performance in a modal basis.

One complex oscillator bank belongs to each physical string for the entire
track. Fretting changes its frequencies while retaining finite vibration
energy and phase. Picks briefly damp that state before adding a new coherent
bridge-force wave; hammer-ons, pull-offs and legato slides only add a small
finger-contact excitation. This is an alternative string source, with no body,
room, normalization, external samples, or independently overlapping notes.

The diagonal fret-state transfer is a reduced engineering approximation. It
preserves continuous oscillator phase and bounded energy, but is not an exact
projection of a measured string displacement onto a new fret boundary.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
import time

import numpy as np

import guitar_physics as phys

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        return lambda fn: fn


LEGATO = frozenset(("hammer_on", "pull_off", "legato_slide"))
SUSTAIN = frozenset(("tie", "sustain"))


@njit(cache=True)
def _advance(real, imag, frequencies, decay, ratios, excitation_real,
             excitation_imag, contact_samples, contact_retention, off_sample,
             release_rate, sample_rate, control_hop):
    """Rotate existing state; optional contact forces never reset it."""
    frames = len(ratios)
    modes = len(real)
    out = np.zeros(frames, dtype=np.float64)
    cosine = np.zeros(modes, dtype=np.float64)
    sine = np.zeros(modes, dtype=np.float64)
    radius = np.exp(-decay / sample_rate)
    contact_gain = contact_retention ** (1.0 / max(1, contact_samples))
    release_gain = math.exp(-release_rate / sample_rate)
    # Equal weights summed over a raised-cosine pulse integrate to unity.
    for frame in range(frames):
        if frame % control_hop == 0:
            for mode in range(modes):
                angle = 2.0 * math.pi * frequencies[mode] * ratios[frame] / sample_rate
                cosine[mode] = math.cos(angle)
                sine[mode] = math.sin(angle)
        damping = 1.0
        if frame >= off_sample:
            damping *= release_gain
        injection = 0.0
        if frame < contact_samples:
            damping *= contact_gain
            injection = (1.0 - math.cos(2.0 * math.pi * (frame + .5)
                                      / contact_samples)) / contact_samples
        value = 0.0
        for mode in range(modes):
            # Output precedes integration: a new finger/pick contact cannot
            # create an instantaneous sample discontinuity at its onset.
            value += real[mode]
            old_real = real[mode]
            old_imag = imag[mode]
            gain = radius[mode] * damping
            real[mode] = (old_real * cosine[mode] - old_imag * sine[mode]) * gain
            imag[mode] = (old_real * sine[mode] + old_imag * cosine[mode]) * gain
            if injection:
                # Release a coherent displaced string, rather than summing
                # fixed-phase harmonic impulses through the contact window.
                # The latter adds an unwanted pulse/sinc low-pass that erases
                # upper harmonics on higher frets. This rotating drive keeps
                # each contribution aligned with the traveling string wave.
                drive_real = excitation_real[mode]
                drive_imag = excitation_imag[mode]
                excitation_real[mode] = drive_real * cosine[mode] - drive_imag * sine[mode]
                excitation_imag[mode] = drive_real * sine[mode] + drive_imag * cosine[mode]
                real[mode] += excitation_real[mode] * injection
                imag[mode] += excitation_imag[mode] * injection
        out[frame] = value
    return out


def _rows(events):
    rows = []
    for event in events:
        if isinstance(event, Mapping):
            row = dict(event)
        else:
            midi, velocity, start, duration = event
            row = dict(midi=midi, velocity=velocity, start=start, duration=duration)
        for field in ("start", "duration", "midi", "velocity"):
            if field not in row or not math.isfinite(float(row[field])):
                raise ValueError(f"{field} must be finite")
        if row["start"] < 0. or row["duration"] < 0.:
            raise ValueError("note timing must be nonnegative")
        midi = int(row["midi"])
        if midi != row["midi"]:
            raise ValueError("midi pitch must be integral")
        idx = int(row.get("string_index", phys.choose_string(midi)))
        if not 0 <= idx <= 5 or idx != row.get("string_index", idx):
            raise ValueError("string index must be an integer in 0..5")
        fret = phys.fret_for_midi(midi, idx)
        if row.get("fret", fret) != fret:
            raise ValueError("score pitch disagrees with string/fret")
        row.update(midi=midi, string_index=idx, fret=fret)
        if row.get("bend_points"):
            points = np.asarray(row["bend_points"], dtype=float)
            if (points.ndim != 2 or points.shape[1] != 2
                    or not np.isfinite(points).all()
                    or np.any(np.diff(points[:, 0]) < 0)):
                raise ValueError("bend_points must be ordered finite [fraction, semitones] pairs")
        if row["duration"] > 0. and row["velocity"] > 0:
            rows.append(row)
    return sorted(rows, key=lambda row: (row["start"], row["string_index"], row["midi"]))


def _excitation(row, mode_count, sample_rate, technique, rng, brightness):
    """Phase-coherent finite-contact bridge slope, with gentle contact variation."""
    midi, idx, fret = row["midi"], row["string_index"], row["fret"]
    position = row.get("pluck_fraction", row.get("pluck_position"))
    if position is None:
        position = phys.pluck_position(technique) * 2. ** (fret / 12.) / (1. + .060 * fret)
        position = np.clip(position + rng.normal(0., .0035), .035, .43)
    if not .02 <= position <= .48:
        raise ValueError("pluck position must lie between .02 and .48")
    frequencies = phys.partial_frequencies(midi, mode_count, idx)
    modes = np.arange(1, mode_count + 1, dtype=float)
    velocity = float(np.clip(row["velocity"] / 127., 0., 1.))
    cutoff = (2800. + 5700. * velocity) * brightness
    amplitudes = (.50 * np.sin(np.pi * modes * position)
                  / (np.pi * modes * position * (1. - position)))
    amplitudes *= np.sinc(modes * .008 / brightness)
    amplitudes *= np.exp(-(frequencies / cutoff) ** 1.25) * velocity ** 1.45
    variation = rng.normal(0., .035, mode_count)
    variation = np.convolve(np.pad(variation, (1, 1), mode="edge"), [.25, .5, .25], "valid")
    amplitudes *= (1. + variation) * phys.bridge_force_scale(midi, idx)
    phases = rng.normal(0., .018, mode_count) * np.minimum(modes / 5., 1.)
    return amplitudes * np.exp(1j * phases), frequencies


def _pitch_ratios(row, frames, sample_rate, previous_midi, legato, relax):
    t = np.arange(frames, dtype=float) / sample_rate
    semitones = np.zeros(frames)
    articulation = row.get("articulation", "pick")
    if ((legato and previous_midi is not None)
            or (articulation == "legato_slide" and row.get("source_midi") is not None)):
        source = row.get("source_midi")
        if source is None:
            source = previous_midi
        transition = min(.090, .65 * row["duration"]) if articulation == "legato_slide" else .0025
        u = np.clip(t / max(transition, 1. / sample_rate), 0., 1.)
        semitones += (float(source) - row["midi"]) * (1. - u * u * (3. - 2. * u))
    if row.get("bend_points"):
        points = np.asarray(row["bend_points"], dtype=float)
        semitones += np.interp(t / max(row["duration"], 1. / sample_rate), points[:, 0], points[:, 1])
    if row.get("vibrato"):
        depth = .23 if str(row["vibrato"]).lower() == "wide" else .10
        semitones += depth * np.clip((t - .16) / .20, 0., 1.) * np.sin(2. * np.pi * 5.1 * t)
    if relax and articulation not in LEGATO and articulation not in SUSTAIN:
        velocity = np.clip(row["velocity"] / 127., 0., 1.)
        cents = 1.2 + 4.2 * velocity ** 2
        semitones += .01 * cents * np.exp(-t / (.042 + .008 * (5 - row["string_index"])))
    return 2. ** (semitones / 12.)


def render_string_performance(events, sample_rate=48_000, tail=.35,
                              technique="pick", seed=1946, *, duration=None,
                              brightness=None, decay_scale=1.,
                              polarization=True, pitch_relaxation=True,
                              body_loss=None, frequency_shift=None,
                              return_diagnostics=False):
    """Return one mono bridge track with persistent state for all six strings.

    Mapping events accept GP score fields (start, duration, midi, velocity,
    string_index, fret, articulation, bend_points, source_midi, vibrato).
    Four-tuples use (midi, velocity, start, duration), choosing a playable string.
    A new same-string event supersedes the previous note's off time. A genuine
    intervening rest damps the existing state at its score off. ``duration``
    optionally fixes the exact output seconds, including tails; otherwise the
    output ends ``tail`` seconds after the last score note-off.

    Optional ``body_loss(frequencies, string_index)`` supplies a vector of
    passive excess amplitude decay rates in inverse seconds. Its frequencies
    contain the fundamental as the first element. Horizontal polarization
    receives 18% of this bridge loading. ``frequency_shift`` has the same
    signature and returns Hz; its fundamental-relative shift preserves the
    requested tuning. Both callbacks default to no loading/shift.
    """
    if sample_rate < 8000 or not math.isfinite(float(sample_rate)):
        raise ValueError("sample rate must be at least 8000 Hz")
    sample_rate = int(sample_rate)
    if not math.isfinite(tail) or tail < 0:
        raise ValueError("tail must be finite and nonnegative")
    if duration is not None and (not math.isfinite(duration) or duration < 0):
        raise ValueError("duration must be finite and nonnegative")
    if not math.isfinite(decay_scale) or decay_scale <= 0:
        raise ValueError("decay_scale must be finite and positive")
    phys.pluck_position(technique)
    if brightness is None:
        brightness = {"warm": .72, "finger": .84, "strum": .94, "pick": 1., "bright": 1.2}[technique]
    if not math.isfinite(brightness) or brightness <= 0:
        raise ValueError("brightness must be finite and positive")
    rows = _rows(events)
    end = duration if duration is not None else max((r["start"] + r["duration"] for r in rows), default=0.) + tail
    frames = max(1, round(end * sample_rate))
    output = np.zeros(frames)
    diagnostics = {"frames": frames, "sample_rate": sample_rate,
                   "source": "continuous_modal_string", "events": [], "strings": []}
    begin = time.perf_counter()
    # Oscillator coefficients update every 0.25 ms for pitch controls. States
    # are never interpolated/reset; constant-note frequencies remain exact.
    control_hop = max(1, round(.00025 * sample_rate))
    for idx in range(6):
        notes = [row for row in rows if row["string_index"] == idx and round(row["start"] * sample_rate) < frames]
        if not notes:
            continue
        mode_count = max(phys.partial_count(row["midi"], sample_rate, idx) for row in notes)
        bank_count = 2 if polarization else 1
        real = np.zeros(mode_count * bank_count)
        imag = np.zeros_like(real)
        rng = np.random.default_rng(np.random.SeedSequence([seed, idx]))
        previous_midi = None
        previous_off = None
        previous_score_off = None
        for index, row in enumerate(notes):
            start = round(row["start"] * sample_rate)
            following = round(notes[index + 1]["start"] * sample_rate) if index + 1 < len(notes) else frames
            stop = min(frames, following)
            span = stop - start
            if span <= 0:
                continue
            articulation = row.get("articulation", "pick")
            contiguous = previous_off is not None and start <= previous_off + 1
            if previous_score_off is not None:
                contiguous = contiguous or abs(float(row.get("score_start", row["start"])) - previous_score_off) < 1e-7
            legato = articulation in LEGATO and contiguous
            sustain = articulation in SUSTAIN and contiguous
            before_energy = float(np.dot(real, real) + np.dot(imag, imag))
            before_force = float(np.sum(real))
            excitation, frequencies = _excitation(row, mode_count, sample_rate, technique, rng, brightness)
            ratios = _pitch_ratios(row, span, sample_rate, previous_midi, legato, pitch_relaxation)
            # Remove out-of-band states before a fret/bend could alias them.
            valid = frequencies * max(1., float(np.max(ratios))) < min(14000., .43 * sample_rate)
            excitation *= valid
            f0 = float(phys.fundamental(row["midi"]))
            # Comparable to frequency-dependent loop losses of the waveguide;
            # fretted high notes do not lose brightness just for being high.
            fundamental_t60 = np.clip(7. * (165. / f0) ** .35 / (1. + .016 * row["fret"]), 1.5, 10.) * decay_scale
            decay = math.log(1000.) / fundamental_t60
            # The supplied steel-string reference's stable 2.2--4 kHz tonal
            # ridges retain energy much longer than v3. A conservative change
            # to the material-loss slope preserves that detail without adding
            # treble EQ, noise or extra low-frequency sustain (see reference
            # analysis under analysis/guitar_feel_v4/reference).
            decay = decay + (.14 * (2. * np.pi / 48000.) ** 2 * 82.5
                             * np.maximum(0., frequencies ** 2 - f0 ** 2) / decay_scale)
            loading = np.zeros(mode_count)
            if body_loss is not None:
                loading = np.broadcast_to(np.asarray(body_loss(frequencies.copy(), idx), dtype=float), (mode_count,)).copy()
                if not np.isfinite(loading).all() or np.any(loading < 0):
                    raise ValueError("body_loss must return finite nonnegative decay rates")
            if frequency_shift is not None:
                shift = np.broadcast_to(np.asarray(frequency_shift(frequencies.copy(), idx), dtype=float), (mode_count,))
                if not np.isfinite(shift).all():
                    raise ValueError("frequency_shift must return finite frequency offsets")
                # Keep the played fundamental in tune; retain differential
                # modal detuning from the reactive bridge impedance.
                frequencies += shift - frequencies / f0 * shift[0]
                if np.any(frequencies <= 0):
                    raise ValueError("frequency_shift produced a nonpositive frequency")
            if articulation == "dead_note" or row.get("muted"):
                decay += 1. / .013
            if bank_count == 2:
                split = 2. ** ((.65 + .13 * idx + .018 * row["fret"]) / 1200.)
                frequencies = np.concatenate((frequencies, frequencies * split))
                decay = np.concatenate((decay + loading, decay / 1.4 + .18 * loading))
                excitation = np.concatenate((.82 * excitation, .18 * excitation))
                valid = np.concatenate((valid, valid))
            else:
                decay += loading
            real *= valid
            imag *= valid
            frequencies = np.where(valid, frequencies, 0.)
            contact = max(2, round((.00055 if technique in ("pick", "bright") else .00080) * sample_rate))
            retention = .22 if before_energy > 1e-16 else 1.
            if legato:
                contact = max(2, round(.0011 * sample_rate))
                retention = .88 if articulation == "hammer_on" else .94
                excitation *= .075 if articulation == "hammer_on" else .12
            elif articulation in LEGATO:
                # A notated hammer beginning after a rest can sound by finger
                # contact, but cannot inherit a string that was already muted.
                excitation *= .35
            if sustain:
                contact = 0
                retention = 1.
                excitation *= 0.
            score_off = round((row["start"] + row["duration"]) * sample_rate)
            # Only this event controls release. The preceding event cannot
            # cut off a new fret, and a following contiguous note gets live state.
            off = min(span, max(0, score_off - start))
            release_t60 = float(row.get("release_t60", phys.release_t60(row["midi"], idx)))
            if not math.isfinite(release_t60) or release_t60 <= 0:
                raise ValueError("release_t60 must be finite and positive")
            part = _advance(real, imag, frequencies, decay, ratios,
                            excitation.real.copy(), excitation.imag.copy(),
                            contact, retention, off, math.log(1000.) / release_t60,
                            sample_rate, control_hop)
            if contiguous:
                # The first contact sample is the old physical displacement,
                # including tiny high-mode residue that is subsequently
                # removed when the new fret would push it above Nyquist.
                part[0] = before_force
            output[start:stop] += part
            diagnostics["events"].append({"start": row["start"], "string_index": idx,
                "midi": row["midi"], "articulation": articulation,
                "inherits_state": bool(legato or sustain),
                "state_energy_before": before_energy,
                "excitation_energy": float(np.vdot(excitation, excitation).real),
                "contact_retention": retention, "score_off_sample": score_off})
            previous_midi = row["midi"]
            previous_off = score_off
            previous_score_off = float(row.get("score_start", row["start"])) + float(row.get("score_duration", row["duration"]))
        diagnostics["strings"].append({"string_index": idx, "events": len(notes),
                                         "oscillators": mode_count * bank_count})
    diagnostics["elapsed_seconds"] = time.perf_counter() - begin
    diagnostics["audio_seconds"] = frames / sample_rate
    diagnostics["real_time_factor"] = diagnostics["elapsed_seconds"] / max(frames / sample_rate, 1e-12)
    return (output, diagnostics) if return_diagnostics else output
