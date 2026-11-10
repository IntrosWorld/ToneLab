"""Acoustic string/body renderer with fixed gain and tab-aware excitation.

All audio is synthesized. The body remains a reduced modal model with passive
string loading; polarization splitting, upper-body radiation and bridge loss are
engineering approximations, not a measurement-calibrated instrument.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
import math
from pathlib import Path

import numpy as np
from scipy.signal import bilinear, butter, lfilter, sosfilt

import guitar_physics as phys

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "body_modes_v4.json"
AMPLITUDE_FLOOR = 10.0 ** (-90.0 / 20.0)
# Shared radiation calibration against the supplied fingerstyle reference.
# The upper path represents unresolved modes; geometric low-body pressure is
# retained. These are synthesis gains, not changes to mechanical eigenmodes.
SPATIAL_BODY_GAIN = .35
UNRESOLVED_BODY_GAIN = .30
UNRESOLVED_CROSSOVER_HZ = 1000.


def _resonator(x, freq, q, gain, sr):
    """Causal, DC-blocking bandpass with unity gain at its center frequency."""
    omega = 2.0 * math.pi * freq / sr
    alpha = math.sin(omega) / (2.0 * q)
    return lfilter([gain * alpha / (1.0 + alpha), 0.,
                    -gain * alpha / (1.0 + alpha)],
                   [1., -2.0 * math.cos(omega) / (1.0 + alpha),
                    (1.0 - alpha) / (1.0 + alpha)], x)


def _delay(x, frames):
    out = np.zeros_like(x)
    if 0 < frames < len(x):
        out[frames:] = x[:-frames]
    elif frames == 0:
        out[:] = x
    return out


class GuitarBody:
    def __init__(self, sample_rate, path=MODEL_PATH):
        self.sample_rate = sample_rate
        self.data = json.loads(Path(path).read_text(encoding="utf-8"))

    def process(self, bridge_force):
        """Fixed linear low-body transfer, independent of performance level."""
        if self.data.get("radiation_kernel_rate"):
            return self._spatial_pressure(bridge_force)
        if "coupled_modes" in self.data:
            return self._coupled_pressure(bridge_force)
        out = np.zeros((len(bridge_force), 2), dtype=np.float64)
        for mode in self.data["modes"]:
            # The exported weight ALREADY includes |bridge coupling|. Applying
            # it twice suppresses the top relative to the air-cavity mode.
            drive = float(mode["weight"]) * np.sign(float(mode["bridge"]))
            # Highly persistent isolated plate modes turn a pluck into a
            # pitched knock. Use a more lossy wooden body in the audio model.
            radiation_q = float(np.clip(.65 * mode["q"], 8., 23.))
            common = _resonator(bridge_force, mode["hz"], radiation_q,
                                .72 * drive, self.sample_rate)
            radiation_tilt = min(1.9, (mode["hz"] / 180.0)**.35)
            left, right = float(mode["left"]), float(mode["right"])
            # Nearby microphone perspectives; modest crossfeed avoids huge
            # phase opposition from two point samples of a plate mode shape.
            out[:, 0] += common * (.80 * left + .20 * right) * radiation_tilt
            out[:, 1] += common * (.80 * right + .20 * left) * radiation_tilt
        air = _resonator(bridge_force, self.data["helmholtz_hz"], 7.0,
                         .18, self.sample_rate)
        out[:, 0] += air
        out[:, 1] += .94 * air
        return out

    def _spatial_pressure(self, bridge_force):
        """Retarded surface/air radiation from every solved signed mode."""
        sr = self.sample_rate
        out = np.zeros((len(bridge_force), 2))
        kernel_rate = self.data["radiation_kernel_rate"]
        for mode in self.data["coupled_modes"]:
            if mode["hz"] >= .43 * sr:
                continue
            omega = 2. * sr * math.tan(math.pi * mode["hz"] / sr)
            b, a = bilinear([1., 0., 0.],
                            [1., omega / mode["q"], omega*omega], fs=sr)
            acceleration = lfilter(b, a, bridge_force)
            for channel, name in enumerate(("radiation_left", "radiation_right")):
                kernel = np.asarray(mode[name], dtype=float)
                if sr != kernel_rate:
                    times = np.arange(len(kernel)) * sr / kernel_rate
                    lower = np.floor(times).astype(int)
                    fraction = times-lower
                    resampled = np.zeros(int(lower[-1])+2)
                    np.add.at(resampled, lower, kernel*(1.-fraction))
                    np.add.at(resampled, lower+1, kernel*fraction)
                    kernel = resampled
                active = np.flatnonzero(kernel)
                if len(active):
                    first, last = int(active[0]), int(active[-1])+1
                    pressure = lfilter(kernel[first:last], [1.], acceleration)
                    # Baffled Rayleigh integral, rho_air/(2*pi); force scale
                    # remains common to every note, with no peak normalization.
                    out[:, channel] += (1.2/(2.*np.pi))*_delay(pressure, first)
        return out

    def string_body_decay(self, frequencies, string_index):
        """Passive bridge loading as modal amplitude decay, in inverse seconds.

        Bridge mobility uses squared mechanical residues, never the signed
        pressure residues. The finite reflection coefficient remains passive
        even near a strong body resonance. This is a modal weak-coupling
        reduction, not a full time-domain string/plate feedback simulation.
        """
        frequencies = np.asarray(frequencies, dtype=float)
        if not len(frequencies):
            return np.zeros_like(frequencies)
        omega = 2.*np.pi*frequencies
        mobility = np.zeros(len(frequencies), dtype=complex)
        for mode in self.data.get("coupled_modes", []):
            pole = 2.*np.pi*mode["hz"]
            mobility += mode["bridge"]**2 * 1j*omega / (
                pole*pole-omega*omega+1j*omega*pole/mode["q"])
        impedance = np.sqrt(phys.open_tension(string_index)
                            * phys.STRINGS[string_index].linear_density_kg_m)
        reflection = (1.-impedance*mobility)/(1.+impedance*mobility)
        return -frequencies[0]*np.log(np.clip(np.abs(reflection), 1e-12, 1.))

    def _coupled_pressure(self, bridge_force):
        """Pressure from the dimensioned top plus enclosed-air modal solve.

        Mass-normalized modes include the air-slug coordinate. Pressure is
        proportional to total volume acceleration, not raw string force or a
        stack of equally loud resonators. Signed geometric residues preserve
        cancellation; the .055 gain is fixed for this coefficient bank.
        """
        pressure = np.zeros(len(bridge_force))
        sr = self.sample_rate
        for mode in self.data["coupled_modes"]:
            if mode["hz"] >= .43 * sr:
                continue
            omega = 2. * sr * math.tan(math.pi * mode["hz"] / sr)
            gain = .055 * mode["signed_weight"]
            b, a = bilinear([gain, 0., 0.],
                            [1., omega / mode["q"], omega * omega], fs=sr)
            pressure += lfilter(b, a, bridge_force)
        # Nearby microphones, with physical positive travel time. Direct upper
        # radiation remains common enough to fold to mono without cancellation.
        left = _delay(pressure, round(.53 / 343. * sr))
        right = .98 * _delay(pressure, round(.55 / 343. * sr))
        return np.column_stack((left, right))

    def bridge_loss(self, frequencies):
        """Positive extra decay near radiating modes; weak-coupling estimate."""
        result = np.zeros_like(frequencies)
        for mode in self.data.get("coupled_modes", self.data["modes"]):
            width = mode["hz"] / mode["q"]
            result += .70 * mode["weight"] / (
                1.0 + ((frequencies - mode["hz"]) / width)**2)
        return np.minimum(result, 1.5)

    def upper_radiation(self, force):
        """Broad, damped approximation to unresolved high soundboard modes.

        This has no sampled impulse response, distortion, chorus or electric
        pickup/amp path. Coefficients are fixed, not fitted to each waveform.
        """
        sr = self.sample_rate
        if self.data.get("radiation_kernel_rate"):
            # Only the unresolved upper modes need a broadband approximation.
            # The low/mid force bypass formerly swamped the computed body.
            high = sosfilt(butter(2, min(UNRESOLVED_CROSSOVER_HZ, .30*sr), btype="highpass",
                                  fs=sr, output="sos"), force)
            high = sosfilt(butter(2, min(6800., .43*sr), fs=sr,
                                  output="sos"), high)
            for freq, q, gain in ((2300., 2.8, .22), (3300., 3.1, -.16),
                                  (4600., 3.8, .12)):
                if freq < .43*sr:
                    high += _resonator(high, freq, q, gain, sr)
            # A shared close-mic upper field avoids inventing a frequency-
            # dependent stereo null for modes whose shapes are unresolved.
            common = _delay(high, round(.546/343.*sr))
            return UNRESOLVED_BODY_GAIN*np.column_stack((common, .98*common))
        upper = sosfilt(butter(2, 65., btype="highpass", fs=sr, output="sos"), force)
        upper = sosfilt(butter(2, min(7200., .43 * sr), fs=sr, output="sos"), upper)
        # Sound radiation rises above the low top modes. A raw force bypass
        # made the previous render almost entirely fundamental/low harmonics.
        radiating = sosfilt(butter(1, 640., btype="highpass", fs=sr, output="sos"), upper)
        mid = .68 * upper + .32 * radiating
        for freq, q, gain in ((880., 2.1, .14), (1560., 2.7, .16),
                              (2700., 3.2, .11), (4200., 3.8, .045)):
            if freq < .43 * sr:
                mid += _resonator(upper, freq, q, gain, sr)
        # Two close microphones: causal path-length differences, with no
        # circular np.roll wrap-around from the end into the attack.
        left = .90 * mid + .10 * _delay(mid, round(.00012 * sr))
        right = .80 * mid + .16 * _delay(mid, round(.00019 * sr))
        right += .04 * _delay(mid, round(.00031 * sr))
        return np.column_stack((left, right))


class AcousticGuitar:
    def __init__(self, sample_rate=48_000, technique="finger", seed=1946,
                 source="continuous"):
        if sample_rate < 8000:
            raise ValueError("sample rate must be at least 8000 Hz")
        phys.pluck_position(technique)
        if source not in ("waveguide", "modal", "continuous"):
            raise ValueError("source must be 'waveguide', 'modal' or 'continuous'")
        self.sample_rate = int(sample_rate)
        self.technique = technique
        self.source = source
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.body = GuitarBody(self.sample_rate)

    def note_bridge_force(self, midi, velocity, duration, tail=3.0, *,
                          string_index=None, articulation="pick",
                          bend_points=None, source_midi=None, vibrato=None,
                          release_t60=None):
        if duration < 0 or tail < 0:
            raise ValueError("duration and tail must be nonnegative")
        idx = phys.choose_string(midi) if string_index is None else int(string_index)
        fret = phys.fret_for_midi(midi, idx)
        sr = self.sample_rate
        damping = phys.release_t60(midi, idx) if release_t60 is None else release_t60
        # A released/damped string is inaudible after ~100 dB attenuation;
        # body and room tails are handled once, after summing all strings.
        tail = min(tail, 1.7 * damping)
        total = max(1, round((duration + tail) * sr))
        t = np.arange(total, dtype=np.float64) / sr
        force = np.zeros(total)
        if velocity <= 0 or duration == 0:
            return force
        if self.source == "continuous":
            from string_performance import render_string_performance
            row = dict(midi=midi, velocity=velocity, start=0., duration=duration,
                       string_index=idx, articulation=articulation,
                       bend_points=bend_points, source_midi=source_midi,
                       vibrato=vibrato, release_t60=damping)
            return render_string_performance(
                [row], sample_rate=sr, tail=tail, technique=self.technique,
                seed=self.seed, body_loss=self.body.string_body_decay)
        count = phys.partial_count(midi, sr, idx)
        freqs = phys.partial_frequencies(midi, count, idx)
        vel = np.clip(velocity / 127., 0., 1.)
        # Players move toward the bridge for upper-fret passages. Leaving the
        # hand fixed at the soundhole put these lead notes close to mid-string,
        # suppressing their upper harmonics into a round, bell-like tone.
        position = phys.pluck_position(self.technique) * 2.**(fret / 12.) / (1. + .060 * fret)
        position = float(np.clip(position + self.rng.normal(0., .0035), .035, .43))
        amps = phys.partial_amplitudes(midi, velocity, count, self.technique,
                                       position=position, string_index=idx)
        legato = articulation in ("hammer_on", "pull_off", "legato_slide")
        if legato:
            amps *= .84 * np.exp(-freqs / 4700.)
        # Analytic integral of a small initial tension relaxation: a hard pluck
        # begins a few cents sharp then settles, without static random detuning.
        pitch_cents = (1.2 + 4.2 * vel**2) * (0.25 if legato else 1.)
        tau = .042 + .008 * (5 - idx)
        phase_time = t + (2.**(pitch_cents / 1200.) - 1.) * tau * (-np.expm1(-t / tau))
        pitch_semitones = np.zeros(total)
        if articulation == "legato_slide" and source_midi is not None:
            slide_time = min(.12, .65 * duration)
            u = np.clip(t / max(slide_time, 1. / sr), 0., 1.)
            pitch_semitones += (float(source_midi) - midi) * (1. - u * u * (3. - 2. * u))
        if bend_points:
            points = np.asarray(bend_points, dtype=float)
            if points.ndim != 2 or points.shape[1] != 2 or np.any(np.diff(points[:, 0]) < 0):
                raise ValueError("bend_points must be ordered [fraction, semitones] pairs")
            pitch_semitones += np.interp(t / max(duration, 1. / sr), points[:, 0], points[:, 1])
        if vibrato:
            depth = .10 if str(vibrato).lower() != "wide" else .23
            ramp = np.clip((t - .16) / .20, 0., 1.)
            pitch_semitones += depth * ramp * np.sin(2. * np.pi * 5.1 * t)
        if self.source == "waveguide":
            from plucked_string import render_plucked_string

            # The pick's short, irregular contact packet travels and reflects
            # along the string, with frequency-dependent loss on every trip.
            # This supplies pluck texture throughout the early decay instead
            # of adding an isolated click to a sum of smooth sinusoidal tones.
            ratios = (1. + (2.**(pitch_cents / 1200.) - 1.) * np.exp(-t / tau))
            ratios *= 2.**(pitch_semitones / 12.)
            seed = int(self.rng.integers(0, 2**31 - 1))
            options = dict(string_index=idx, seed=seed, release_t60=damping,
                           tail=tail, fret=fret, stiffness=phys.inharmonicity(midi, idx),
                           legato=legato,
                           brightness={"warm": .72, "finger": .84, "strum": .94,
                                       "pick": 1., "bright": 1.2}[self.technique])
            vertical = render_plucked_string(float(phys.fundamental(midi)), duration,
                                             sr, velocity, position,
                                             frequency_ratio=ratios, **options)
            split = 2.**((.65 + .13 * idx + .018 * fret) / 1200.)
            horizontal = render_plucked_string(float(phys.fundamental(midi)), duration,
                                               sr, velocity, position,
                                               frequency_ratio=ratios * split,
                                               decay_scale=1.4, **options)
            force = .82 * vertical + .18 * horizontal
            if articulation == "dead_note":
                force *= np.exp(-t / .013)
            return force * phys.bridge_force_scale(midi, idx)
        if bend_points or vibrato or (articulation == "legato_slide" and source_midi is not None):
            dt = np.diff(phase_time, prepend=-1. / sr)
            phase_time = np.cumsum(dt * 2.**(pitch_semitones / 12.))
            phase_time -= phase_time[0]
        loss = self.body.bridge_loss(freqs)
        split_cents = .65 + .13 * idx + .018 * fret
        split_ratio = 2.**(split_cents / 1200.)
        max_pitch_ratio = max(1., float(np.max(2.**(pitch_semitones / 12.))))
        for i, (freq, amp) in enumerate(zip(freqs, amps)):
            if freq * max_pitch_ratio * split_ratio >= .46 * sr or abs(amp) < AMPLITUDE_FLOOR:
                continue
            decay = float(phys.decay_rate(freq, midi, idx))
            phase = 2. * np.pi * freq * phase_time
            # Unequal losses and a tiny frequency split in the two transverse
            # polarizations create a moving, two-stage decay without chorus.
            vertical = .77 * np.exp(-(1.18 * decay + loss[i]) * t) * np.cos(phase)
            horizontal = .23 * np.exp(-.62 * decay * t) * np.cos(phase * split_ratio)
            force += amp * (vertical + horizontal)
        if articulation == "dead_note":
            force *= np.exp(-t / .013)
        # Finite release of pick/string contact prevents a sample discontinuity.
        attack_s = .0018 if legato else (.00055 if self.technique in ("pick", "bright", "strum") else .0011)
        attack = np.minimum(t / attack_s, 1.)
        force *= .5 - .5 * np.cos(np.pi * attack)
        if not legato:
            noise_n = min(total, round(phys.pick_noise_duration_s(self.technique) * sr))
            if noise_n:
                noise = self.rng.standard_normal(noise_n)
                low = 700. if idx < 4 else 1200.
                high = min(7600., .43 * sr)
                noise = sosfilt(butter(1, (low, high), btype="bandpass", fs=sr, output="sos"), noise)
                nt = np.arange(noise_n) / sr
                envelope = (1. - np.exp(-nt / .00015)) * np.exp(-nt / .0013)
                force[:noise_n] += .028 * vel**1.5 * noise * envelope
        release = min(total, round(duration * sr))
        if release < total:
            rt = np.arange(total - release) / sr
            force[release:] *= np.exp(-math.log(1000.) * rt / damping)
        return force * phys.bridge_force_scale(midi, idx)

    def render(self, events, tail=3.5, *, normalize=True, room=True):
        if tail < 0:
            raise ValueError("tail must be nonnegative")
        rows = []
        for event in events:
            if isinstance(event, Mapping):
                row = dict(event)
                if "string_index" in row:
                    expected = phys.fret_for_midi(row["midi"], row["string_index"])
                    if "fret" in row and row["fret"] != expected:
                        raise ValueError("score pitch disagrees with string/fret")
            else:
                midi, velocity, start, duration = event
                row = dict(midi=midi, velocity=velocity, start=start, duration=duration)
            if row["start"] < 0 or row["duration"] < 0:
                raise ValueError("note timing must be nonnegative")
            rows.append(row)
        rows.sort(key=lambda row: (row["start"], row.get("string_index", -1), row["midi"]))
        # A physical string cannot keep playing its previous fret after the
        # next fret is articulated. Leave only a short damping crossfade.
        by_string = {}
        for row in ([] if self.source == "continuous" else rows):
            if "string_index" in row:
                previous = by_string.get(row["string_index"])
                if previous is not None:
                    gap = row["start"] - previous["start"]
                    if previous["duration"] >= gap - 1e-7:
                        previous["duration"] = max(0., gap)
                        previous["release_t60"] = .030
                by_string[row["string_index"]] = row
        end = max((r["start"] + r["duration"] for r in rows), default=0.) + tail
        bridge = np.zeros(max(1, round(end * self.sample_rate)))
        for row in rows:
            if row["duration"] <= 0:
                continue
            if self.source == "continuous" and "string_index" in row:
                continue
            options = {key: row[key] for key in ("string_index", "articulation", "bend_points",
                        "source_midi", "vibrato", "release_t60") if key in row}
            note = self.note_bridge_force(row["midi"], row["velocity"], row["duration"], tail, **options)
            offset = round(row["start"] * self.sample_rate)
            span = min(len(note), len(bridge) - offset)
            if span > 0:
                bridge[offset:offset + span] += note[:span]
        if self.source == "continuous":
            from string_performance import render_string_performance
            continuous_rows = [r for r in rows if "string_index" in r]
            bridge += render_string_performance(
                continuous_rows, sample_rate=self.sample_rate, tail=tail,
                duration=len(bridge)/self.sample_rate,
                technique=self.technique, seed=self.seed,
                body_loss=self.body.string_body_decay)
        body_level = {"warm": .90, "finger": .80, "strum": .82,
                      "pick": .69, "bright": .60}[self.technique]
        if self.body.data.get("radiation_kernel_rate"):
            body_level = SPATIAL_BODY_GAIN
        out = body_level * self.body.process(bridge) + self.body.upper_radiation(bridge)
        out = sosfilt(butter(2, 42., btype="highpass", fs=self.sample_rate, output="sos"), out, axis=0)
        if room:
            out = self._room(out)
        peak = float(np.max(np.abs(out)))
        if normalize and peak:
            out *= .91 / peak
        return out

    def _room(self, stereo):
        """Quiet, damped close-room reflections with fully causal delays."""
        wet = np.zeros_like(stereo)
        soft = sosfilt(butter(1, min(4100., .40 * self.sample_rate),
                              fs=self.sample_rate, output="sos"), stereo, axis=0)
        taps = ((.013, .095), (.023, .074), (.037, -.059),
                (.053, .046), (.079, .033), (.113, -.022), (.157, .014))
        for channel in range(2):
            for delay_s, gain in taps:
                frames = round(delay_s * self.sample_rate) + channel * 13
                reflection = _delay(soft[:, channel], frames)
                wet[:, channel] += gain * reflection
                wet[:, 1 - channel] += .28 * gain * reflection
        return stereo + wet
