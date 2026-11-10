#!/usr/bin/env python3
"""Reference renderer for the grand-piano engine.

This is the *design* model.  It implements exactly the signal flow the RTL
implements, in floating point, so the sound can be tuned in seconds instead of
waiting on a Vivado XSim run.  Once it sounds right, `fixed_point.py` reproduces
it in the RTL's arithmetic and the SystemVerilog is checked against that.

Signal flow, per note:

    hammer excitation  ->  stiff-string partial bank
                           (inharmonic, per-partial decay,
                            twin polarisations, detuned unisons)
                       ->  damper envelope
    all notes summed   ->  soundboard body resonances (from the FEA solve)
                       ->  radiation tilt + stereo pickup
                       ->  room

Nothing here reads audio.  Every coefficient comes from `piano_physics.py` or
the finite-difference soundboard solve.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

import piano_physics as phys
from audio_dsp import allpass, damped_comb, causal_smooth, pedal_resonator
from audio_io import write_wav, audio_metrics

RTL_DIR = Path(__file__).resolve().parents[1] / "rtl"

# Partials beyond these limits are indistinguishable but cost real time.
# How far up the partial series the full string treatment runs.  Partials above
# these limits are synthesised as a single decaying sinusoid, which is static
# and lifeless -- so keeping the limits low is a large part of what makes a
# modal piano sound synthetic.  Raising them is the main cost/realism trade in
# this engine.
TWIN_POLARISATION_PARTIALS = 40   # double decay and shimmer
UNISON_PARTIALS = 44              # detuned strings, hence beating
AMPLITUDE_FLOOR_DB = -78.0
VELOCITY_BINS = 8
# How long a single note may ring.  This is a sustain limit, not an
# optimisation: a bottom-octave string has a 40 s T60, so a 14 s cap chopped it
# off while still 21 dB above where it started -- an abrupt stop in the middle
# of the sustain, which is exactly where it is most audible with the pedal
# down.  Only bass notes reach lengths like this, and few of them sound at once.
MAX_TEMPLATE_SECONDS = 32.0
# If a note still has to be truncated, fade it rather than cutting it, so the
# end is a decay instead of a click.
TRUNCATION_FADE_S = 0.08
# How much of the output comes through the resonant body rather than straight
# off the strings.  The body path rings, so raising this softens attacks.
BODY_MIX = 0.20
# How much of the hall is in the final mix.
ROOM_MIX = 0.34
# Level of the reverb network once its own gain has been normalised out.  This
# is a mix decision, not a physical constant -- but it is content-independent,
# which is the point: scaling the tail to the dry peak instead makes the
# reverb louder on sparse material than on dense material.
ROOM_GAIN = 8.0


# ---------------------------------------------------------------------------
# One decaying partial
# ---------------------------------------------------------------------------
def _add_partial(buf, sample_rate, freq, amp, rate, phase, glide_cents=0.0,
                 contact_s=0.0):
    """Accumulate amp * exp(-rate t) * sin(phase(t)) into `buf`.

    The partial is truncated where it drops under the amplitude floor, which is
    what keeps this renderer fast: most high partials are inaudible within a
    few hundred milliseconds even though the note rings for seconds.

    `glide_cents` starts the partial sharp and lets it settle, modelling the
    tension the hammer adds to the string on a hard blow.  It is integrated
    into the phase rather than applied as a frequency step so the partial stays
    continuous.
    """
    floor = 10.0 ** (AMPLITUDE_FLOOR_DB / 20.0)
    if amp <= floor or freq <= 0.0 or freq >= 0.5 * sample_rate:
        return
    span = int(sample_rate * math.log(amp / floor) / max(rate, 1e-6)) + 1
    span = min(span, len(buf))
    if span <= 1:
        return
    # Float32 phase loses resolution over long high-register sustains.
    t = np.arange(span, dtype=np.float64) / sample_rate
    angle = 2.0 * np.pi * freq * t
    if glide_cents > 0.0:
        # Integral of a decaying frequency offset df*exp(-t/tau).
        tau = phys.GLIDE_TAU_S
        df = freq * (2.0 ** (glide_cents / 1200.0) - 1.0)
        angle = angle + 2.0 * np.pi * df * tau * (1.0 - np.exp(-t / tau))

    envelope = np.exp(-rate * t)
    if contact_s > 0.0:
        # The hammer pushes the string rather than releasing it, so the partial
        # ramps up over the contact time instead of starting at full amplitude.
        # Without this the attack is a pluck.
        envelope = envelope * (1.0 - np.exp(-t / (contact_s / 3.0)))
    buf[:span] += (amp * envelope * np.sin(angle + phase)).astype(np.float32)


# ---------------------------------------------------------------------------
# Soundboard
# ---------------------------------------------------------------------------
class Soundboard:
    """Body resonances from the finite-difference plate solve.

    Every string on a piano drives the same board, so this is applied once to
    the summed string signal rather than per voice -- which is both physically
    correct and far cheaper than a body filter per note.
    """

    def __init__(self, sample_rate, path=RTL_DIR / "soundboard_modes.json"):
        self.sample_rate = sample_rate
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.modes = data["modes"]

    def process(self, x, pickup="left", mix=BODY_MIX):
        """Body colouration as heard at one microphone position.

        Each mode is weighted by its own amplitude at that point on the plate,
        taken from the FEA mode shapes.  Two points see different mixtures of
        the modes, including sign flips where a mode's nodal line falls between
        them, so the two channels decorrelate the way a real instrument does.
        Panning one mono signal cannot produce that.

        Each mode is normalised to unity gain *at its own resonance* before
        being weighted, and the bank is scaled by the total weight so the body
        path stays level-comparable with the dry path.  Rescaling the summed
        body path to the dry peak instead -- the obvious-looking thing to do --
        blows up the high-Q low modes, and since a Q of 70 at 46 Hz takes about
        half a second to ring up, that smears every attack into a swell.
        """
        from scipy.signal import lfilter

        out = np.zeros_like(x)
        total = 0.0
        for mode in self.modes:
            hz = mode["hz"]
            if hz >= 0.45 * self.sample_rate:
                continue
            gain = mode["weight"] * mode.get(pickup, 1.0)
            if gain == 0.0:
                continue
            w = 2.0 * np.pi * hz / self.sample_rate
            r = math.exp(-w / (2.0 * max(mode["q"], 1e-3)))
            a = [1.0, -2.0 * r * math.cos(w), r * r]
            # |H| at resonance for b0 = 1, used to normalise the mode.
            peak = abs(1.0 / (1.0 - 2.0 * r * math.cos(w) * np.exp(-1j * w)
                              + r * r * np.exp(-2j * w)))
            out += lfilter([gain / max(peak, 1e-9)], a, x)
            total += abs(gain)
        if total > 0.0:
            out /= total
        return (1.0 - mix) * x + mix * out


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class RenderCancelled(Exception):
    """Raised at a render checkpoint when the caller requests cancellation."""


class GrandPiano:
    def __init__(self, sample_rate=48000, seed=1234, *, room_mix=ROOM_MIX,
                 body_mix=BODY_MIX, normalize=True, max_cache_mb=256,
                 progress=None, cancel=None):
        if int(sample_rate) != sample_rate or not 8000 <= sample_rate <= 192000:
            raise ValueError("sample rate must be an integer from 8000 to 192000 Hz")
        if not 0 <= room_mix <= 1 or not 0 <= body_mix <= 1:
            raise ValueError("room and body mix must be between 0 and 1")
        if not np.isfinite(max_cache_mb) or max_cache_mb < 0:
            raise ValueError("cache size must be finite and non-negative")
        self.sample_rate = int(sample_rate)
        self.room_mix = float(room_mix)
        self.body_mix = float(body_mix)
        self.normalize = bool(normalize)
        self.progress = progress
        self.cancel = cancel
        self.max_cache_bytes = int(max_cache_mb * 1024 * 1024)
        self.cache_bytes = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.rng = np.random.default_rng(seed)
        self._templates = OrderedDict()
        self.soundboard = Soundboard(sample_rate)
        # Fixed per-note phases keep renders deterministic and repeatable.
        self._phase = {}

    def _checkpoint(self, fraction=None, message=""):
        if self.cancel is not None and self.cancel():
            raise RenderCancelled("render cancelled")
        if fraction is not None and self.progress is not None:
            self.progress(float(fraction), message)

    @staticmethod
    def _velocity_layers(velocity):
        position = float(np.clip(velocity * VELOCITY_BINS / 127.0 - 0.5,
                                 0.0, VELOCITY_BINS - 1))
        lower = int(position)
        blend = position - lower
        if blend <= 1e-12 or lower == VELOCITY_BINS - 1:
            return [(lower, 1.0)]
        return [(lower, 1.0 - blend), (lower + 1, blend)]

    # -- string bank ------------------------------------------------------
    def _phases(self, pitch, count):
        if pitch not in self._phase:
            rng = np.random.default_rng(9000 + pitch)
            self._phase[pitch] = rng.uniform(0.0, 2.0 * np.pi, 512)
        return self._phase[pitch][:count]

    def note_template(self, pitch, velocity_bin, seconds=None):
        """Undamped mono note.  Cached: the model is linear and time-invariant
        for a given pitch and velocity, so every occurrence reuses one render.
        """
        key = (pitch, velocity_bin)
        if not phys.LOWEST_MIDI <= pitch <= phys.HIGHEST_MIDI or int(pitch) != pitch:
            raise ValueError("pitch must be an integer from 21 to 108")
        if not 0 <= velocity_bin < VELOCITY_BINS or int(velocity_bin) != velocity_bin:
            raise ValueError("invalid velocity layer")
        if seconds is not None and (not np.isfinite(seconds) or seconds <= 0):
            raise ValueError("template duration must be positive and finite")
        self._checkpoint()

        sr = self.sample_rate
        velocity = (velocity_bin + 0.5) * 127.0 / VELOCITY_BINS
        count = phys.partial_count(pitch, sr)
        freqs = phys.partial_frequencies(pitch, count)
        amps = phys.partial_amplitudes(pitch, velocity, count)
        amps = amps * phys.radiation_gain(freqs)

        # Buffer only as long as the slowest partial actually rings.
        longest = float(phys.t60(freqs[0])) * (AMPLITUDE_FLOOR_DB / -60.0)
        length = int(sr * min(MAX_TEMPLATE_SECONDS, max(0.35, longest))) + 1
        if seconds is not None:
            length = min(length, int(sr * max(0.35, math.ceil(seconds * 2) / 2)) + 1)
        if key in self._templates and len(self._templates[key]) >= length:
            self.cache_hits += 1
            self._templates.move_to_end(key)
            return self._templates[key]
        self.cache_misses += 1
        buf = np.zeros(length, dtype=np.float32)

        strings = phys.strings_per_note(pitch)
        phases = self._phases(pitch, count)
        glide = float(phys.attack_glide_cents(pitch, velocity))
        contact = float(phys.contact_time_s(pitch, velocity))

        for n in range(count):
            self._checkpoint()
            base_freq = freqs[n]
            base_amp = amps[n]
            if base_amp <= 0.0:
                continue
            rate = float(phys.decay_rate(base_freq))

            use_unison = n < UNISON_PARTIALS
            voices = strings if use_unison else 1
            for s in range(voices):
                cents = phys.unison_detune_cents(pitch, s, strings) if use_unison else 0.0
                freq = base_freq * (2.0 ** (cents / 1200.0))
                amp = base_amp / math.sqrt(voices)
                phase = phases[n] + s * 1.7

                if n < TWIN_POLARISATION_PARTIALS:
                    # Twin polarisations: a fast one that carries the attack and
                    # a slow one that carries the aftersound.  Their beat is the
                    # reason a real piano note shimmers as it decays.
                    _add_partial(buf, sr, freq, amp * phys.PROMPT_LEVEL,
                                 rate / phys.PROMPT_T60_SCALE, phase, glide, contact)
                    _add_partial(buf, sr,
                                 freq * (1.0 + phys.POLARISATION_DETUNE_RATIO),
                                 amp * phys.AFTER_LEVEL,
                                 rate * phys.AFTER_T60_SCALE, phase + 0.9, glide, contact)
                else:
                    _add_partial(buf, sr, freq, amp, rate, phase, glide, contact)

        # Duplex segments beyond the bridge, tuned near harmonics.  These give
        # the middle and upper registers their shimmer.
        dup_f, dup_a = phys.duplex_partials(pitch, velocity, sr)
        for k, (freq, amp) in enumerate(zip(dup_f, dup_a)):
            _add_partial(buf, sr, float(freq), float(amp),
                         math.log(1000.0) / phys.DUPLEX_T60,
                         0.7 + 1.3 * k, 0.0, contact)

        # Phantom longitudinal partials: not part of the transverse series, and
        # the reason a hard bass note growls.
        ghost_f, ghost_a = phys.longitudinal_partials(pitch, velocity, freqs)
        for freq, amp in zip(ghost_f, ghost_a):
            if freq < 0.45 * sr:
                _add_partial(buf, sr, float(freq), float(amp),
                             float(phys.decay_rate(freq)) / phys.LONGITUDINAL_T60_SCALE,
                             0.31)

        buf += self._hammer_noise(pitch, velocity, length)

        # A note that reaches the length cap is still audible there, so ending
        # the buffer outright would be a step to silence.  Fade the last few
        # tens of milliseconds instead.
        if length >= int(sr * MAX_TEMPLATE_SECONDS):
            fade = min(len(buf), int(sr * TRUNCATION_FADE_S))
            if fade > 1:
                buf[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)

        # Stable attack calibration: rendering a longer sustain must not
        # change the level of its first strike through a later unison beat.
        peak = float(np.abs(buf[:max(1, int(sr * 0.3))]).max())
        if peak > 0.0:
            buf /= peak
            buf *= self._note_loudness(pitch, velocity)
        if key in self._templates:
            self.cache_bytes -= self._templates.pop(key).nbytes
        while self._templates and self.cache_bytes + buf.nbytes > self.max_cache_bytes:
            _, old = self._templates.popitem(last=False)
            self.cache_bytes -= old.nbytes
        if buf.nbytes <= self.max_cache_bytes:
            self._templates[key] = buf
            self.cache_bytes += buf.nbytes
        return buf

    def _note_loudness(self, pitch, velocity):
        """Level of the note before the board.

        Pianos are not equally loud across the compass: the bass has far more
        radiating string and the extreme treble is short and quiet.
        """
        v = np.clip(velocity / 127.0, 0.02, 1.0)
        # Long bass strings move a lot of air; the extreme treble is short,
        # stiff and quiet.
        bass = 1.0 + 0.35 * np.clip((52 - pitch) / 31.0, 0.0, 1.0)
        # The treble is quieter than the middle, but only somewhat: a 0.45 cut
        # buries the top two octaves, where a real grand still projects.
        treble = 1.0 - 0.24 * np.clip((pitch - 52) / 56.0, 0.0, 1.0)
        return float(0.9 * (v ** 1.25) * bass * treble)

    def _hammer_noise(self, pitch, velocity, length):
        """The mechanical thump of felt, key and action.

        This is a low knock, not a click.  The audible noise of a piano action
        is the key bottoming out and the hammer shank and keybed moving -- all
        heavy wooden things, so the energy sits low.  Filtering it anywhere
        near the hammer's spectral corner (a few kHz at C4) produces a bright
        tick on top of the attack, which reads as a plucked string.
        """
        from scipy.signal import butter, lfilter

        sr = self.sample_rate
        span = min(length, int(sr * 0.09))
        if span <= 8:
            return np.zeros(length, dtype=np.float32)
        rng = np.random.default_rng(5000 + pitch * 13 + int(velocity))
        noise = rng.normal(0.0, 1.0, span)
        t = np.arange(span) / sr

        # Thump: heavy, low, and following the note down the compass.
        corner = float(np.clip(phys.fundamental(pitch) * 1.6, 70.0, 420.0))
        thump = lfilter(*butter(2, corner / (sr * 0.5), btype="low"), noise)
        thump *= np.exp(-t / 0.028)

        # A much quieter mid-band component for the felt itself.
        felt = lfilter(*butter(2, [300.0 / (sr * 0.5), 2200.0 / (sr * 0.5)],
                               btype="band"), noise)
        felt *= np.exp(-t / 0.006) * 0.28

        mix = thump + felt
        peak = float(np.abs(mix).max())
        if peak > 0.0:
            mix /= peak
        level = 0.055 * (velocity / 127.0) ** 1.5
        out = np.zeros(length, dtype=np.float32)
        out[:span] = (mix * level).astype(np.float32)
        return out

    # -- events -----------------------------------------------------------
    def render(self, events, pedal_spans=(), tail=4.0):
        """Render (pitch, velocity, start_s, release_s) events.

        `pedal_spans` is a list of (down_s, up_s) sustain-pedal intervals, used
        to drive sympathetic resonance.
        """
        events, pedal_spans = list(events), list(pedal_spans)
        # Public API validation also covers callers that do not load MIDI.
        for pitch, velocity, start, release in events:
            if (int(pitch) != pitch or not phys.LOWEST_MIDI <= pitch <= phys.HIGHEST_MIDI
                    or int(velocity) != velocity or not 1 <= velocity <= 127
                    or not np.isfinite([start, release]).all()
                    or start < 0 or release < start):
                raise ValueError("invalid note: require pitch 21..108, velocity 1..127, 0 <= start <= release")
        for down, up in pedal_spans:
            if not np.isfinite([down, up]).all() or down < 0 or up < down:
                raise ValueError("invalid pedal interval")
        if not np.isfinite(tail) or tail < 0:
            raise ValueError("tail must be finite and non-negative")
        self._checkpoint(0.0, "Preparing strings")
        sr = self.sample_rate
        end = max((e[3] for e in events), default=0.0)
        total = int(sr * (end + tail)) + 1
        # Two dry buses: notes are placed across the board by pitch before the
        # body filter, because bass and treble strings really do drive the
        # soundboard in different places.
        dry = np.zeros((total, 2), dtype=np.float32)
        mono = np.zeros(total, dtype=np.float32)

        # Render only the audible part of a template, shared by all repeats.
        horizons = {}
        for pitch, velocity, start, release in events:
            damping = phys.damper_t60(pitch)
            horizon = (end + tail - start if damping is None else
                       min(end + tail - start, release - start + 1.5 * damping + 0.03))
            for vbin, weight in self._velocity_layers(velocity):
                horizons[pitch, vbin] = max(horizons.get((pitch, vbin), 0.35), horizon)

        for index, (pitch, velocity, start, release) in enumerate(events):
            self._checkpoint(0.65 * index / max(len(events), 1),
                             f"Note {index + 1}/{len(events)}")
            begin = int(start * sr)
            layers = [(self.note_template(pitch, vbin, horizons[pitch, vbin]),
                       vbin, weight) for vbin, weight in self._velocity_layers(velocity)]
            span = min(max(len(template) for template, _, _ in layers), total - begin)
            if span <= 0:
                continue
            segment = np.zeros(span, dtype=np.float32)
            for template, vbin, weight in layers:
                count = min(span, len(template))
                layer_velocity = (vbin + 0.5) * 127.0 / VELOCITY_BINS
                segment[:count] += template[:count] * (weight / self._note_loudness(pitch, layer_velocity))
            peak = float(np.max(np.abs(segment[:max(1, int(sr * 0.3))]), initial=0.0))
            if peak > 0:
                segment *= self._note_loudness(pitch, velocity) / peak

            damp_t60 = phys.damper_t60(pitch)
            if damp_t60 is not None:
                off = int((release - start) * sr)
                if off < span:
                    n = span - off
                    t = np.arange(n, dtype=np.float32) / sr
                    env = np.exp(-t * math.log(1000.0) / damp_t60).astype(np.float32)
                    segment[off:] *= env
                    segment[off:] += self._release_noise(pitch, velocity, n)

            # Place the note by where its strings sit under the microphones.
            tl, tr, gl, gr = phys.stereo_arrival(pitch)
            for channel, delay, gain in ((0, tl, gl), (1, tr, gr)):
                start_i = begin + int(delay * sr)
                n = min(span, total - start_i)
                if n > 0:
                    dry[start_i:start_i + n, channel] += segment[:n] * gain
            mono[begin:begin + span] += segment

        self._checkpoint(0.65, "Sympathetic strings")
        dry += self._sympathetic(mono, pedal_spans, total, events)
        self._checkpoint(0.86, "Soundboard and room")
        stereo = self._finish(dry)
        # A requested duration may end on a ringing treble or held pedal.
        # Give exported clips a short click-free ending without removing tail.
        fade = min(len(stereo), max(2, int(sr * 0.02)))
        if len(stereo):
            stereo[-fade:] *= np.linspace(1.0, 0.0, fade)[:, None]
        self._checkpoint(1.0, "Render complete")
        return stereo

    def _release_noise(self, pitch, velocity, length):
        """The damper felt landing on the string, and the key returning."""
        sr = self.sample_rate
        span = min(length, int(sr * 0.030))
        if span <= 2:
            return np.zeros(length, dtype=np.float32)
        rng = np.random.default_rng(7700 + pitch)
        noise = rng.normal(0.0, 1.0, span).astype(np.float32)
        t = np.arange(span, dtype=np.float32) / sr
        env = np.exp(-t / 0.006).astype(np.float32)
        out = np.zeros(length, dtype=np.float32)
        out[:span] = noise * env * 0.004 * (0.4 + 0.6 * velocity / 127.0)
        return out

    # The top strings carry no dampers, so they ring whenever anything is
    # played.  Modelling them as full-Q resonators driven by the entire mix
    # badly overstates that: on unpedalled material they become an audible
    # chime at fixed chromatic pitches, ringing for seconds after every note.
    # Real top strings are short, stiff and lightly coupled.  This scales how
    # much of that always-on path survives; the pedalled bank is unaffected.
    UNDAMPED_ALWAYS = 0.12

    # How loud the pedalled sympathetic bank sits under the struck notes.
    # This is the "sustain level": raising it thickens pedalled passages,
    # lowering it leaves the notes clearer while the strings still ring.
    # `--sustain-level` scales it; 1.0 is the tuned value.
    SYMPATHETIC_LEVEL = 0.20

    def _sympathetic(self, mono, pedal_spans, total, events=(), level=None):
        """Strings ringing in sympathy with the ones that were struck.

        `level` defaults to SYMPATHETIC_LEVEL so it can be tuned per render.

        Two separate mechanisms:

        * With the damper pedal raised, every string in the instrument is free
          to be driven.  No amount of reverb imitates this, because the
          response is tuned to the instrument's own scale.
        * The top octave and a half of a grand carry **no dampers at all**, so
          those strings ring whenever anything is played, pedal or not.  This
          runs constantly and is a large part of why a real grand's treble
          sounds alive rather than like isolated decaying tones.
        """
        if level is None:
            level = self.SYMPATHETIC_LEVEL
        if not np.isfinite(level) or level < 0:
            raise ValueError("sympathetic level must be finite and non-negative")
        if level == 0 or not np.any(mono):
            return np.zeros((total, 2), dtype=np.float32)
        from scipy.signal import lfilter

        sr = self.sample_rate
        gate = np.zeros(total, dtype=np.float64)
        for down, up in pedal_spans:
            a = max(0, int(down * sr))
            b = min(total, int(up * sr))
            if b > a:
                gate[a:b] = 1.0
        pedalled_any = bool(gate.any())
        if pedalled_any:
            # Soften the gate so the pedal does not click.
            win = max(4, int(sr * 0.05))
            gate = causal_smooth(gate, win)

        # Drive the bank through a high-pass.  Sympathetic coupling runs
        # through the bridge, which is a poor path for the lowest frequencies,
        # and letting the bass drive the whole bank just piles low energy back
        # into the mix as mud.
        from scipy.signal import butter
        source = lfilter(*butter(2, 150.0 / (sr * 0.5), btype="high"),
                         mono.astype(np.float64))
        pedalled = source * gate

        # A string cannot ring in sympathy with itself.  Driving every
        # resonator from the whole mix means a played note pumps the resonator
        # tuned to its own pitch, adding a second, longer-ringing copy of that
        # note -- which is heard as a bell over the music.  Each string is
        # therefore silenced as a sympathetic resonator for as long as it is
        # the one being played.
        busy = {}
        for pitch, _velocity, start, release in events:
            busy.setdefault(pitch, []).append((start, release))

        def free_gate(pitch):
            spans_for = busy.get(pitch)
            if not spans_for:
                return None
            g = np.ones(total)
            damp = phys.damper_t60(pitch)
            hold = (4.0 * damp) if damp is not None else 6.0
            for start, release in spans_for:
                a = max(0, int(start * sr))
                b = min(total, int((release + hold) * sr))
                if b > a:
                    g[a:b] = 0.0
            win = max(4, int(sr * 0.02))
            return causal_smooth(g, win)
        # The pedalled bank and the always-on top strings are accumulated
        # separately.  They have to be, because the bank is peak-normalised at
        # the end -- scaling the *drive* of the undamped strings would simply be
        # undone by that normalisation, which is what made the first attempt at
        # this have no effect whatsoever.
        out = np.zeros((total, 2), dtype=np.float64)
        out_undamped = np.zeros((total, 2), dtype=np.float64)

        for pitch in range(24, 109):
            self._checkpoint(0.65 + 0.20 * (pitch - 24) / 85,
                             f"Resonance {pitch - 23}/85")
            f0 = float(phys.fundamental(pitch))
            if f0 > 4500.0:
                break
            undamped = pitch >= phys.UNDAMPED_FROM_MIDI
            if not undamped and not pedalled_any:
                continue
            # The top strings have no dampers, so they ring whenever anything
            # is played.  Driving them at full level from the whole mix badly
            # overstates that: with the pedal up they become an audible chime
            # at fixed chromatic pitches, ringing for seconds after each note,
            # because the upper partials of the notes below keep feeding them.
            # Real top strings are short, stiff and weakly coupled.
            drive = source if undamped else pedalled

            # A string cannot ring in sympathy with itself.  While a pitch is
            # being played, its own resonator is that same string, so leaving
            # it driven adds a second, longer-ringing copy of the note.
            own = free_gate(pitch)
            if own is not None:
                drive = drive * own

            tl, tr, gl, gr = phys.stereo_arrival(pitch)
            # Sympathetic strings sit on the board like any other, so they
            # reach the two microphones at different times.  Using only the
            # level difference leaves this whole layer nearly mono and narrows
            # the image the struck notes just established.
            dl, dr = int(tl * sr), int(tr * sr)
            for mult, weight in ((1.0, 1.0), (2.0, 0.4), (3.0, 0.18)):
                freq = f0 * mult
                if freq >= 0.45 * sr:
                    continue
                w = 2.0 * np.pi * freq / sr
                # A struck string's own T60 sets how long it rings in sympathy.
                r = math.exp(-math.log(1000.0) / (float(phys.t60(freq)) * sr))
                a = [1.0, -2.0 * r * math.cos(w), r * r]
                gain = weight * (0.55 if undamped else 1.0)
                if undamped:
                    resp = lfilter([gain * (1.0 - r) * math.sin(w)], a, drive)
                else:
                    resp = pedal_resonator(drive, sr, pedal_spans, freq,
                                           float(phys.t60(freq)), phys.damper_t60(pitch), gain)
                target = out_undamped if undamped else out
                if dl < total:
                    target[dl:, 0] += resp[:total - dl] * gl
                if dr < total:
                    target[dr:, 1] += resp[:total - dr] * gr

        reference = float(np.abs(out).max())
        if reference <= 0.0:
            reference = float(np.abs(out_undamped).max())
        if reference > 0.0:
            scale = level * float(np.abs(mono).max()) / reference
            out = out * scale + out_undamped * scale * self.UNDAMPED_ALWAYS
        return out.astype(np.float32)

    def _finish(self, dry):
        left = self.soundboard.process(dry[:, 0].astype(np.float64), pickup="left", mix=self.body_mix)
        right = self.soundboard.process(dry[:, 1].astype(np.float64), pickup="right", mix=self.body_mix)
        stereo = np.stack([left, right], axis=1)

        stereo = self._room(stereo, mix=self.room_mix)
        peak = float(np.abs(stereo).max())
        if peak > 0.0:
            # Optional fixed gain preserves loudness between separate renders;
            # overload protection attenuates the entire clip without clipping.
            gain = 0.89 / peak if self.normalize else min(0.25, 0.98 / peak)
            stereo *= gain
        return stereo

    def _room(self, stereo, mix=ROOM_MIX, room_size=0.86, damping=0.18,
              pre_delay_s=0.018):
        """Recital hall around the instrument.

        Parallel damped comb filters build the tail density and series allpass
        sections smear it, which is the standard way to get a tail that sounds
        like a room rather than a set of discrete echoes.  Summing a handful of
        delayed copies -- the previous approach here -- leaves audible repeats
        and a metallic flutter.

        Each comb's feedback runs through a one-pole low-pass, so the tail
        loses treble as it decays.  A tail that keeps its treble sounds like a
        spring, not a hall.
        """
        sr = self.sample_rate
        n = len(stereo)
        if not 0 <= mix <= 1 or not 0 <= room_size < 1 or not 0 <= damping < 1 or pre_delay_s < 0:
            raise ValueError("invalid room parameters")
        if mix == 0 or not np.any(stereo):
            return stereo.copy()

        # Mutually prime lengths, so repeats never line up into a ringing pitch.
        comb_ref = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617]
        allpass_ref = [556, 441, 341, 225]
        scale = sr / 44100.0
        combs = [max(4, int(d * scale)) for d in comb_ref]
        allpasses = [max(4, int(d * scale)) for d in allpass_ref]

        pre = int(sr * pre_delay_s)
        wet = np.zeros((n, 2))

        for channel in range(2):
            source = np.zeros(n)
            if pre < n:
                source[pre:] = stereo[:-pre, channel] if pre else stereo[:, channel]
            else:
                continue
            # Offset one side so the two tails are different rooms, not one
            # room copied.
            offset = 0 if channel == 0 else max(1, round(23 * scale))
            acc = np.zeros(n)
            for length in combs:
                self._checkpoint()
                d = length + offset
                # Damped comb: y[n] = x[n] + g * lowpass(y[n-d]).
                # Folding the one-pole into the recursion gives
                #   (1 - damp z^-1) y - g(1-damp) z^-d y = (1 - damp z^-1) x
                acc += damped_comb(source, d, room_size, damping)
            acc /= len(combs)

            for length in allpasses:
                self._checkpoint()
                d = length + offset
                acc = allpass(acc, d)
            wet[:, channel] = acc

        # Normalise by the network's own gain, not by the dry peak.  A comb
        # with feedback g has a DC gain of 1/(1-g) -- about 7 here -- so this
        # brings the tail to unity.  Rescaling to the dry peak instead makes
        # the reverb level depend on the material: on a single short note the
        # tail gets lifted until the note audibly swells after its attack.
        #
        # This normalisation is exact only at DC.  The damped comb has less
        # gain as frequency rises, so heavy damping leaves the whole wet path
        # bass-tilted and costs the mix its 1-4 kHz presence -- which is where
        # a piano sings.  Damping is kept light for that reason.
        wet *= (1.0 - room_size) * ROOM_GAIN
        return (1.0 - mix) * stereo + mix * wet


# ---------------------------------------------------------------------------
# MIDI / IO
# ---------------------------------------------------------------------------
from midi_events import load_events, load_performance, clip_events


def demo_events():
    """A short phrase that exposes attack, decay, unison beating and register."""
    events = []
    t = 0.0
    for pitch in (28, 40, 52, 64, 76, 88):
        events.append((pitch, 96, t, t + 1.6))
        t += 0.55
    t += 1.4
    for pitch, vel in ((60, 30), (60, 70), (60, 110)):
        events.append((pitch, vel, t, t + 1.2))
        t += 1.6
    t += 0.8
    for pitch in (48, 55, 60, 64, 67, 72):
        events.append((pitch, 84, t, t + 4.0))
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--midi", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sustain-all", action="store_true")
    parser.add_argument("--sustain-level", type=float, default=1.0,
                        help="scale the pedalled sympathetic bank; "
                             "1.0 is the tuned value, lower is drier")
    parser.add_argument("--rate", "--sample-rate", type=int, default=48000)
    parser.add_argument("--seconds", "--duration", type=float,
                        help="clip the MIDI performance at this time, then add the tail")
    parser.add_argument("--tail", type=float, default=4.0, help="release/room tail in seconds")
    parser.add_argument("--room-mix", type=float, default=ROOM_MIX)
    parser.add_argument("--body-mix", type=float, default=BODY_MIX)
    parser.add_argument("--no-normalize", action="store_true", help="preserve dynamics between files with fixed gain")
    parser.add_argument("--cache-mb", type=float, default=256.0)
    parser.add_argument("--stats", type=Path, help="write render settings and audio measurements as JSON")
    args = parser.parse_args()
    if args.out.suffix.lower() != ".wav":
        parser.error("--out must end in .wav")
    if args.stats and args.stats.suffix.lower() != ".json":
        parser.error("--stats must end in .json")
    if args.midi and args.out.resolve() == args.midi.resolve():
        parser.error("output must differ from the input MIDI")
    if args.stats and (args.stats.resolve() == args.out.resolve()
                       or (args.midi and args.stats.resolve() == args.midi.resolve())):
        parser.error("--stats must differ from MIDI and WAV paths")
    if args.seconds is not None and (not np.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("--seconds must be positive and finite")
    if not np.isfinite(args.tail) or args.tail < 0:
        parser.error("--tail must be finite and non-negative")
    if not np.isfinite(args.sustain_level) or args.sustain_level < 0:
        parser.error("--sustain-level must be finite and non-negative")

    last_progress = [-1]
    def report(fraction, message):
        percent = int(fraction * 100)
        if percent != last_progress[0]:
            print(f"PROGRESS {percent} {message}", flush=True)
            last_progress[0] = percent

    started = time.perf_counter()
    try:
        piano = GrandPiano(args.rate, room_mix=args.room_mix, body_mix=args.body_mix,
                           normalize=not args.no_normalize, max_cache_mb=args.cache_mb,
                           progress=report)
    except (ValueError, OverflowError) as error:
        parser.error(str(error))
    piano.SYMPATHETIC_LEVEL *= args.sustain_level
    warnings = []
    if args.midi:
        try:
            performance = load_performance(args.midi, args.sustain_all)
        except ValueError as error:
            parser.error(str(error))
        events, spans = performance.events, performance.pedal_spans
        warnings = list(performance.warnings)
        for warning in warnings:
            print(f"WARNING {warning}", flush=True)
    else:
        events, spans = demo_events(), []
    if args.seconds is not None:
        events, spans = clip_events(events, spans, args.seconds)
    print(f"rendering {len(events)} notes, {len(spans)} pedal spans, "
          f"sustain level {piano.SYMPATHETIC_LEVEL:.3f}", flush=True)
    stereo = piano.render(events, spans, tail=args.tail)
    write_wav(args.out, stereo, args.rate)
    elapsed = time.perf_counter() - started
    metrics = audio_metrics(stereo, args.rate)
    metrics.update({"midi": str(args.midi) if args.midi else None,
                    "notes": len(events), "pedal_spans": len(spans),
                    "midi_warnings": warnings,
                    "render_seconds": elapsed, "cache_mb": piano.cache_bytes / 1024**2,
                    "cache_hits": piano.cache_hits, "cache_misses": piano.cache_misses,
                    "settings": {"seconds": args.seconds, "tail": args.tail,
                                 "room_mix": args.room_mix, "body_mix": args.body_mix,
                                 "sustain_all": args.sustain_all, "sustain_level": args.sustain_level,
                                 "normalize": not args.no_normalize}})
    if args.stats:
        args.stats.parent.mkdir(parents=True, exist_ok=True)
        args.stats.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}  ({len(stereo)/args.rate:.2f} s audio in {elapsed:.2f} s; "
          f"peak {metrics['peak_dbfs']:.2f} dBFS)", flush=True)


if __name__ == "__main__":
    main()
