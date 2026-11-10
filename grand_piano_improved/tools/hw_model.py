#!/usr/bin/env python3
"""Hardware model: turn the physics into a list of resonators and a ROM.

Every component of the sound in `render_model.py` has the same form:

    a * exp(-lambda t) * sin(2 pi f t + phi)

which is exactly the free response of a damped two-state resonator started
with the right initial conditions.  So the RTL is a bank of such resonators,
loaded at note-on, stepped once per audio sample, and summed.  Partials, the
two polarisations, the detuned unisons, the duplex segments and the
longitudinal modes are all just entries in that list.

Two things are deliberately *not* resonators, because they are common to every
component of a note and so factor out of the sum:

  * the hammer contact ramp, and
  * the damper envelope after note-off.

Both become a per-voice gain, which is exact rather than an approximation --
`render_model.py` applies the same ramp to every partial of a note.

Fixed point
-----------
The representation matters more than anything else here, and it is where the
older `modal_piano` in this workspace came apart -- its notes were dark and its
release behaviour wrong because a direct-form resonator encodes pitch as the
difference between two nearly equal numbers.

Two quantities are near-degenerate if stored naively:

  * the rotation coefficient cos(theta).  At 27.5 Hz, theta is 3.6e-3, so
    cos(theta) = 0.9999935.  In Q1.16 that rounds to 1.0 and the note never
    moves.  Stored instead as k = 2 sin(theta/2) -- the "magic circle" form --
    the coefficient is itself small, so a fixed-point word gives it good
    *relative* precision.
  * the decay factor r = exp(-lambda/fs).  For a 24 s T60 this is
    1 - 6.0e-6, which is below one LSB of Q1.16 -- the partial would either
    never decay or decay 2.5x too fast.  Stored instead as the loss
    lambda = 1 - r, which is small and again keeps its relative precision.

Storing the small quantity rather than the near-unity one is the whole trick.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import piano_physics as phys

# --------------------------------------------------------------------------
# Fixed-point formats
# --------------------------------------------------------------------------
STATE_W, STATE_Q = 32, 28      # Q3.28, range +-8
COEF_W, COEF_Q = 32, 28        # k = 2 sin(theta/2), Q3.28
LOSS_W, LOSS_Q = 32, 30        # lambda = 1 - exp(-rate/fs), Q1.30
GAIN_W, GAIN_Q = 32, 30        # per-voice envelope gains, Q1.30
SAMPLE_RATE = 48000

# --------------------------------------------------------------------------
# Hardware sizing
# --------------------------------------------------------------------------
# A pedalled piano piece needs far more polyphony than a synthesiser patch:
# demo-2 has 27 notes sounding at once, and 43 once their ring-out is counted.
# Eight voices stole the oldest note constantly, and since bass notes are
# struck first and held longest, the bass was what disappeared.
NUM_VOICES = 28
SLOTS_PER_VOICE = 128
# Two slots are stepped per clock, so a pass costs TOTAL_SLOTS/2 cycles and
# still fits the 2083-cycle budget at 100 MHz / 48 kHz.
SLOTS_PER_CYCLE = 2
VELOCITY_LAYERS = 4
# Longest a single note is rendered for in the fast model.
MAX_NOTE_SECONDS = 16.0

# Reductions against the floating-point model, to fit the slot budget.
# A note costs 3*TWIN + CAP + 3 slots: partials below the twin limit carry two
# strings and two polarisations, the rest are single, and a note gets at most
# three extras (duplex above C3 or longitudinal below A2 -- never both).  That
# is 127 of the 128 available.
#
# Twin polarisation is favoured over raw partial count because it is what makes
# the *sustain* live: above the limit a partial is a static sinusoid with no
# double decay and no beating, and that is what is left once the attack has
# gone.
HW_UNISON_PARTIALS = 24        # render_model uses 44
HW_TWIN_PARTIALS = 24          # render_model uses 40
HW_MAX_STRINGS = 2             # render_model uses up to 3
HW_PARTIAL_CAP = 52            # render_model uses 64


def _sat(value, width):
    lo, hi = -(1 << (width - 1)), (1 << (width - 1)) - 1
    return int(max(lo, min(hi, value)))


def to_fixed(value, frac, width):
    return _sat(int(round(float(value) * (1 << frac))), width)


# --------------------------------------------------------------------------
# Partial phases
# --------------------------------------------------------------------------
# Partials start at scattered phases so a note's onset is not a single huge
# in-phase spike.  Any decorrelated sequence does the job, but the RTL now
# generates its own coefficient tables at elaboration and has to arrive at the
# *same* numbers, so this uses SplitMix64 -- eight lines of 64-bit arithmetic
# that SystemVerilog reproduces exactly -- rather than numpy's PCG64, whose
# seeding would have to be reimplemented to match.
_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15


def phase_table(seed, count):
    """`count` phases in [0, 2*pi), from a seed.  Mirrored in the SystemVerilog."""
    state = seed & _MASK64
    out = []
    for _ in range(count):
        state = (state + _SPLITMIX_GAMMA) & _MASK64
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
        z ^= z >> 31
        out.append((z >> 11) * (2.0 ** -53) * 2.0 * math.pi)
    return out


# --------------------------------------------------------------------------
# Component list
# --------------------------------------------------------------------------
def components(pitch, sample_rate=SAMPLE_RATE):
    """Frequencies, decay rates and phases for one note.

    Velocity does not appear here: it changes only amplitudes, so the resonator
    coefficients are shared by every velocity layer and stored once.
    """
    count = min(HW_PARTIAL_CAP, phys.partial_count(pitch, sample_rate))
    freqs = phys.partial_frequencies(pitch, count)
    strings = min(HW_MAX_STRINGS, phys.strings_per_note(pitch))
    phases = phase_table(9000 + pitch, count)

    out = []   # (freq, rate, phase, kind, index)
    for n in range(count):
        base = float(freqs[n])
        rate = float(phys.decay_rate(base))
        voices = strings if n < HW_UNISON_PARTIALS else 1
        for s in range(voices):
            cents = (phys.unison_detune_cents(pitch, s, strings)
                     if n < HW_UNISON_PARTIALS else 0.0)
            freq = base * (2.0 ** (cents / 1200.0))
            phase = float(phases[n]) + s * 1.7
            scale = 1.0 / math.sqrt(voices)
            if n < HW_TWIN_PARTIALS:
                out.append((freq, rate / phys.PROMPT_T60_SCALE, phase,
                            "prompt", n, scale * phys.PROMPT_LEVEL))
                out.append((freq * (1.0 + phys.POLARISATION_DETUNE_RATIO),
                            rate * phys.AFTER_T60_SCALE, phase + 0.9,
                            "after", n, scale * phys.AFTER_LEVEL))
            else:
                out.append((freq, rate, phase, "plain", n, scale))

    for k, ratio in enumerate(phys.DUPLEX_RATIOS):
        freq = float(phys.fundamental(pitch)) * ratio
        if pitch >= phys.DUPLEX_FROM_MIDI and freq < 0.45 * sample_rate:
            out.append((freq, math.log(1000.0) / phys.DUPLEX_T60,
                        0.7 + 1.3 * k, "duplex", k, 1.0))

    ghost_f, _ = phys.longitudinal_partials(
        pitch, 127, phys.partial_frequencies(pitch, 8))
    for k, freq in enumerate(ghost_f):
        if freq < 0.45 * sample_rate:
            out.append((float(freq),
                        float(phys.decay_rate(freq)) / phys.LONGITUDINAL_T60_SCALE,
                        0.31, "longitudinal", k, 1.0))

    return out[:SLOTS_PER_VOICE]


def amplitudes(pitch, velocity, comps, sample_rate=SAMPLE_RATE):
    """Amplitude of each component at a given velocity."""
    count = min(HW_PARTIAL_CAP, phys.partial_count(pitch, sample_rate))
    partial_amp = phys.partial_amplitudes(pitch, velocity, count)
    freqs = phys.partial_frequencies(pitch, count)
    partial_amp = partial_amp * phys.radiation_gain(freqs)

    dup_f, dup_a = phys.duplex_partials(pitch, velocity, sample_rate)
    gh_f, gh_a = phys.longitudinal_partials(
        pitch, velocity, phys.partial_frequencies(pitch, 8))

    out = []
    for freq, rate, phase, kind, index, scale in comps:
        if kind in ("prompt", "after", "plain"):
            out.append(float(partial_amp[index]) * scale)
        elif kind == "duplex":
            out.append(float(dup_a[index]) if index < len(dup_a) else 0.0)
        else:
            out.append(float(gh_a[index]) if index < len(gh_a) else 0.0)
    return np.array(out)


def note_loudness(pitch, velocity):
    v = np.clip(velocity / 127.0, 0.02, 1.0)
    bass = 1.0 + 0.35 * np.clip((52 - pitch) / 31.0, 0.0, 1.0)
    treble = 1.0 - 0.24 * np.clip((pitch - 52) / 56.0, 0.0, 1.0)
    return float(0.9 * (v ** 1.25) * bass * treble)


def normalised_amplitudes(pitch, velocity, comps, sample_rate=SAMPLE_RATE):
    """Amplitudes scaled so the summed note peaks at its intended loudness.

    `render_model.py` normalises the rendered note buffer; doing the same
    scaling here keeps the RTL's output level identical without the hardware
    needing to know the peak.
    """
    amps = amplitudes(pitch, velocity, comps, sample_rate)
    if not len(amps) or amps.max() <= 0:
        return amps
    # Peak of the sum over the first 300 ms, where every note peaks.
    span = int(0.3 * sample_rate)
    t = np.arange(span) / sample_rate
    total = np.zeros(span)
    for (freq, rate, phase, _, _, _), amp in zip(comps, amps):
        if amp <= 0.0:
            continue
        total += amp * np.exp(-rate * t) * np.sin(2 * np.pi * freq * t + phase)
    peak = float(np.abs(total).max())
    if peak <= 0.0:
        return amps
    return amps * (note_loudness(pitch, velocity) / peak)


# --------------------------------------------------------------------------
# Fixed-point coefficients
# --------------------------------------------------------------------------
def coefficients(freq, rate, sample_rate=SAMPLE_RATE):
    """(k, loss) for one resonator, as real numbers before quantisation."""
    theta = 2.0 * math.pi * freq / sample_rate
    k = 2.0 * math.sin(theta / 2.0)
    loss = 1.0 - math.exp(-rate / sample_rate)
    return k, loss


def velocity_for_layer(layer):
    return (layer + 0.5) * 127.0 / VELOCITY_LAYERS


# --------------------------------------------------------------------------
# ROM
# --------------------------------------------------------------------------
def build_rom(sample_rate=SAMPLE_RATE):
    """Pack every note's resonator list into flat arrays."""
    base, count = [], []
    k_list, loss_list, cphi_list, sphi_list = [], [], [], []
    amp_layers = [[] for _ in range(VELOCITY_LAYERS)]

    for pitch in range(phys.LOWEST_MIDI, phys.HIGHEST_MIDI + 1):
        comps = components(pitch, sample_rate)
        base.append(len(k_list))
        count.append(len(comps))

        layer_amps = [normalised_amplitudes(pitch, velocity_for_layer(v),
                                            comps, sample_rate)
                      for v in range(VELOCITY_LAYERS)]

        for i, (freq, rate, phase, _, _, _) in enumerate(comps):
            k, loss = coefficients(freq, rate, sample_rate)
            k_list.append(to_fixed(k, COEF_Q, COEF_W))
            loss_list.append(to_fixed(loss, LOSS_Q, LOSS_W))
            cphi_list.append(to_fixed(math.cos(phase), COEF_Q, COEF_W))
            sphi_list.append(to_fixed(math.sin(phase), COEF_Q, COEF_W))
            for v in range(VELOCITY_LAYERS):
                amp_layers[v].append(to_fixed(layer_amps[v][i], STATE_Q, STATE_W))

    # Per-note envelope coefficients.
    attack, damper, undamped = [], [], []
    for pitch in range(phys.LOWEST_MIDI, phys.HIGHEST_MIDI + 1):
        tau = phys.contact_time_s(pitch, 100) / 3.0
        attack.append(to_fixed(1.0 - math.exp(-1.0 / (tau * sample_rate)),
                               GAIN_Q, GAIN_W))
        t60 = phys.damper_t60(pitch)
        if t60 is None:
            damper.append(to_fixed(1.0, GAIN_Q, GAIN_W))
            undamped.append(1)
        else:
            damper.append(to_fixed(math.exp(-math.log(1000.0) / (t60 * sample_rate)),
                                   GAIN_Q, GAIN_W))
            undamped.append(0)

    return {
        "base": base, "count": count,
        "k": k_list, "loss": loss_list, "cphi": cphi_list, "sphi": sphi_list,
        "amp": amp_layers,
        "attack": attack, "damper": damper, "undamped": undamped,
        "entries": len(k_list),
    }


# --------------------------------------------------------------------------
# Everything after the strings
# --------------------------------------------------------------------------
# Sizing for the stages that turn a bank of strings into an instrument in a
# room.  All of them are resonators too, so they use the same magic-circle
# form: a soundboard mode with Q=70 at 46 Hz has r = 0.99996, which is just as
# degenerate in a direct form as a string partial is.
DELAY_LEN = 64                 # per-note stereo arrival, samples
SYMP_SLOTS = 256               # sympathetic strings
BODY_MODES = 16                # soundboard modes, per channel
REVERB_COMBS = (1214, 1293, 1390, 1476, 1548, 1623, 1695, 1760)
REVERB_ALLPASS = (605, 480, 371, 245)
REVERB_STEREO_OFFSET = 23
ROOM_SIZE_Q = 0.86
ROOM_DAMPING_Q = 0.18
BODY_MIX_Q = 0.20
ROOM_MIX_Q = 0.34
ROOM_GAIN_Q = 8.0
# render_model peak-normalises the sympathetic bank to a fixed fraction of the
# dry signal, which caps it however long the pedal is held.  The RTL cannot look
# ahead to do that, so its bank is a plain sum -- and with the pedal down for
# twelve seconds the resonators, which carry the struck string's own T60 of up
# to forty seconds in the bass, accumulate and then ring on long after the notes
# stop.  Measured against the Python render of the same passage the RTL tail sat
# 16 dB high; this is that 16 dB, applied as a fixed content-independent gain.
SYMP_LEVEL_Q = 0.032
# Lifting the pedal drops the dampers onto the sympathetic strings and stops
# them.  Ceasing to *drive* them is not the same thing: left undriven they carry
# on with the struck string's own T60, which is 21.7 s at A#2, so the wash
# outlives the music by many seconds.  This is the damper landing on them.
SYMP_DAMPER_T60 = 0.30
# The top strings have no dampers and so ring constantly.  Modelled at full
# level they become an audible chime on unpedalled material; this scales that
# always-on path only, matching render_model.GrandPiano.UNDAMPED_ALWAYS.
UNDAMPED_ALWAYS_Q = 0.12
SYMP_HIGHPASS_HZ = 150.0
# Master headroom.  render_model.py normalises the finished render to a target
# peak; hardware cannot look ahead, so it needs standing headroom instead.
# Without it a dense pedalled passage runs into the rails -- demo-2 clipped
# 0.145% of its samples, which is exactly the harshness this is meant to avoid.
MASTER_GAIN_Q = 1.20           # -10.5 dB, times the 4x headroom shift below
# The effects chain runs at a quarter scale so the reverb combs, whose DC gain
# is 1/(1-feedback) = 7.14, cannot drive their stored state past what Q3.28
# holds.  Purely headroom: the master gain carries the same factor back.
FX_HEADROOM_SHIFT = 2

# How long the dampers take to reach the strings after the pedal is released.
# They are felt-covered masses on levers, not switches.  This matters because
# pianists re-pedal constantly: demo-2 releases and re-presses the pedal in
# 1.92 ms, and treating that as instantaneous damping kills every sounding
# note.  A real damper has barely started to move in that time.
DAMPER_TRAVEL_MS = 28.0


def driven_gain(loss, weight):
    """Input gain for a driven magic-circle resonator with peak gain `weight`.

    A resonator driven once per sample settles to roughly g/(2*lambda) at its
    own frequency, so scaling the drive by 2*lambda normalises the peak.
    """
    return 2.0 * loss * weight


def stereo_rom(sample_rate=SAMPLE_RATE):
    """Per-note arrival delay and level at each microphone.

    Only the *difference* between the two arrivals matters, so the common part
    is removed -- that turns a 200-sample buffer into a 64-sample one.
    """
    raw = []
    for pitch in range(phys.LOWEST_MIDI, phys.HIGHEST_MIDI + 1):
        tl, tr, gl, gr = phys.stereo_arrival(pitch)
        raw.append((tl * sample_rate, tr * sample_rate, gl, gr))
    floor = min(min(l, r) for l, r, _, _ in raw)
    out = []
    for l, r, gl, gr in raw:
        dl = int(round(l - floor))
        dr = int(round(r - floor))
        assert 0 <= dl < DELAY_LEN and 0 <= dr < DELAY_LEN, (dl, dr)
        out.append((dl, dr, to_fixed(gl, GAIN_Q, GAIN_W), to_fixed(gr, GAIN_Q, GAIN_W)))
    return out


def body_rom(path=None, sample_rate=SAMPLE_RATE):
    """Soundboard body modes, as driven resonators for each pickup."""
    if path is None:
        path = Path(__file__).resolve().parents[1] / "rtl" / "soundboard_modes.json"
    modes = json.loads(Path(path).read_text(encoding="utf-8"))["modes"]
    total = sum(abs(m["weight"] * m.get("left", 1.0)) for m in modes) or 1.0
    total_r = sum(abs(m["weight"] * m.get("right", 1.0)) for m in modes) or 1.0

    rows = []
    for i in range(BODY_MODES):
        if i >= len(modes):
            rows.append((0, 0, 0, 0))
            continue
        m = modes[i]
        hz = float(m["hz"])
        # Q relates to the amplitude decay rate as alpha = pi*f/Q.
        alpha = math.pi * hz / max(float(m["q"]), 1e-3)
        k, loss = coefficients(hz, alpha, sample_rate)
        gl = driven_gain(loss, m["weight"] * m.get("left", 1.0) / total)
        gr = driven_gain(loss, m["weight"] * m.get("right", 1.0) / total_r)
        rows.append((to_fixed(k, COEF_Q, COEF_W),
                     to_fixed(loss, LOSS_Q, LOSS_W),
                     to_fixed(gl, COEF_Q, COEF_W),
                     to_fixed(gr, COEF_Q, COEF_W)))
    return rows


def sympathetic_rom(sample_rate=SAMPLE_RATE):
    """Undamped and pedalled strings that ring in sympathy.

    Entries carry a flag for the top strings, which have no damper at all and
    so ring whether or not the pedal is down.
    """
    rows = []
    for pitch in range(24, phys.HIGHEST_MIDI + 1):
        f0 = float(phys.fundamental(pitch))
        if f0 > 4500.0:
            break
        undamped = 1 if pitch >= phys.UNDAMPED_FROM_MIDI else 0
        _, _, gl, gr = phys.stereo_arrival(pitch)
        for mult, weight in ((1.0, 1.0), (2.0, 0.4), (3.0, 0.18)):
            freq = f0 * mult
            if freq >= 0.45 * sample_rate or len(rows) >= SYMP_SLOTS:
                continue
            rate = float(phys.decay_rate(freq))
            k, loss = coefficients(freq, rate, sample_rate)
            gain = driven_gain(loss, weight * (0.55 if undamped else 1.0))
            rows.append((to_fixed(k, COEF_Q, COEF_W),
                         to_fixed(loss, LOSS_Q, LOSS_W),
                         to_fixed(gain, COEF_Q, COEF_W),
                         to_fixed(gl, GAIN_Q, GAIN_W),
                         to_fixed(gr, GAIN_Q, GAIN_W),
                         undamped, pitch))
    while len(rows) < SYMP_SLOTS:
        rows.append((0, 0, 0, 0, 0, 0, 0))
    return rows


def _write_hex(path, values, width):
    mask = (1 << width) - 1
    digits = (width + 3) // 4
    path.write_text("\n".join(f"{v & mask:0{digits}x}" for v in values) + "\n",
                    encoding="utf-8")


def export(rom, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_hex(out_dir / "rom_k.hex", rom["k"], COEF_W)
    _write_hex(out_dir / "rom_loss.hex", rom["loss"], LOSS_W)
    _write_hex(out_dir / "rom_cphi.hex", rom["cphi"], COEF_W)
    _write_hex(out_dir / "rom_sphi.hex", rom["sphi"], COEF_W)
    for v in range(VELOCITY_LAYERS):
        _write_hex(out_dir / f"rom_amp{v}.hex", rom["amp"][v], STATE_W)
    _write_hex(out_dir / "rom_base.hex", rom["base"], 32)
    _write_hex(out_dir / "rom_count.hex", rom["count"], 32)
    _write_hex(out_dir / "rom_attack.hex", rom["attack"], GAIN_W)
    _write_hex(out_dir / "rom_damper.hex", rom["damper"], GAIN_W)
    _write_hex(out_dir / "rom_undamped.hex", rom["undamped"], 8)

    stereo = stereo_rom()
    _write_hex(out_dir / "rom_delay_l.hex", [r[0] for r in stereo], 16)
    _write_hex(out_dir / "rom_delay_r.hex", [r[1] for r in stereo], 16)
    _write_hex(out_dir / "rom_pan_l.hex",   [r[2] for r in stereo], GAIN_W)
    _write_hex(out_dir / "rom_pan_r.hex",   [r[3] for r in stereo], GAIN_W)

    body = body_rom()
    _write_hex(out_dir / "rom_body_k.hex",    [r[0] for r in body], COEF_W)
    _write_hex(out_dir / "rom_body_loss.hex", [r[1] for r in body], LOSS_W)
    _write_hex(out_dir / "rom_body_gl.hex",   [r[2] for r in body], COEF_W)
    _write_hex(out_dir / "rom_body_gr.hex",   [r[3] for r in body], COEF_W)

    symp = sympathetic_rom()
    _write_hex(out_dir / "rom_symp_k.hex",    [r[0] for r in symp], COEF_W)
    _write_hex(out_dir / "rom_symp_loss.hex", [r[1] for r in symp], LOSS_W)
    _write_hex(out_dir / "rom_symp_gain.hex", [r[2] for r in symp], COEF_W)
    _write_hex(out_dir / "rom_symp_pl.hex",   [r[3] for r in symp], GAIN_W)
    _write_hex(out_dir / "rom_symp_pr.hex",   [r[4] for r in symp], GAIN_W)
    _write_hex(out_dir / "rom_symp_und.hex",  [r[5] for r in symp], 8)
    _write_hex(out_dir / "rom_symp_pitch.hex", [r[6] for r in symp], 8)

    manifest = {
        "entries": rom["entries"],
        "notes": len(rom["base"]),
        "max_components": max(rom["count"]),
        "state_w": STATE_W, "state_q": STATE_Q,
        "coef_w": COEF_W, "coef_q": COEF_Q,
        "loss_w": LOSS_W, "loss_q": LOSS_Q,
        "gain_w": GAIN_W, "gain_q": GAIN_Q,
        "num_voices": NUM_VOICES, "slots_per_voice": SLOTS_PER_VOICE,
        "velocity_layers": VELOCITY_LAYERS,
        "sample_rate": SAMPLE_RATE,
    }
    (out_dir / "rom_manifest.json").write_text(json.dumps(manifest, indent=2),
                                               encoding="utf-8")

    # Emit the RTL parameters from the same source as the ROM, so the two
    # cannot drift apart.
    header = f"""// Generated by tools/hw_model.py -- do not edit.
`ifndef GRAND_PIANO_PARAMS_SVH
`define GRAND_PIANO_PARAMS_SVH

localparam int STATE_W        = {STATE_W};
localparam int STATE_Q        = {STATE_Q};
localparam int COEF_W         = {COEF_W};
localparam int COEF_Q         = {COEF_Q};
localparam int LOSS_W         = {LOSS_W};
localparam int LOSS_Q         = {LOSS_Q};
localparam int GAIN_W         = {GAIN_W};
localparam int GAIN_Q         = {GAIN_Q};

localparam int NUM_VOICES     = {NUM_VOICES};
localparam int SLOTS_PER_VOICE= {SLOTS_PER_VOICE};
localparam int TOTAL_SLOTS    = {NUM_VOICES * SLOTS_PER_VOICE};
localparam int SLOTS_PER_CYCLE= {SLOTS_PER_CYCLE};
localparam int VELOCITY_LAYERS= {VELOCITY_LAYERS};

localparam int ROM_ENTRIES    = {rom['entries']};
localparam int NUM_NOTES      = {len(rom['base'])};
localparam int LOWEST_MIDI    = {phys.LOWEST_MIDI};
localparam int SAMPLE_RATE    = {SAMPLE_RATE};

// Stages after the strings.
localparam int DELAY_LEN      = {DELAY_LEN};
localparam int SYMP_SLOTS     = {SYMP_SLOTS};
localparam int BODY_MODES     = {BODY_MODES};
localparam int NUM_COMBS      = {len(REVERB_COMBS)};
localparam int NUM_ALLPASS    = {len(REVERB_ALLPASS)};
localparam int REVERB_LEN     = {max(REVERB_COMBS) + REVERB_STEREO_OFFSET + 1};
localparam int STEREO_OFFSET  = {REVERB_STEREO_OFFSET};

localparam int BODY_MIX_G     = {to_fixed(BODY_MIX_Q, GAIN_Q, GAIN_W)};
localparam int ROOM_MIX_G     = {to_fixed(ROOM_MIX_Q, GAIN_Q, GAIN_W)};
localparam int ROOM_FEEDBACK  = {to_fixed(ROOM_SIZE_Q, GAIN_Q, GAIN_W)};
localparam int ROOM_DAMPING   = {to_fixed(ROOM_DAMPING_Q, GAIN_Q, GAIN_W)};
// Reverb normalised by the network's own gain, (1-feedback)*ROOM_GAIN, rather
// than by the dry peak -- scaling to the peak makes the level depend on the
// material and swells single notes after their attack.
localparam int ROOM_WET_G     = {to_fixed((1.0 - ROOM_SIZE_Q) * ROOM_GAIN_Q, GAIN_Q, GAIN_W)};
localparam int SYMP_LEVEL_G   = {to_fixed(SYMP_LEVEL_Q, GAIN_Q, GAIN_W)};
localparam int SYMP_HP_A      = {to_fixed(math.exp(-2.0 * math.pi * SYMP_HIGHPASS_HZ / SAMPLE_RATE), GAIN_Q, GAIN_W)};
localparam int UNDAMPED_ALWAYS_G = {to_fixed(UNDAMPED_ALWAYS_Q, GAIN_Q, GAIN_W)};
localparam int SYMP_DAMP_G    = {to_fixed(math.exp(-math.log(1000.0) / (SYMP_DAMPER_T60 * SAMPLE_RATE)), GAIN_Q, GAIN_W)};
localparam int MASTER_GAIN_G  = {to_fixed(MASTER_GAIN_Q, GAIN_Q, GAIN_W)};
localparam int FX_HEADROOM_SHIFT = {FX_HEADROOM_SHIFT};
localparam int DAMPER_TRAVEL  = {int(round(DAMPER_TRAVEL_MS * SAMPLE_RATE / 1000.0))};

`endif
"""
    (out_dir / "grand_piano_params.svh").write_text(header, encoding="utf-8")
    return manifest


# --------------------------------------------------------------------------
# Bit-accurate reference
# --------------------------------------------------------------------------
def simulate_note(pitch, velocity, samples, rom=None, sample_rate=SAMPLE_RATE):
    """Bit-accurate simulation of one note, matching the RTL exactly.

    Vectorised over components, but integer throughout, so the RTL can be
    checked against it sample for sample rather than merely by ear.
    """
    comps = components(pitch, sample_rate)
    amps = normalised_amplitudes(pitch, velocity, comps, sample_rate)
    layer = int(np.clip(velocity * VELOCITY_LAYERS / 128.0, 0, VELOCITY_LAYERS - 1))
    amps = normalised_amplitudes(pitch, velocity_for_layer(layer), comps, sample_rate)

    n = len(comps)
    k = np.array([to_fixed(coefficients(f, r, sample_rate)[0], COEF_Q, COEF_W)
                  for f, r, _, _, _, _ in comps], dtype=np.int64)
    loss = np.array([to_fixed(coefficients(f, r, sample_rate)[1], LOSS_Q, LOSS_W)
                     for f, r, _, _, _, _ in comps], dtype=np.int64)
    x = np.array([to_fixed(a * math.cos(p), STATE_Q, STATE_W)
                  for a, (_, _, p, _, _, _) in zip(amps, comps)], dtype=np.int64)
    y = np.array([to_fixed(a * math.sin(p), STATE_Q, STATE_W)
                  for a, (_, _, p, _, _, _) in zip(amps, comps)], dtype=np.int64)

    attack_a = to_fixed(1.0 - math.exp(
        -1.0 / ((phys.contact_time_s(pitch, velocity) / 3.0) * sample_rate)),
        GAIN_Q, GAIN_W)
    one = 1 << GAIN_Q
    env = 0

    out = np.zeros(samples, dtype=np.int64)
    for i in range(samples):
        # Magic-circle step: x += k*y ; y -= k*x_new.
        x = x + ((k * y) >> COEF_Q)
        y = y - ((k * x) >> COEF_Q)
        # Damping as an explicit small loss.
        x = x - ((loss * x) >> LOSS_Q)
        y = y - ((loss * y) >> LOSS_Q)
        env = env + (((one - env) * attack_a) >> GAIN_Q)
        out[i] = (int(y.sum()) * env) >> GAIN_Q
    return out.astype(np.float64) / (1 << STATE_Q)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parents[1] / "rtl")
    args = parser.parse_args()

    rom = build_rom()
    manifest = export(rom, args.out)
    print(f"resonator entries : {manifest['entries']}")
    print(f"max per note      : {manifest['max_components']} "
          f"(slot budget {SLOTS_PER_VOICE})")
    print(f"voices            : {NUM_VOICES}")
    print(f"ROM written to    : {args.out}")


if __name__ == "__main__":
    main()
