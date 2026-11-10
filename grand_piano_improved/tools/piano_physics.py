#!/usr/bin/env python3
"""Concert-grand piano physics.

Every quantity here is produced by an equation with published acoustic
grounding.  Nothing in this file is derived from a recording, a sample
library, a wavetable, or a convolution impulse response.

References for the model form (constants fitted to published measurements):
  * Fletcher & Rossing, "The Physics of Musical Instruments", ch. 12
  * Conklin, "Design and tone in the mechanoacoustic piano", JASA 1996
  * Askenfelt & Jansson, "From touch to string vibrations", JASA 1990
  * Young, "Inharmonicity of plain wire piano strings", JASA 1952
"""
from __future__ import annotations

import numpy as np

A4_MIDI = 69
A4_HZ = 440.0
LOWEST_MIDI = 21   # A0
HIGHEST_MIDI = 108  # C8


# --------------------------------------------------------------------------
# Pitch
# --------------------------------------------------------------------------
def fundamental(midi):
    """Equal-tempered fundamental in Hz."""
    return A4_HZ * 2.0 ** ((np.asarray(midi, dtype=float) - A4_MIDI) / 12.0)


# --------------------------------------------------------------------------
# Inharmonicity
# --------------------------------------------------------------------------
# Stiff-string partials follow f_n = n*f0*sqrt(1 + B*n^2).  B is smallest in
# the middle of a concert grand and rises steeply at both ends: the bass uses
# thick overwound cores and the top treble uses very short, stiff wire.
# Anchors below are representative concert-grand measurements (Young 1952,
# Conklin 1996); a quadratic in log10(B) versus MIDI number interpolates them.
_B_ANCHOR_MIDI = np.array([21.0, 40.0, 60.0, 88.0, 108.0])
_B_ANCHOR_VALUE = np.array([1.5e-4, 4.0e-5, 7.0e-5, 1.2e-3, 6.0e-3])
_B_FIT = np.polyfit(_B_ANCHOR_MIDI, np.log10(_B_ANCHOR_VALUE), 2)


def inharmonicity(midi):
    """Stiff-string inharmonicity coefficient B."""
    return 10.0 ** np.polyval(_B_FIT, np.asarray(midi, dtype=float))


def partial_frequencies(midi, count):
    """Frequencies of the first `count` partials of a stiff string."""
    n = np.arange(1, count + 1, dtype=float)
    return n * fundamental(midi) * np.sqrt(1.0 + inharmonicity(midi) * n * n)


# --------------------------------------------------------------------------
# Damping / decay
# --------------------------------------------------------------------------
# Partial decay is dominated by air damping and internal friction at low
# frequency and by soundboard radiation at high frequency.  A concert grand
# with the damper raised holds a low partial for tens of seconds while a
# 4 kHz partial is gone in about two.
T60_REFERENCE_HZ = 100.0
T60_AT_REFERENCE = 24.0
# How fast the decay time shortens as partials rise.  This is the single
# strongest control over whether the instrument reads as struck or plucked: a
# pluck loses its upper partials almost immediately, while a piano holds them
# and keeps singing.  Too steep and every note collapses into its fundamental
# within a second.
T60_SLOPE = 0.66
T60_FLOOR = 0.40
T60_CEILING = 40.0


def t60(freq):
    """Time in seconds for a partial at `freq` to fall 60 dB."""
    freq = np.maximum(np.asarray(freq, dtype=float), 1.0)
    value = T60_AT_REFERENCE * (T60_REFERENCE_HZ / freq) ** T60_SLOPE
    return np.clip(value, T60_FLOOR, T60_CEILING)


def decay_rate(freq):
    """Exponential amplitude decay rate in nepers per second."""
    return np.log(1000.0) / t60(freq)


# The two transverse polarisations of a real string are weakly coupled to the
# bridge.  The polarisation that drives the bridge strongly loses energy fast;
# the other leaks slowly and produces the long "aftersound".  Their small
# frequency offset is what makes a piano tone shimmer instead of sit still.
# This double decay is the single most recognisable feature of piano tone.
# The prompt decay must be faster than the aftersound, but not by so much that
# the note dumps most of its energy immediately -- that reads as a pluck rather
# than a struck string.  Measured grands sit nearer a factor of three.
PROMPT_LEVEL = 0.45
AFTER_LEVEL = 0.55
PROMPT_T60_SCALE = 0.55
AFTER_T60_SCALE = 1.0
# The two polarisations see different bridge impedance, so their frequencies
# differ by a small *relative* amount -- not a fixed number of Hz.  A fixed Hz
# offset makes every partial beat at the same rate, and partials beating in
# lockstep produce audible whole-note level pumping instead of shimmer.
POLARISATION_DETUNE_RATIO = 4.0e-4


# --------------------------------------------------------------------------
# Unisons
# --------------------------------------------------------------------------
def strings_per_note(midi):
    """Concert grands use one wound string low, two in the tenor, three up."""
    midi = int(midi)
    if midi <= 27:
        return 1
    if midi <= 40:
        return 2
    return 3


def unison_detune_cents(midi, index, count):
    """Deliberate tuner's detune between unison strings, in cents.

    A piano tuner leaves a few tenths of a cent between unison strings.  Too
    little and the note is lifeless; too much and it sounds out of tune.
    """
    if count == 1:
        return 0.0
    spread = 0.9 + 1.4 * max(0.0, (60.0 - midi) / 39.0)  # wider in the bass
    if count == 2:
        return spread * (-0.5 if index == 0 else 0.5)
    return spread * (-0.5, 0.12, 0.55)[index]


# --------------------------------------------------------------------------
# Hammer
# --------------------------------------------------------------------------
# Felt hammers are nonlinear springs: a hard blow compresses the felt, shortens
# the contact time and shifts energy upward.  Contact also shortens with pitch.
# `brightness_corner` is the corner of the resulting excitation low-pass.
STRIKE_POSITION = 0.125       # hammer strikes near 1/8 of the speaking length
PARTIAL_SLOPE = 1.35          # base spectral tilt of the modal excitation
# A piano hammer is a broad felt pad, not a plectrum, so it cannot fully null a
# partial the way an ideal point strike would.  Deep comb notches sound hollow
# and harpsichord-like.
COMB_FLOOR = 0.20


def contact_corner_hz(midi, velocity):
    """Corner frequency of the hammer excitation low-pass."""
    v = np.clip(np.asarray(velocity, dtype=float) / 127.0, 0.02, 1.0)
    pitch_term = 900.0 * 2.0 ** ((np.asarray(midi, dtype=float) - 21.0) / 27.0)
    return pitch_term * (0.42 + 1.35 * v ** 0.62)


def contact_time_s(midi, velocity):
    """How long the hammer stays in contact with the string.

    This is what separates a struck string from a plucked one.  A pluck
    releases the string instantaneously, so every partial starts at full
    amplitude; a hammer pushes the string for a few milliseconds, so the
    partials ramp up.  Synthesising a piano with instantaneous partial onsets
    gives an unmistakably plucked attack no matter how the spectrum is tuned.

    Contact is longest on the heavy bass strings and shortens both with pitch
    and with a harder blow, because compressed felt is a stiffer spring.
    """
    midi = np.asarray(midi, dtype=float)
    v = np.clip(np.asarray(velocity, dtype=float) / 127.0, 0.02, 1.0)
    base = 0.0048 * 2.0 ** (-(midi - LOWEST_MIDI) / 30.0)
    return base * (1.25 - 0.45 * v)


def strike_position(midi):
    """Relative strike point; grands move it toward the end in the treble."""
    midi = np.asarray(midi, dtype=float)
    return STRIKE_POSITION * (1.0 - 0.30 * np.clip((midi - 60.0) / 48.0, 0.0, 1.0))


def partial_amplitudes(midi, velocity, count):
    """Modal excitation amplitudes for the first `count` partials.

    Three physical factors combine:
      1. the strike-point comb sin(n*pi*p), which notches partials whose node
         sits under the hammer -- an audible, characteristic piano colouration;
      2. the modal energy tilt n**-slope;
      3. the hammer contact low-pass, which is what makes soft playing dull
         and hard playing bright.
    """
    n = np.arange(1, count + 1, dtype=float)
    freqs = partial_frequencies(midi, count)

    comb = np.abs(np.sin(n * np.pi * strike_position(midi)))
    # Never fully null a partial: real hammers are wide and strings are stiff.
    comb = COMB_FLOOR + (1.0 - COMB_FLOOR) * comb

    tilt = n ** -PARTIAL_SLOPE

    corner = contact_corner_hz(midi, velocity)
    lowpass = 1.0 / (1.0 + (freqs / corner) ** 2)

    amps = comb * tilt * lowpass

    # Loudness versus velocity: about 40 dB of usable dynamic range, and the
    # perceived level grows far more slowly than velocity squared.
    v = np.clip(velocity / 127.0, 0.02, 1.0)
    amps = amps * (v ** 1.35)

    return amps


def partial_count(midi, sample_rate=48000.0, cap=64):
    """Number of partials to synthesise: everything below Nyquist, capped."""
    f0 = float(fundamental(midi))
    limit = 0.92 * sample_rate / 2.0
    count = 1
    while count < cap:
        nxt = count + 1
        f = nxt * f0 * np.sqrt(1.0 + float(inharmonicity(midi)) * nxt * nxt)
        if f > limit:
            break
        count = nxt
    return max(1, count)


# --------------------------------------------------------------------------
# Duplex / aliquot scaling
# --------------------------------------------------------------------------
# On a good grand the string does not simply stop at the bridge.  The short
# segment beyond it is deliberately tuned -- aliquot or duplex scaling -- so it
# rings at a harmonic of the speaking length.  It is why the middle and upper
# registers of a concert instrument shimmer instead of sounding like plain
# decaying sinusoids, and it is entirely absent from a naive modal model.
DUPLEX_FROM_MIDI = 48
DUPLEX_RATIOS = (8.02, 12.05, 16.10)   # tuned near harmonics, slightly off
DUPLEX_T60 = 0.55


def duplex_partials(midi, velocity, sample_rate=48000.0):
    """Return (frequencies, amplitudes) of the duplex segment resonances.

    The small offsets from exact harmonics are what make these shimmer: tuned
    dead on, they would simply reinforce partials that already exist.
    """
    midi = float(midi)
    if midi < DUPLEX_FROM_MIDI:
        return np.zeros(0), np.zeros(0)
    v = float(np.clip(velocity / 127.0, 0.0, 1.0))
    register = float(np.clip((midi - DUPLEX_FROM_MIDI) / 40.0, 0.0, 1.0))
    level = 0.030 * (0.35 + 0.65 * register) * v ** 1.3

    f0 = float(fundamental(midi))
    limit = 0.45 * sample_rate
    freqs, amps = [], []
    for k, ratio in enumerate(DUPLEX_RATIOS):
        freq = f0 * ratio
        if freq >= limit:
            continue
        freqs.append(freq)
        amps.append(level / (1.0 + k))
    return np.array(freqs), np.array(amps)


# --------------------------------------------------------------------------
# Damper
# --------------------------------------------------------------------------
# Felt dampers are soft and do not stop a string instantly, and they grip the
# bass far less well than the treble.  Top-octave notes have no damper at all.
DAMPER_T60_BASS = 0.42
DAMPER_T60_TREBLE = 0.11
UNDAMPED_FROM_MIDI = 93


def damper_t60(midi):
    """Decay time once the key is released and the damper lands."""
    if midi >= UNDAMPED_FROM_MIDI:
        return None  # no damper on the top octave
    x = np.clip((midi - LOWEST_MIDI) / (UNDAMPED_FROM_MIDI - LOWEST_MIDI), 0.0, 1.0)
    return DAMPER_T60_BASS + (DAMPER_T60_TREBLE - DAMPER_T60_BASS) * x


# --------------------------------------------------------------------------
# Nonlinear string behaviour
# --------------------------------------------------------------------------
# A hard blow stretches the string enough to raise its tension measurably, so
# the note starts slightly sharp and settles as the amplitude falls.  On a real
# grand this is a few cents in the bass and is a large part of why a forte bass
# note sounds "alive" rather than like a static oscillator.
GLIDE_TAU_S = 0.055


def attack_glide_cents(midi, velocity):
    """Initial pitch rise from tension modulation, in cents."""
    v = np.clip(np.asarray(velocity, dtype=float) / 127.0, 0.0, 1.0)
    register = np.clip((52.0 - np.asarray(midi, dtype=float)) / 31.0, 0.0, 1.0)
    return 9.0 * register * v ** 1.6


# Speaking length of the strings, in metres.  A concert grand runs from about
# two metres at the bottom to a few centimetres at the top, and the bass is
# always a compromise against case length.
_LENGTH_ANCHOR_MIDI = np.array([21.0, 36.0, 48.0, 60.0, 72.0, 84.0, 96.0, 108.0])
_LENGTH_ANCHOR_M = np.array([1.95, 1.30, 0.90, 0.62, 0.31, 0.15, 0.08, 0.05])


def speaking_length(midi):
    """Vibrating string length in metres."""
    return np.exp(np.interp(np.asarray(midi, dtype=float),
                            _LENGTH_ANCHOR_MIDI, np.log(_LENGTH_ANCHOR_M)))


# Steel: longitudinal waves travel at sqrt(E/rho), far faster than the
# transverse waves that carry the pitch.
STEEL_E = 2.0e11      # Pa
STEEL_RHO = 7850.0    # kg/m^3
LONGITUDINAL_SPEED = float(np.sqrt(STEEL_E / STEEL_RHO))   # about 5050 m/s

# Phantom partials belong to the bass, played hard.  Extending them up the
# compass and into ordinary dynamics puts a pure inharmonic tone on every note.
LONGITUDINAL_FROM_MIDI = 45.0        # A2, about 110 Hz
LONGITUDINAL_LEVEL = 0.055
LONGITUDINAL_VELOCITY_POWER = 3.5


def longitudinal_partials(midi, velocity, transverse):
    """Return (frequencies, amplitudes) of longitudinal "phantom" partials.

    Striking a string stretches it, so transverse motion pumps a longitudinal
    wave.  Longitudinal modes sit at m * c_L / 2L, and because c_L is tens of
    times the transverse wave speed they land far above the fundamental --
    around 1.3 kHz on the bottom A, where nothing in the transverse series puts
    them.  They decay fast and are the metallic clang of a hard bass note.

    Placing them at sums of transverse partials instead, as is sometimes done,
    is not equivalent: inharmonicity puts f1+f1 within 0.01 Hz of partial 2, so
    they would merely reinforce a partial that is already there.
    """
    midi = float(midi)
    if midi > LONGITUDINAL_FROM_MIDI:
        return np.zeros(0), np.zeros(0)
    v = float(np.clip(velocity / 127.0, 0.0, 1.0))
    register = float(np.clip((LONGITUDINAL_FROM_MIDI - midi) / 24.0, 0.0, 1.0))
    # Longitudinal motion is driven by the square of transverse displacement,
    # but it only rises above the transverse sound at genuine forte -- it is
    # the clang of a hard bass blow, not a feature of ordinary playing.  A
    # gentler velocity law spreads a pure inharmonic tone near 2-4 kHz across
    # every note up to E4, which is heard as a bell rather than a piano.
    level = LONGITUDINAL_LEVEL * register * v ** LONGITUDINAL_VELOCITY_POWER
    if level <= 0.0:
        return np.zeros(0), np.zeros(0)

    length = float(speaking_length(midi))
    first = LONGITUDINAL_SPEED / (2.0 * length)
    freqs, amps = [], []
    for m in (1, 2, 3):
        freqs.append(first * m)
        amps.append(level / m ** 1.5)
    return np.array(freqs), np.array(amps)


# Longitudinal modes are heavily damped and die away well before the note does.
LONGITUDINAL_T60_SCALE = 0.18


# --------------------------------------------------------------------------
# Stereo placement
# --------------------------------------------------------------------------
SPEED_OF_SOUND = 343.0
PLATE_SPAN_X = 1.40          # m, matches the FEA plate
PLATE_SPAN_Y = 1.52
# A spaced pair set outside the rim, as an engineer would place them over a
# nine-foot instrument.  Mics sampling the board at the FEA pickup points would
# sit only ~0.7 m apart, and every string would then reach both within 0.2 ms --
# far too little to decorrelate anything.
MIC_HEIGHT = 0.55            # m above the board
MIC_LEFT = (-0.10, 0.15)     # plate fractions; outside [0,1] means past the rim
MIC_RIGHT = (1.10, 0.90)


def string_position(midi):
    """Where a note's speaking length sits on the board, in metres.

    Bass strings are long and cross the board diagonally; the treble is short
    and sits at the far end.  This is the geometry a pair of microphones over
    the instrument actually hears.
    """
    x = np.clip((np.asarray(midi, dtype=float) - LOWEST_MIDI)
                / (HIGHEST_MIDI - LOWEST_MIDI), 0.0, 1.0)
    return (0.12 + 0.76 * x) * PLATE_SPAN_X, (0.80 - 0.58 * x) * PLATE_SPAN_Y


def stereo_arrival(midi):
    """Per-note (delay_left_s, delay_right_s, gain_left, gain_right).

    A spaced microphone pair placed over a grand hears every string at a
    slightly different time and level.  That arrival difference -- not level
    panning -- is what actually decorrelates the two channels: panning a mono
    source keeps the channels almost perfectly correlated no matter how wide
    the pan law is.
    """
    sx, sy = string_position(midi)
    out = []
    for mx, my in (MIC_LEFT, MIC_RIGHT):
        dx = sx - mx * PLATE_SPAN_X
        dy = sy - my * PLATE_SPAN_Y
        out.append(float(np.sqrt(dx * dx + dy * dy + MIC_HEIGHT ** 2)))
    dl, dr = out
    near = min(dl, dr)
    return dl / SPEED_OF_SOUND, dr / SPEED_OF_SOUND, near / dl, near / dr


# --------------------------------------------------------------------------
# Radiation
# --------------------------------------------------------------------------
RADIATION_CORNER_HZ = 110.0
RADIATION_FLOOR = 0.32
RADIATION_ROLLOFF_HZ = 5200.0


def radiation_gain(freq):
    """Soundboard radiation efficiency versus frequency.

    A soundboard is a poor radiator well below its critical frequency and an
    efficient one above it, so the raw string spectrum is tilted upward before
    it reaches the room.  Above a few kHz the finite plate rolls off again.

    The floor matters as much as the corner.  A nine-foot board is large and
    the bass strings carry enormous energy, so real low notes are powerful
    despite the poor radiation efficiency.  Taking the efficiency curve down to
    near zero at the bottom -- which the bare physics would suggest -- leaves
    the instrument sounding thin and bass-light.
    """
    freq = np.maximum(np.asarray(freq, dtype=float), 1.0)
    ratio = freq / RADIATION_CORNER_HZ
    rise = ratio ** 2 / (1.0 + ratio ** 2)
    fall = 1.0 / (1.0 + (freq / RADIATION_ROLLOFF_HZ) ** 2)
    return RADIATION_FLOOR + (1.0 - RADIATION_FLOOR) * rise * fall
