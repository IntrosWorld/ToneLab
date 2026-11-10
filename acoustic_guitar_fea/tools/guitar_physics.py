#!/usr/bin/env python3
"""Physics utilities for a steel-string acoustic guitar.

The model is intentionally audio-free: coefficients come from string
geometry, material properties, scale length, pluck position, and loss models.
It is the floating-point design source for a possible later hardware model.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

SAMPLE_RATE = 48_000
A4_MIDI = 69
A4_HZ = 440.0
SCALE_LENGTH_M = 0.648
STEEL_E_PA = 2.0e11

# Standard tuning, low E through high E.
OPEN_MIDI = np.array([40, 45, 50, 55, 59, 64], dtype=int)


@dataclass(frozen=True)
class StringSpec:
    name: str
    open_midi: int
    gauge_in: float
    core_diameter_m: float
    linear_density_kg_m: float


# Representative light phosphor-bronze set. Wound strings use the steel core
# diameter for bending stiffness and total linear density for tension.
STRINGS = (
    StringSpec("E2", 40, 0.053, 0.00048, 0.00600),
    StringSpec("A2", 45, 0.042, 0.00043, 0.00405),
    StringSpec("D3", 50, 0.032, 0.00038, 0.00245),
    StringSpec("G3", 55, 0.024, 0.00033, 0.00145),
    StringSpec("B3", 59, 0.016, 0.000406, 0.00091),
    StringSpec("E4", 64, 0.012, 0.000305, 0.00053),
)


def fundamental(midi: int | np.ndarray) -> float | np.ndarray:
    """Equal-tempered fundamental frequency in Hz."""
    return A4_HZ * 2.0 ** ((np.asarray(midi, dtype=float) - A4_MIDI) / 12.0)


def choose_string(midi: int) -> int:
    """Choose the highest open string that can play `midi`.

    This gives conventional guitar fingering and avoids implausibly high frets
    on a lower string. Notes outside E2..E6 are rejected.
    """
    midi = int(midi)
    candidates = np.flatnonzero(OPEN_MIDI <= midi)
    if len(candidates) == 0 or midi > 88:
        raise ValueError(f"MIDI {midi} lies outside the modeled E2..E6 range")
    return int(candidates[-1])


def fret_for_midi(midi: int, string_index: int | None = None) -> int:
    idx = choose_string(midi) if string_index is None else int(string_index)
    fret = int(midi) - STRINGS[idx].open_midi
    if not 0 <= fret <= 24:
        raise ValueError(f"MIDI {midi} is not playable on {STRINGS[idx].name}")
    return fret


def speaking_length(midi: int, string_index: int | None = None) -> float:
    """Fretted speaking length in metres."""
    fret = fret_for_midi(midi, string_index)
    return SCALE_LENGTH_M / 2.0 ** (fret / 12.0)


def open_tension(string_index: int) -> float:
    """String tension in newtons from wave speed and linear density."""
    spec = STRINGS[int(string_index)]
    return spec.linear_density_kg_m * (
        2.0 * SCALE_LENGTH_M * float(fundamental(spec.open_midi))
    ) ** 2


def inharmonicity(midi: int, string_index: int | None = None) -> float:
    """Stiff-string coefficient B = pi^2 E I / (T L^2)."""
    idx = choose_string(midi) if string_index is None else int(string_index)
    spec = STRINGS[idx]
    length = speaking_length(midi, idx)
    inertia = math.pi * spec.core_diameter_m**4 / 64.0
    return math.pi**2 * STEEL_E_PA * inertia / (
        open_tension(idx) * length**2
    )


def partial_frequencies(
    midi: int, count: int, string_index: int | None = None
) -> np.ndarray:
    """Stiff-string transverse partial frequencies."""
    n = np.arange(1, count + 1, dtype=float)
    f0 = float(fundamental(midi))
    b = inharmonicity(midi, string_index)
    # Tune the *sounding fundamental*, not the hypothetical flexible string.
    return n * f0 * np.sqrt((1.0 + b * n * n) / (1.0 + b))


def partial_count(midi: int, sample_rate: float = SAMPLE_RATE,
                  string_index: int | None = None) -> int:
    """Number of partials below a conservative radiation/Nyquist ceiling."""
    ceiling = min(0.46 * sample_rate, 14_000.0)
    for count in range(1, 96):
        if partial_frequencies(midi, count, string_index)[-1] >= ceiling:
            return max(1, count - 1)
    return 95


def pluck_position(technique: str = "finger") -> float:
    """Pluck location as a fraction of speaking length from the bridge."""
    positions = {
        "warm": 0.22,
        "finger": 0.16,
        # A strumming hand normally crosses near the sound-hole edge, farther
        # from the bridge than a bright single-note pick attack.
        "strum": 0.145,
        "pick": 0.115,
        "bright": 0.075,
    }
    try:
        return positions[technique]
    except KeyError as exc:
        raise ValueError(f"unknown technique {technique!r}") from exc


def partial_amplitudes(
    midi: int,
    velocity: int,
    count: int,
    technique: str = "finger",
    position: float | None = None,
    string_index: int | None = None,
) -> np.ndarray:
    """Modal amplitudes of a released triangular string displacement.

    The sin(n*pi*p) comb is the key pluck-position signature. A finite fingertip
    or pick width adds a sinc roll-off, and bridge mobility slightly favours
    upper modes in the force delivered to the top plate.
    """
    n = np.arange(1, count + 1, dtype=float)
    p = pluck_position(technique) if position is None else float(position)
    if not 0.02 <= p <= 0.48:
        raise ValueError("pluck position must be between 0.02 and 0.48")
    velocity_norm = np.clip(float(velocity) / 127.0, 0.0, 1.0)
    # A released triangular displacement has signed sin(n*pi*p)/n**2
    # coefficients; bridge slope contributes one factor n. Preserving the
    # signs and cosine phase is essential to the traveling pluck waveform.
    comb = np.sin(np.pi * n * p)
    widths = {"warm": 0.014, "finger": 0.014, "strum": 0.010,
              "pick": 0.006, "bright": 0.006}
    width = widths[technique]
    finite_contact = np.sinc(n * width)
    freqs = partial_frequencies(midi, count, string_index)
    cutoff = (2200.0 + 7500.0 * velocity_norm**0.8)
    cutoff *= {"warm": .68, "finger": .86, "strum": 1.,
               "pick": 1.10, "bright": 1.25}[technique]
    bridge_force = .50 * comb * finite_contact / (np.pi * n * p * (1.0 - p))
    return bridge_force * np.exp(-(freqs / cutoff)**1.3) * velocity_norm**1.45


def t60(freq_hz: float | np.ndarray, midi: int = 52,
        string_index: int | None = None) -> np.ndarray:
    """Amplitude T60 of a guitar string partial.

    Bridge radiation and internal friction remove high-frequency energy much
    faster. Fretted notes are also slightly more lossy than open strings.
    """
    freq = np.maximum(np.asarray(freq_hz, dtype=float), 20.0)
    fret = fret_for_midi(int(midi), string_index)
    base = 10.5 * (110.0 / freq) ** 0.58
    fret_loss = 1.0 / (1.0 + 0.012 * fret)
    return np.clip(base * fret_loss, 0.24, 13.0)


def decay_rate(freq_hz: float | np.ndarray, midi: int = 52,
               string_index: int | None = None) -> np.ndarray:
    return np.log(1000.0) / t60(freq_hz, midi, string_index)


def pick_noise_duration_s(technique: str) -> float:
    if technique == "strum":
        return 0.0045
    return 0.0035 if technique in ("pick", "bright") else 0.0065


def bridge_force_scale(midi: int, string_index: int | None = None) -> float:
    """Relative force delivered by the selected physical guitar string.

    Per-note peak normalization otherwise erases the large difference in
    characteristic impedance sqrt(T*mu) between wound bass and plain treble
    strings. A moderated ratio restores real downstroke weight without using
    an arbitrary bass EQ.
    """
    idx = choose_string(midi) if string_index is None else int(string_index)
    spec = STRINGS[idx]
    impedance = math.sqrt(open_tension(idx) * spec.linear_density_kg_m)
    reference = math.sqrt(
        open_tension(2) * STRINGS[2].linear_density_kg_m
    )
    return float(np.clip((impedance / reference) ** 0.40, 0.78, 1.30))


def release_t60(midi: int, string_index: int | None = None) -> float:
    """Left-hand/finger damping after note-off."""
    fret = fret_for_midi(midi, string_index)
    return float(np.clip(0.075 + 0.0025 * (24 - fret), 0.075, 0.14))


def bridge_position_for_string(string_index: int) -> tuple[float, float]:
    """Normalised bridge coordinate for each string across the saddle."""
    y = 0.5 + (int(string_index) - 2.5) * 0.012
    return 0.64, y


def helmholtz_frequency(
    cavity_volume_m3: float = 0.014,
    soundhole_radius_m: float = 0.049,
    top_thickness_m: float = 0.0028,
) -> float:
    """Air-cavity A0 resonance using an end-corrected soundhole neck.

    The textbook rigid-wall Helmholtz result is reduced by wall compliance:
    a guitar's top and back move with the enclosed air, increasing the
    effective acoustic compliance compared with a rigid bottle.
    """
    c = 343.0
    area = math.pi * soundhole_radius_m**2
    effective_length = top_thickness_m + 1.7 * soundhole_radius_m
    rigid_wall_hz = c / (2.0 * math.pi) * math.sqrt(
        area / (cavity_volume_m3 * effective_length)
    )
    wall_compliance_factor = 0.76
    return wall_compliance_factor * rigid_wall_hz


def body_q(freq_hz: float | np.ndarray) -> np.ndarray:
    """Broad, lossy wood/air resonances."""
    freq = np.maximum(np.asarray(freq_hz, dtype=float), 1.0)
    return np.clip(34.0 * (180.0 / freq) ** 0.28, 8.0, 42.0)
