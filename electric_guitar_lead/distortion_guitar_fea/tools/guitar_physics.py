"""Equation-derived strings, pickups, and expressive pitch trajectories."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

SAMPLE_RATE = 48_000
SCALE_LENGTH_M = 0.648
STEEL_E_PA = 2.0e11
OPEN_MIDI = np.array([40, 45, 50, 55, 59, 64], dtype=int)

# Interpreted from the supplied physical-guitar UI reference: pick a little
# farther from the bridge, hard plectrum, natural string behaviour, and the
# middle pickup of a three-single-coil solid body.
PICK_POSITION_FRACTION = .142
PICK_CONTACT_FRACTION = .0047
PICKUP_POSITIONS = {"bridge": .102, "middle": .172, "neck": .252}
PICKUP_APERTURE = .027


@dataclass(frozen=True)
class StringSpec:
    name: str
    open_midi: int
    core_diameter_m: float
    linear_density_kg_m: float


STRINGS = (
    StringSpec("E2", 40, 0.00043, 0.00478),
    StringSpec("A2", 45, 0.00038, 0.00308),
    StringSpec("D3", 50, 0.00033, 0.00172),
    StringSpec("G3", 55, 0.00028, 0.00084),
    StringSpec("B3", 59, 0.000330, 0.00060),
    StringSpec("E4", 64, 0.000254, 0.00036),
)


def configure_tuning(open_midi):
    """Apply the score's low-to-high six-string tuning to all string equations."""
    global OPEN_MIDI, STRINGS
    values=np.asarray(open_midi,dtype=int)
    if values.shape != (6,) or np.any(np.diff(values)<=0):
        raise ValueError(f"expected six ascending open-string MIDI notes, got {values.tolist()}")
    OPEN_MIDI=values.copy()
    names=("C2","G2","C3","F3","A3","D4") if values.tolist()==[36,43,48,53,57,62] else tuple(str(v) for v in values)
    STRINGS=tuple(StringSpec(names[i],int(values[i]),spec.core_diameter_m,spec.linear_density_kg_m)
                  for i,spec in enumerate(STRINGS))


def fundamental(midi):
    return 440.0 * 2.0 ** ((np.asarray(midi, dtype=float) - 69.0) / 12.0)


def choose_string(midi: float) -> int:
    candidates = np.flatnonzero(OPEN_MIDI <= int(math.floor(midi)))
    if not len(candidates) or midi > 88:
        raise ValueError(f"MIDI {midi} is outside E2..E6")
    return int(candidates[-1])


def fret_for_midi(midi: float, string_index: int | None = None) -> float:
    idx = choose_string(midi) if string_index is None else int(string_index)
    fret = float(midi) - STRINGS[idx].open_midi
    if not 0 <= fret <= 24.25:
        raise ValueError(f"MIDI {midi} is not playable on {STRINGS[idx].name}")
    return fret


def speaking_length(midi: float, string_index: int | None = None) -> float:
    return SCALE_LENGTH_M / 2.0 ** (fret_for_midi(midi, string_index) / 12.0)


def open_tension(string_index: int) -> float:
    spec = STRINGS[int(string_index)]
    return spec.linear_density_kg_m * (
        2.0 * SCALE_LENGTH_M * float(fundamental(spec.open_midi))
    ) ** 2


def inharmonicity(midi: float, string_index: int | None = None) -> float:
    idx = choose_string(midi) if string_index is None else int(string_index)
    spec = STRINGS[idx]
    inertia = math.pi * spec.core_diameter_m**4 / 64.0
    length = speaking_length(midi, idx)
    return math.pi**2 * STEEL_E_PA * inertia / (open_tension(idx) * length**2)


def partial_count(midi: float, sample_rate: int = SAMPLE_RATE, string_index: int | None = None) -> int:
    f0 = float(fundamental(midi))
    b = inharmonicity(midi,string_index)
    for count in range(1, 80):
        if count * f0 * math.sqrt(1.0 + b * count * count) > min(14_000, .46 * sample_rate):
            return max(1, count - 1)
    return 79


def partial_amplitudes(midi: float, velocity: int, count: int) -> np.ndarray:
    n = np.arange(1, count + 1, dtype=float)
    vel = np.clip(velocity / 127.0, .05, 1.0)
    p = PICK_POSITION_FRACTION + .012 * (1.0 - vel)
    # Keep modal signs.  A real triangular displacement has coherent phase;
    # rectifying this comb makes the attack synthetic and destroys cancellations.
    comb = np.sin(np.pi * n * p)
    contact = np.sinc(n * PICK_CONTACT_FRACTION)
    brightness = np.exp(-n / (20.0 + 34.0 * vel))
    # Magnetic voltage adds a frequency factor later, so displacement must
    # fall steeply enough to retain the woody 250--900 Hz body seen in the
    # supplied physical-model reference.
    a = comb * contact * brightness / n**1.62
    return a / max(float(np.max(np.abs(a))), 1e-12)


def pickup_response(partials: np.ndarray, selected: str = "middle") -> np.ndarray:
    n = np.asarray(partials, dtype=float)
    if selected not in PICKUP_POSITIONS:
        raise ValueError(f"unknown pickup {selected!r}")
    # A single coil samples a narrower magnetic window than a humbucker.
    return np.sin(np.pi*n*PICKUP_POSITIONS[selected])*np.sinc(n*PICKUP_APERTURE)


def structure_mobility(freq_hz, modes: list[dict]):
    freq = np.atleast_1d(np.asarray(freq_hz, dtype=float))
    mobility = np.zeros_like(freq)
    for mode in modes:
        fm, q = float(mode["hz"]), float(mode["q"])
        coupling = abs(float(mode["string_coupling"]))
        detune = freq / fm - fm / np.maximum(freq, 1e-9)
        mobility += coupling**2 / (1.0 + (q * detune) ** 2)
    return mobility


def coupled_t60(freq_hz, midi: float, modes: list[dict], string_index: int | None = None):
    freq = np.maximum(np.asarray(freq_hz, dtype=float), 20.0)
    base = 14.2 * (110.0 / freq) ** .44
    fret_loss = 1.0 / (1.0 + .011 * fret_for_midi(midi,string_index))
    return np.clip(base * fret_loss / (1.0 + 6.2 * structure_mobility(freq, modes)), .22, 14.0)


def smoothstep(x):
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def pitch_curve(start: float, target: float, articulation: str, t: np.ndarray,
                duration: float, vibrato_cents: float = 0.0) -> np.ndarray:
    """Continuous fretting-hand trajectory in MIDI-note units."""
    x = np.clip(t / max(duration, 1e-6), 0.0, 1.0)
    if articulation in ("slide", "dead_slide_in"):
        move = smoothstep(np.clip((x - .08) / .76, 0, 1))
        pitch = start + (target - start) * move
    elif articulation == "legato_slide":
        # GP's legato slide connects to the following destination note. Hold
        # the source fret, then make the hand shift near the end instead of
        # spending most of a short note on exposed microtonal pitches.
        move = smoothstep(np.clip((x - .56) / .34, 0, 1))
        pitch = start + (target - start) * move
    elif articulation in ("slide_out_up", "slide_out_down"):
        move = smoothstep(np.clip((x - .68) / .28, 0, 1))
        pitch = start + (target - start) * move
    elif articulation in ("hammer_on", "pull_off", "legato", "tap"):
        # Fast fingertip contact/release, continuous over the first 8%.
        move = smoothstep(np.clip(x / .08, 0, 1))
        pitch = start + (target - start) * move
    elif articulation == "bend_up":
        move = smoothstep(np.clip((x - .08) / .54, 0, 1))
        pitch = start + (target - start) * move
    elif articulation == "bend_release":
        up = smoothstep(np.clip(x / .30, 0, 1))
        down = smoothstep(np.clip((x - .48) / .40, 0, 1))
        pitch = start + (target - start) * (up - down)
    elif articulation == "prebend_release":
        pitch = start + (target - start) * smoothstep(np.clip((x - .18) / .62, 0, 1))
    else:
        pitch = np.full_like(t, target, dtype=float)
    if vibrato_cents:
        fade = smoothstep(np.clip((x - .18) / .20, 0, 1))
        pitch += (vibrato_cents / 100.0) * fade * np.sin(2 * np.pi * 5.6 * t)
    return pitch


ARTICULATION_EXCITATION = {
    "pick": 1.00, "accent": 1.12, "hammer_on": .34, "pull_off": .29,
    "legato": .16, "slide": .48, "legato_slide": .22,
    "bend_up": .88, "bend_release": .90, "prebend_release": .78,
    "pinch_harmonic": .92, "tap": .31, "palm_mute": .70,
    "dead_note": .58, "dead_slide_in": .54,
    "slide_out_up": .44, "slide_out_down": .44,
}
