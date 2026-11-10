"""Four-string electric bass: tension-string and neck/body modal FEA."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from scipy import linalg
from scipy.signal import butter, lfilter, sosfilt

SR = 48_000
SCALE = .864
OPEN_MIDI = np.array([28, 33, 38, 43])
MU = np.array([.0146, .0096, .0060, .0031])


@dataclass(frozen=True)
class BassNote:
    start: float
    duration: float
    midi: int
    velocity: int = 108
    string_index: int | None = None


def configure_tuning(open_midi):
    """Configure the score's four open strings, low to high."""
    global OPEN_MIDI
    values = np.asarray(open_midi, dtype=int)
    if values.shape != (4,) or np.any(np.diff(values) <= 0):
        raise ValueError(f"expected four ascending bass open pitches, got {values.tolist()}")
    OPEN_MIDI = values


def hz(midi):
    return 440.0 * 2 ** ((np.asarray(midi) - 69) / 12)


def string_index(midi):
    return int(np.flatnonzero(OPEN_MIDI <= midi)[-1])


def fe_string_modes(index, elements=36, count=14):
    n = elements + 1
    le = SCALE / elements
    tension = MU[index] * (2 * SCALE * hz(OPEN_MIDI[index])) ** 2
    stiffness = np.zeros((n, n))
    mass = np.zeros((n, n))
    ke = tension / le * np.array([[1., -1.], [-1., 1.]])
    me = MU[index] * le / 6 * np.array([[2., 1.], [1., 2.]])
    for element in range(elements):
        ix = np.ix_([element, element + 1], [element, element + 1])
        stiffness[ix] += ke
        mass[ix] += me
    values, vectors = linalg.eigh(
        stiffness[1:-1, 1:-1], mass[1:-1, 1:-1], subset_by_index=[0, count - 1]
    )
    frequencies = np.sqrt(np.maximum(values, 0)) / (2 * np.pi)
    coupling = vectors[int(.22 * (n - 2)), :] * vectors[int(.78 * (n - 2)), :]
    coupling /= max(np.max(np.abs(coupling)), 1e-12)
    return frequencies, coupling


def body_modes():
    elements, length = 20, 1.08
    le, dof = length / elements, 2 * (elements + 1)
    e_mod, inertia, rho_a = 10.8e9, 7.8e-6, 4.1
    stiffness = np.zeros((dof, dof))
    mass = np.zeros((dof, dof))
    ke = e_mod * inertia / le**3 * np.array([
        [12, 6*le, -12, 6*le], [6*le, 4*le**2, -6*le, 2*le**2],
        [-12, -6*le, 12, -6*le], [6*le, 2*le**2, -6*le, 4*le**2]
    ])
    me = rho_a * le / 420 * np.array([
        [156, 22*le, 54, -13*le], [22*le, 4*le**2, 13*le, -3*le**2],
        [54, 13*le, 156, -22*le], [-13*le, -3*le**2, -22*le, 4*le**2]
    ])
    for element in range(elements):
        ids = [2*element, 2*element+1, 2*element+2, 2*element+3]
        ix = np.ix_(ids, ids)
        stiffness[ix] += ke
        mass[ix] += me
    values, vectors = linalg.eigh(stiffness[2:, 2:], mass[2:, 2:], subset_by_index=[0, 9])
    frequencies = np.sqrt(np.maximum(values, 0)) / (2 * np.pi)
    coupling = vectors[-2, :]
    coupling /= max(np.max(np.abs(coupling)), 1e-12)
    return frequencies, coupling


def generate_model(path: Path):
    strings = []
    for index in range(4):
        frequencies, coupling = fe_string_modes(index)
        strings.append({
            "name": f"MIDI {int(OPEN_MIDI[index])}", "open_midi": int(OPEN_MIDI[index]),
            "frequencies_hz": frequencies.round(5).tolist(),
            "pickup_coupling": coupling.round(7).tolist(),
        })
    frequencies, coupling = body_modes()
    model = {
        "method": "two-node tension-string FEA plus Euler-Bernoulli neck/body FEA",
        "scale_length_m": SCALE, "strings": strings,
        "body_modes": [
            {"hz": round(float(f), 4), "coupling": round(float(c), 6), "q": 36 - 1.8*i}
            for i, (f, c) in enumerate(zip(frequencies, coupling))
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=2), encoding="utf-8")
    return model


def _resonator(signal, frequency, q):
    radius = math.exp(-math.pi * frequency / (q * SR))
    coefficient = 2 * radius * math.cos(2 * math.pi * frequency / SR)
    return lfilter([1-radius], [1, -coefficient, radius*radius], signal)


def synth_note(note: BassNote, model, rng):
    index = string_index(note.midi) if note.string_index is None else note.string_index
    fret = note.midi - OPEN_MIDI[index]
    n = int((note.duration + .34) * SR)
    time = np.arange(n) / SR
    fundamental = float(hz(note.midi))
    ratios = np.array(model["strings"][index]["frequencies_hz"]) / float(hz(OPEN_MIDI[index]))
    coupling = np.array(model["strings"][index]["pickup_coupling"])
    signal = np.zeros(n)
    velocity = note.velocity / 127
    stiffness = .000045 + .0000035*fret
    for mode, (ratio, weight) in enumerate(zip(ratios, coupling), 1):
        # Wound-string bending stiffness adds the slight upper-partial stretch
        # missing from an ideal tension-only string.
        frequency = fundamental * ratio * math.sqrt(1 + stiffness*mode*mode)
        decay = 3.05 * (110 / max(frequency, 40))**.32 / (1 + .018*fret)
        # Finger excitation at about 24% of speaking length creates the natural
        # modal comb; pickup velocity response gently restores definition.
        excitation = math.sin(math.pi*mode*.24)
        amplitude = weight*excitation / mode**.62 * np.exp(-mode/18) * (1+.018*mode)
        signal += amplitude * np.exp(-math.log(1000)*time/decay) * np.sin(
            2*np.pi*frequency*time + rng.uniform(-.08, .08)
        )
    signal *= 1 - np.exp(-time/.0045)
    contact = int(.012 * SR)
    noise = rng.standard_normal(contact) * np.exp(-np.arange(contact)/(contact*.15))
    contact_noise = sosfilt(butter(2, [450, 4200], "bandpass", fs=SR, output="sos"), noise)
    signal[:contact] += .020 * velocity * contact_noise
    release = int(note.duration * SR)
    if release < n:
        signal[release:] *= np.exp(-math.log(1000)*np.arange(n-release)/(.065*SR))
    return signal * velocity * .19


def render_bass(model, score, seconds, seed=2401):
    rng = np.random.default_rng(seed)
    mono = np.zeros(int(seconds * SR))
    for note in score:
        rendered = synth_note(note, model, rng)
        start = int(note.start * SR)
        span = min(len(rendered), len(mono) - start)
        if span > 0:
            mono[start:start+span] += rendered[:span]
    dry = mono.copy()
    for mode in model["body_modes"][:7]:
        mono += .012 * mode["coupling"] * _resonator(dry, mode["hz"], mode["q"])
    mono = sosfilt(butter(2, 32, "highpass", fs=SR, output="sos"), mono)
    # Parallel clean DI and a mildly driven cabinet path preserve the low
    # fundamental while giving the line audible note definition under guitars.
    low = sosfilt(butter(2, 175, "lowpass", fs=SR, output="sos"), mono)
    cabinet = sosfilt(butter(3, 3900, "lowpass", fs=SR, output="sos"), mono-low)
    cabinet = np.tanh(2.7*cabinet)/2.7
    mono = .92*low + .78*cabinet + .16*mono
    # Feed-forward compressor, approximately 8 ms attack / 95 ms release.
    detector = np.abs(mono)
    attack = math.exp(-1/(.008*SR))
    release = math.exp(-1/(.095*SR))
    envelope = np.empty_like(detector)
    state = 0.
    for i, value in enumerate(detector):
        coefficient = attack if value > state else release
        state = coefficient*state + (1-coefficient)*value
        envelope[i] = state
    threshold, ratio = .105, 3.2
    target = np.where(envelope > threshold,
                      (threshold+(envelope-threshold)/ratio)/np.maximum(envelope, 1e-9), 1.)
    mono *= target
    mono = np.tanh(1.45*mono)/1.45
    mono *= .76 / max(np.max(np.abs(mono)), 1e-9)
    right = np.zeros_like(mono)
    right[19:] = mono[:-19]
    return np.column_stack([mono, .97*right])
