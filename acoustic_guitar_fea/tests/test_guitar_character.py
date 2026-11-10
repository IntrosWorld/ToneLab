"""Audible-band safeguards for the plucked-string acoustic render.

These are broad spectral regressions, not a numerical definition of realism.
In particular, adding a bright contact click must not hide a dull sustain.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from render_model import AcousticGuitar  # noqa: E402
from string_performance import render_string_performance  # noqa: E402

RATE = 48_000


def note(midi, string_index, technique="pick", seed=1946):
    event = dict(midi=midi, string_index=string_index, velocity=85,
                 start=0.0, duration=0.8)
    return AcousticGuitar(RATE, technique, seed).render(
        [event], tail=0.1, normalize=False, room=False).mean(axis=1)


def spectrum(audio, start, end):
    section = audio[round(start * RATE):round(end * RATE)]
    power = np.abs(np.fft.rfft(section * np.hanning(len(section))))**2
    frequencies = np.fft.rfftfreq(len(section), 1.0 / RATE)
    return frequencies, power / power.sum()


@pytest.mark.parametrize("seed", [42, 1946, 7251])
@pytest.mark.parametrize("midi,string_index", [(76, 4), (80, 5)])
def test_high_fret_lead_retains_upper_harmonics_after_pick_contact(
        midi, string_index, seed):
    # These are frequent high-fret notes in the requested GP lead. Inspect
    # 50–150 ms, well after the short contact transient. Broad fractional
    # energy bounds catch collapse to an almost isolated fundamental without
    # prescribing exact harmonic amplitudes or depending on output loudness.
    # Test excitation before the spatial microphone transfer. The resolved
    # body has a real antiresonance at E5's second partial, so requiring every
    # recorded note to have the same upper-energy floor suppresses geometry.
    audio = AcousticGuitar(RATE, "pick", seed).note_bridge_force(
        midi, 85, .8, tail=.1, string_index=string_index)
    frequencies, power = spectrum(audio, 0.05, 0.15)
    fundamental = 440.0 * 2.0**((midi - 69.0) / 12.0)
    assert power[frequencies > 1.5 * fundamental].sum() > 0.25
    assert power[frequencies > 2.5 * fundamental].sum() > 0.01
    # A broadband hiss could satisfy upper-band energy alone. The sustained
    # note must still concentrate its energy around string harmonics.
    harmonic_distance = np.abs(
        frequencies / fundamental - np.rint(frequencies / fundamental))
    assert power[harmonic_distance < 0.10].sum() > 0.85


@pytest.mark.parametrize("midi,string_index", [(52, 2), (64, 5), (76, 4), (80, 5)])
def test_intrinsic_string_softens_and_acoustic_note_keeps_ringing(midi, string_index):
    audio = note(midi, string_index)
    raw = render_string_performance([dict(midi=midi, string_index=string_index,
        velocity=85, start=0., duration=.8)], RATE, technique="pick", tail=.1)
    early_f, early_p = spectrum(raw, 0.005, 0.045)
    late_f, late_p = spectrum(raw, 0.40, 0.60)
    # Intrinsic material loss increases with partial frequency. Added bridge
    # mobility need not: near a body resonance the fundamental can decay more
    # quickly than its overtones. Thus the acoustic centroid need not fall.
    assert early_f @ early_p > 1.05 * (late_f @ late_p)
    early_rms = np.sqrt(np.mean(audio[round(.005 * RATE):round(.045 * RATE)]**2))
    late_rms = np.sqrt(np.mean(audio[round(.40 * RATE):round(.60 * RATE)]**2))
    # Continued pitched decay matters; a short percussive click fading to
    # silence would also exhibit a falling spectral centroid.
    assert 0.03 < late_rms / early_rms < 0.8


@pytest.mark.parametrize("midi,string_index", [(52, 2), (64, 5)])
def test_pick_and_finger_have_distinct_sustained_timbres(midi, string_index):
    picked = note(midi, string_index, "pick")
    fingered = note(midi, string_index, "finger")
    frequencies, pick_power = spectrum(picked, .05, .15)
    _, finger_power = spectrum(fingered, .05, .15)
    assert frequencies @ pick_power > 1.05 * (frequencies @ finger_power)
    assert np.linalg.norm(pick_power - finger_power) > 0.025
