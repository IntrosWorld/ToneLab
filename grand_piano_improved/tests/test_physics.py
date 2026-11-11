"""Invariants of the piano physics model.

These guard the properties that make the engine sound like a piano.  Several
of them encode bugs that were actually hit while building it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import piano_physics as phys  # noqa: E402


# ---------------------------------------------------------------------------
# Pitch and inharmonicity
# ---------------------------------------------------------------------------
def test_a4_is_440():
    assert phys.fundamental(69) == pytest.approx(440.0)


def test_octave_doubles():
    assert phys.fundamental(72) == pytest.approx(2 * phys.fundamental(60))


def test_inharmonicity_is_lowest_in_the_middle():
    """B is smallest in the tenor and rises at both ends of the compass."""
    middle = phys.inharmonicity(48)
    assert phys.inharmonicity(21) > middle
    assert phys.inharmonicity(108) > middle


def test_treble_is_far_stiffer_than_the_tenor():
    assert phys.inharmonicity(108) > 20 * phys.inharmonicity(48)


def test_partials_are_stretched_sharp():
    """Every partial sits above its harmonic position, increasingly so."""
    for pitch in (28, 60, 88):
        f = phys.partial_frequencies(pitch, 12)
        ratios = f / (np.arange(1, 13) * phys.fundamental(pitch))
        assert np.all(ratios >= 1.0)
        assert np.all(np.diff(ratios) > 0)


# ---------------------------------------------------------------------------
# Decay
# ---------------------------------------------------------------------------
def test_higher_partials_decay_faster():
    freqs = np.array([100.0, 400.0, 1600.0, 6400.0])
    t = phys.t60(freqs)
    assert np.all(np.diff(t) < 0)


def test_decay_times_are_physically_plausible():
    assert 15.0 < phys.t60(100.0) < 45.0     # low partials ring for tens of seconds
    assert 0.4 < phys.t60(4000.0) < 4.0      # treble partials die quickly


def test_decay_rate_matches_t60():
    """A partial must fall exactly 60 dB in its own T60."""
    for f in (80.0, 500.0, 3000.0):
        decayed = np.exp(-phys.decay_rate(f) * phys.t60(f))
        assert 20 * np.log10(decayed) == pytest.approx(-60.0, abs=1e-6)


def test_prompt_polarisation_decays_faster_than_aftersound():
    """Double decay only exists if the two polarisations differ."""
    assert phys.PROMPT_T60_SCALE < phys.AFTER_T60_SCALE
    assert phys.POLARISATION_DETUNE_RATIO > 0.0


def test_polarisation_beat_rate_scales_with_frequency():
    """A fixed Hz offset would beat every partial in lockstep.

    Partials beating together modulate the whole note's level; partials beating
    at their own rates give shimmer instead.
    """
    low = 262.0 * phys.POLARISATION_DETUNE_RATIO
    high = 2620.0 * phys.POLARISATION_DETUNE_RATIO
    assert high > 5.0 * low
    assert 0.05 < low < 1.0        # audible beating, not a detuned note


# ---------------------------------------------------------------------------
# Hammer
# ---------------------------------------------------------------------------
def test_harder_playing_is_brighter():
    soft = phys.contact_corner_hz(60, 20)
    hard = phys.contact_corner_hz(60, 120)
    assert hard > 2.0 * soft


def test_harder_playing_is_louder():
    soft = phys.partial_amplitudes(60, 20, 8).sum()
    hard = phys.partial_amplitudes(60, 120, 8).sum()
    assert hard > 4.0 * soft


def test_strike_point_notches_the_eighth_partial():
    """The hammer sits near 1/8 of the string, so partial 8 is suppressed."""
    amps = phys.partial_amplitudes(60, 100, 12)
    assert amps[7] < 0.5 * amps[6]
    assert amps[7] < 0.5 * amps[8]


def test_strike_notch_is_not_bottomless():
    """A broad felt hammer cannot null a partial the way a plectrum would.

    Fully nulled partials sound hollow and harpsichord-like.
    """
    amps = phys.partial_amplitudes(60, 100, 12)
    assert amps[7] > 0.1 * amps[6]


def test_hammer_contact_ramps_the_attack():
    """Instantaneous partial onsets are a pluck, not a struck string."""
    assert phys.contact_time_s(60, 100) > 0.0005
    assert phys.contact_time_s(21, 100) > phys.contact_time_s(96, 100)
    assert phys.contact_time_s(60, 30) > phys.contact_time_s(60, 127)


def test_prompt_decay_is_not_overwhelmingly_fast():
    """A note that dumps most of its energy at once reads as a pluck."""
    assert 0.25 < phys.PROMPT_T60_SCALE < 0.7


def _combined_level_db(freq, seconds):
    """Level of a partial's two polarisations together, in dB from onset."""
    rate = phys.decay_rate(freq)
    level = (phys.PROMPT_LEVEL * np.exp(-rate / phys.PROMPT_T60_SCALE * seconds)
             + phys.AFTER_LEVEL * np.exp(-rate * phys.AFTER_T60_SCALE * seconds))
    return 20 * np.log10(level / (phys.PROMPT_LEVEL + phys.AFTER_LEVEL))


def test_a_note_still_sings_after_one_second():
    """The pluck test.

    A real grand loses roughly 8-12 dB in the first second of a mid-register
    note.  An engine that loses 20 dB sounds plucked no matter how its spectrum
    is tuned, because the ear reads the energy decay, not the spectrum.
    """
    assert -14.0 < _combined_level_db(261.6, 1.0) < -4.0


def test_upper_partials_decay_faster_but_not_instantly():
    """Upper partials genuinely do die fast on a real piano.

    A 2 kHz partial has a T60 of two or three seconds, so losing about 22 dB in
    the first second is correct, not plucky.  What makes an engine sound
    plucked is the *total* energy collapsing, and that is carried by the strong
    low-mid partials -- which is what the test above pins down.  This one only
    guards against the upper partials vanishing altogether.
    """
    assert _combined_level_db(2000.0, 1.0) > -28.0
    assert _combined_level_db(2000.0, 1.0) < _combined_level_db(262.0, 1.0)


def test_amplitudes_fall_with_partial_number():
    amps = phys.partial_amplitudes(60, 100, 24)
    assert amps[0] > amps[3] > amps[15]


def test_partial_count_stays_below_nyquist():
    for pitch in range(21, 109, 3):
        count = phys.partial_count(pitch, 48000.0)
        assert count >= 1
        assert phys.partial_frequencies(pitch, count)[-1] < 24000.0


# ---------------------------------------------------------------------------
# Nonlinearity
# ---------------------------------------------------------------------------
def test_glide_is_a_bass_forte_effect():
    assert phys.attack_glide_cents(28, 127) > phys.attack_glide_cents(28, 20)
    assert phys.attack_glide_cents(28, 127) > phys.attack_glide_cents(84, 127)
    assert phys.attack_glide_cents(84, 20) == pytest.approx(0.0, abs=0.5)


def test_phantom_partials_only_in_the_bass():
    transverse = phys.partial_frequencies(30, 8)
    freqs, amps = phys.longitudinal_partials(30, 120, transverse)
    assert len(freqs) > 0 and np.all(amps > 0)
    treble, _ = phys.longitudinal_partials(84, 120, phys.partial_frequencies(84, 8))
    assert len(treble) == 0


def test_phantom_partials_land_far_above_the_transverse_series():
    """They must not merely reinforce a partial that already exists.

    Placing them at sums of transverse partials does exactly that: with real
    inharmonicity, f1+f1 sits within 0.01 Hz of partial 2.
    """
    transverse = phys.partial_frequencies(21, 16)
    freqs, _ = phys.longitudinal_partials(21, 120, transverse)
    assert freqs[0] > 8 * transverse[0]
    assert not np.any(np.isclose(freqs[0], transverse, rtol=2e-3))


def test_longitudinal_speed_far_exceeds_transverse():
    for pitch in (21, 40, 60):
        length = phys.speaking_length(pitch)
        transverse = 2.0 * length * phys.fundamental(pitch)
        assert phys.LONGITUDINAL_SPEED > 8.0 * transverse


def test_strings_get_shorter_with_pitch():
    lengths = phys.speaking_length(np.arange(21, 109))
    assert np.all(np.diff(lengths) < 0)
    assert 1.5 < lengths[0] < 2.5      # bottom A of a concert grand
    assert 0.03 < lengths[-1] < 0.10   # top C


# ---------------------------------------------------------------------------
# Dampers and unisons
# ---------------------------------------------------------------------------
def test_top_octave_has_no_damper():
    assert phys.damper_t60(100) is None
    assert phys.damper_t60(60) is not None


def test_dampers_grip_the_treble_faster():
    assert phys.damper_t60(30) > phys.damper_t60(80)


def test_unison_count_rises_with_pitch():
    assert phys.strings_per_note(24) == 1
    assert phys.strings_per_note(36) == 2
    assert phys.strings_per_note(60) == 3


def test_unison_detune_is_small_but_present():
    spread = [phys.unison_detune_cents(60, i, 3) for i in range(3)]
    assert max(spread) - min(spread) > 0.2   # audible beating
    assert max(abs(c) for c in spread) < 4.0  # still in tune


# ---------------------------------------------------------------------------
# Stereo geometry
# ---------------------------------------------------------------------------
def test_bass_leans_left_and_treble_leans_right():
    _, _, bass_l, bass_r = phys.stereo_arrival(28)
    _, _, treble_l, treble_r = phys.stereo_arrival(100)
    assert bass_l > bass_r
    assert treble_r > treble_l


def test_arrival_difference_is_large_enough_to_decorrelate():
    """Mics too close together leave the channels effectively mono."""
    tl, tr, _, _ = phys.stereo_arrival(28)
    assert abs(tl - tr) > 0.3e-3


# ---------------------------------------------------------------------------
# Soundboard solve
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def board():
    path = ROOT / "rtl" / "soundboard_modes.json"
    if not path.exists():
        pytest.skip("run tools/soundboard_fea.py first")
    return json.loads(path.read_text(encoding="utf-8"))["modes"]


def test_lowest_body_mode_is_where_a_real_soundboard_sits(board):
    """A non-symmetric stiffness assembly put this cluster at 1252 Hz."""
    lowest = min(m["hz"] for m in board)
    assert 30.0 < lowest < 120.0


def test_body_modes_are_broad(board):
    assert all(5.0 < m["q"] < 120.0 for m in board)


def test_pickups_see_different_mode_mixtures(board):
    """Stereo width comes from the two points genuinely differing."""
    left = np.array([m["left"] for m in board])
    right = np.array([m["right"] for m in board])
    assert np.corrcoef(left, right)[0, 1] < 0.99
    assert np.any(np.sign(left) != np.sign(right))
