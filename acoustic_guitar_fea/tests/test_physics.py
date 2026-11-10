from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import guitar_physics as phys  # noqa: E402


def test_standard_tuning():
    assert list(phys.OPEN_MIDI) == [40, 45, 50, 55, 59, 64]
    assert phys.fundamental(69) == pytest.approx(440.0)


def test_frets_shorten_by_equal_temperament():
    assert phys.speaking_length(52, 0) == pytest.approx(
        phys.SCALE_LENGTH_M / 2.0
    )


def test_string_tensions_are_plausible():
    tensions = np.array([phys.open_tension(i) for i in range(6)])
    assert np.all((tensions > 50.0) & (tensions < 110.0))


def test_stiff_partials_are_sharp():
    freqs = phys.partial_frequencies(52, 12)
    harmonic = np.arange(1, 13) * phys.fundamental(52)
    assert freqs[0] == pytest.approx(harmonic[0], rel=1e-14)
    assert np.all(freqs[1:] > harmonic[1:])
    assert np.all(np.diff(freqs / harmonic) > 0)


def test_pluck_position_creates_comb_signature():
    p = phys.pluck_position("finger")
    harmonic_near_null = round(1.0 / p)
    amps = np.abs(phys.partial_amplitudes(52, 100, 16, "finger"))
    index = harmonic_near_null - 1
    assert amps[index] < 0.55 * max(amps[index - 1], amps[index + 1])


def test_pluck_modes_preserve_signed_comb_and_velocity():
    soft = phys.partial_amplitudes(52, 35, 16, position=0.2)
    loud = phys.partial_amplitudes(52, 105, 16, position=0.2)
    assert np.all(loud[:4] > 0.0)
    assert np.all(loud[5:9] < 0.0)
    assert abs(loud[4]) < 1e-14
    assert np.linalg.norm(loud) > 3.0 * np.linalg.norm(soft)
    assert np.count_nonzero(phys.partial_amplitudes(52, 0, 16)) == 0


def test_same_pitch_on_different_strings_retains_fretting_timbre():
    # E4 can be played open on the first string or at fret 14 on the D string.
    open_e = phys.partial_frequencies(64, 12, string_index=5)
    fretted_e = phys.partial_frequencies(64, 12, string_index=2)
    assert open_e[0] == pytest.approx(fretted_e[0], rel=1e-14)
    assert abs(open_e[-1] - fretted_e[-1]) > 1.0
    assert phys.t60(1000.0, 64, string_index=2) < phys.t60(
        1000.0, 64, string_index=5
    )


def test_bright_pluck_moves_toward_bridge():
    assert phys.pluck_position("bright") < phys.pluck_position("finger")


def test_strumming_excitation_sits_between_finger_and_pick():
    assert phys.pluck_position("finger") > phys.pluck_position("strum")
    assert phys.pluck_position("strum") > phys.pluck_position("pick")


def test_wound_strings_deliver_more_bridge_force_than_plain_strings():
    assert phys.bridge_force_scale(40) > phys.bridge_force_scale(64)
    assert 0.6 < phys.bridge_force_scale(64) < 1.0
    assert 1.0 < phys.bridge_force_scale(40) < 1.6


def test_high_partials_decay_faster():
    values = phys.t60(np.array([100.0, 400.0, 1600.0, 6400.0]), 52)
    assert np.all(np.diff(values) < 0)
    assert values[0] > 5.0
    assert values[-1] < 2.0


def test_helmholtz_resonance_is_guitar_like():
    assert 75.0 < phys.helmholtz_frequency() < 125.0


def test_partial_count_stays_below_nyquist():
    for note in range(40, 89, 4):
        count = phys.partial_count(note)
        assert phys.partial_frequencies(note, count)[-1] < 0.5 * phys.SAMPLE_RATE


@pytest.fixture(scope="module")
def body():
    path = ROOT / "model" / "body_modes.json"
    if not path.exists():
        pytest.skip("run tools/soundboard_fea.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def test_fea_has_irregular_active_domain(body):
    nx, ny = body["grid"]
    assert body["active_nodes"] < 0.75 * nx * ny


def test_fea_body_modes_are_audible_and_broad(body):
    modes = body["modes"]
    assert len(modes) >= 10
    assert all(70.0 < mode["hz"] < 1600.0 for mode in modes)
    assert all(5.0 < mode["q"] < 50.0 for mode in modes)


def test_stereo_samples_different_mode_shapes(body):
    left = np.array([mode["left"] for mode in body["modes"]])
    right = np.array([mode["right"] for mode in body["modes"]])
    assert np.corrcoef(left, right)[0, 1] < 0.98
