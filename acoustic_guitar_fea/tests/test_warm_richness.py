"""Physical safeguards for the full-bodied pluck and coupled acoustic body.

These bounds reject missing fundamentals, unstable randomized excitation and
pitch-dependent loss regressions; they are not a numerical test of realism.
"""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from acoustic_engine import GuitarBody
from plucked_string import render_plucked_string

RATE = 48_000


def test_coupled_body_is_linear_and_causal():
    body = GuitarBody(12_000, ROOT / "model/body_modes_v3.json")
    assert body.data["coupled_modes"]
    rng = np.random.default_rng(312)
    a = np.pad(rng.normal(0, .02, 2048), (128, 0))
    b = np.pad(rng.normal(0, .03, 2048), (128, 0))
    response = body.process(a)
    assert not np.count_nonzero(response[:128])
    assert np.linalg.norm(response) > 0
    np.testing.assert_allclose(body.process(.17 * a + b),
                               .17 * response + body.process(b),
                               rtol=1e-9, atol=1e-11)


def test_opposite_signed_radiation_residues_cancel():
    body = GuitarBody(12_000, ROOT / "model/body_modes_v3.json")
    signed = np.array([m["signed_weight"] for m in body.data["coupled_modes"]])
    assert np.any(signed > 0) and np.any(signed < 0)
    # Opposite geometric pressure residues at the same pole must cancel.
    # Taking absolute weights or combining modal powers destroys this law.
    mode = deepcopy(body.data["coupled_modes"][0])
    opposite = deepcopy(mode)
    opposite["signed_weight"] = -mode["signed_weight"]
    body.data["coupled_modes"] = [mode, opposite]
    excitation = np.zeros(4096)
    excitation[128] = 1
    np.testing.assert_allclose(body.process(excitation), 0, atol=1e-14)


@pytest.mark.parametrize("fundamental,string_index", [(82.4069, 0), (164.8138, 2)])
def test_bass_pluck_retains_low_harmonic_body(fundamental, string_index):
    for seed in (42, 1946, 7251):
        force = render_plucked_string(fundamental, .4, RATE, 90, .15,
                                      string_index=string_index, seed=seed)
        segment = force[round(.03 * RATE):round(.20 * RATE)]
        power = np.abs(np.fft.rfft(segment * np.hanning(len(segment))))**2
        frequencies = np.fft.rfftfreq(len(segment), 1 / RATE)
        # Bridge-force body must remain in the first few partials; a random
        # comb packet can move most energy into a bright, hollow upper band.
        assert power[frequencies < 3.5 * fundamental].sum() / power.sum() > .70


def test_repeated_equal_plucks_do_not_randomly_lose_the_fundamental():
    fundamental = 622.254
    amplitudes = []
    for seed in range(12):
        force = render_plucked_string(fundamental, .35, RATE, 90, .13,
                                      string_index=5, seed=seed)
        low = sosfiltfilt(butter(4, (fundamental * .85, fundamental * 1.15),
                                 fs=RATE, btype="bandpass", output="sos"), force)
        amplitudes.append(np.linalg.norm(low[2400:12000]))
    # Subtle contact variation is welcome; 10–15 dB random fundamental losses
    # at identical velocity turn otherwise equal notes into hollow plucks.
    assert 20 * np.log10(max(amplitudes) / min(amplitudes)) < 3


def test_absolute_partial_decay_remains_comparable_across_frets():
    losses_db = []
    for fundamental in (165., 660.):
        force = render_plucked_string(fundamental, 1., RATE, 90, .14,
                                      string_index=5, seed=42)
        # 2640 Hz is the 16th partial of the low note and 4th of the high one.
        # Material loss at the same frequency cannot become several times
        # faster solely because the high note makes more loop round trips.
        partial = sosfiltfilt(butter(4, (2530., 2750.), fs=RATE,
                                     btype="bandpass", output="sos"), force)
        early = np.linalg.norm(partial[2400:7200])
        late = np.linalg.norm(partial[16800:21600])
        losses_db.append(20 * np.log10(late / early))
    assert abs(losses_db[0] - losses_db[1]) < 4
