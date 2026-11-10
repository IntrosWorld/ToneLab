"""Physical regression checks for retarded soundboard radiation.

These check acoustic travel, signed geometry, passive loading and the join to
unresolved modes. They do not equate spectral balance with perceived realism.
"""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from acoustic_engine import GuitarBody, SPATIAL_BODY_GAIN, UNRESOLVED_CROSSOVER_HZ


@pytest.fixture(scope="module")
def body():
    return GuitarBody(48_000, ROOT / "model" / "body_modes_v4.json")


@pytest.fixture(scope="module")
def response(body):
    impulse = np.zeros(48_000)
    impulse[0] = 1.
    return body.process(impulse), body.upper_radiation(impulse)


@pytest.mark.parametrize("rate", [12_000, 48_000])
def test_acoustic_energy_waits_for_physical_travel(rate):
    guitar_body = GuitarBody(rate, ROOT / "model" / "body_modes_v4.json")
    impulse = np.zeros(round(.20 * rate))
    offset = 137
    impulse[offset] = 1.
    # Both microphones lie half a metre above the top. Linear interpolation
    # may distribute a fractional arrival into the preceding sample.
    earliest = offset + int(np.floor(.50 / 343. * rate)) - 1
    for audio in (guitar_body.process(impulse),
                  guitar_body.upper_radiation(impulse)):
        assert not np.any(audio[:earliest])
        assert np.linalg.norm(audio[earliest:]) > 0
        assert np.isfinite(audio).all()


def test_spatial_transfer_is_linear_and_time_invariant(body):
    rng = np.random.default_rng(427)
    x = rng.normal(0., .01, 3072)
    y = rng.normal(0., .02, 3072)
    np.testing.assert_allclose(body.process(.3*x + y),
                               .3*body.process(x) + body.process(y),
                               atol=1e-11, rtol=1e-9)
    shifted = np.pad(x, (91, 0))
    np.testing.assert_allclose(body.process(shifted)[91:], body.process(x),
                               atol=1e-11, rtol=1e-9)


def test_opposite_surface_motion_cancels_pressure(body):
    probe = deepcopy(body)
    mode = deepcopy(body.data["coupled_modes"][0])
    opposite = deepcopy(mode)
    for field in ("radiation_left", "radiation_right"):
        opposite[field] = (-np.asarray(mode[field])).tolist()
    probe.data["coupled_modes"] = [mode, opposite]
    impulse = np.zeros(4096)
    impulse[43] = 1.
    # Equal poles with equal/opposite area displacement cancel as pressures,
    # rather than adding as magnitudes or powers.
    np.testing.assert_allclose(probe.process(impulse), 0., atol=1e-14)


def test_delay_kernels_match_independent_rayleigh_frequency_projection(body):
    modes = body.data["coupled_modes"]
    normalization = max(abs(m["bridge"]*m["integrated_radiation"])
                        for m in modes)
    rate = body.data["radiation_kernel_rate"]
    for mode in modes:
        for field, channel in (("radiation_left", "left"),
                               ("radiation_right", "right")):
            kernel = np.asarray(mode[field])
            measured = np.sum(kernel * np.exp(-2j*np.pi*mode["hz"]
                                               * np.arange(len(kernel))/rate))
            # The solver also projects the unbinned mesh directly using
            # exp(-j*k*r)/r. Agreement checks the signed delay-bin export
            # against that independent geometric calculation.
            projected = normalization * complex(mode[channel],
                                                   mode[channel + "_imag"])
            assert abs(measured-projected) < .012*abs(projected) + 1e-8


def test_loading_is_passive_and_uses_mechanical_not_pressure_residues(body):
    frequencies = np.geomspace(65., 12_000., 2500)
    baseline = body.string_body_decay(frequencies, 0)
    assert np.isfinite(baseline).all()
    assert np.all(baseline >= 0.)
    assert np.max(baseline) > 1.
    probe = deepcopy(body)
    for mode in probe.data["coupled_modes"]:
        mode["signed_weight"] *= -3.
        mode["weight"] *= 3.
        for field in ("radiation_left", "radiation_right"):
            mode[field] = (-7.*np.asarray(mode[field])).tolist()
    # Moving the microphone cannot change the mechanical energy drained
    # from a string; loading depends on squared bridge mode shapes.
    np.testing.assert_array_equal(probe.string_body_decay(frequencies, 0),
                                  baseline)
    # A vanishing bridge displacement disconnects the body mechanically.
    for mode in probe.data["coupled_modes"]:
        mode["bridge"] = 0.
    np.testing.assert_array_equal(probe.string_body_decay(frequencies, 0), 0.)


def test_body_resonance_increases_string_energy_loss(body):
    probe = deepcopy(body)
    mode = deepcopy(body.data["coupled_modes"][1])
    probe.data["coupled_modes"] = [mode]
    frequencies = np.array([.7*mode["hz"], mode["hz"], 1.3*mode["hz"]])
    decay = probe.string_body_decay(frequencies, 2)
    assert decay[1] > 10.*max(decay[0], decay[2])
    assert np.all(probe.string_body_decay(frequencies, 0) >
                  probe.string_body_decay(frequencies, 5))


def test_join_to_unresolved_modes_has_no_broad_cancellation_hole(response):
    pressure, upper = response
    modeled = SPATIAL_BODY_GAIN*np.fft.rfft(pressure, axis=0)
    unresolved = np.fft.rfft(upper, axis=0)
    total = modeled + unresolved
    frequencies = np.fft.rfftfreq(len(pressure), 1/48_000)
    join = ((frequencies >= .8*UNRESOLVED_CROSSOVER_HZ)
            & (frequencies < 1.6*UNRESOLVED_CROSSOVER_HZ))
    separate_power = np.mean(abs(modeled[join])**2 + abs(unresolved[join])**2)
    joined_power = np.mean(abs(total[join])**2)
    # Phase can make narrow physical notches, but the whole crossover region
    # should not be lost through an artificial relative microphone delay.
    assert joined_power > .25*separate_power
