"""Independent regressions for resonator state and validation-tool options."""
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from audio_dsp import pedal_resonator
from render_model import GrandPiano
import validate_render


def test_pedal_resonator_matches_undamped_transfer_function_while_held():
    rate, frequency, t60, gain = 8000, 261.63, 12.0, .4
    source = np.random.default_rng(72).normal(size=1000)
    radius = np.exp(-np.log(1000) / (t60 * rate))
    angle = 2 * np.pi * frequency / rate
    expected = lfilter([gain * (1-radius) * np.sin(angle)],
                       [1, -2*radius*np.cos(angle), radius*radius], source)
    actual = pedal_resonator(source, rate, [(0, 1)], frequency, t60, .2, gain)
    np.testing.assert_allclose(actual, expected, atol=1e-13)


def test_pedal_resonator_matches_sample_recurrence_at_state_changes():
    rate, frequency, t60, damp, gain = 8000, 987.7, 4.0, .12, .7
    source = np.random.default_rng(28).normal(size=300)
    spans = [(2/rate, 3/rate), (40/rate, 100/rate), (80/rate, 130/rate),
             (150/rate, 180/rate), (180/rate, 181/rate)]
    radius = np.exp(-np.log(1000)/(t60*rate))
    damp_radius = radius*np.exp(-np.log(1000)/(damp*rate))
    angle = 2*np.pi*frequency/rate
    b = gain*(1-radius)*np.sin(angle)
    expected = np.zeros_like(source)
    for i in range(len(source)):
        r = radius if any(int(a*rate) <= i < int(z*rate) for a,z in spans) else damp_radius
        previous = expected[i-1] if i else 0
        previous2 = expected[i-2] if i >= 2 else 0
        expected[i] = b*source[i] + 2*r*np.cos(angle)*previous - r*r*previous2
    actual = pedal_resonator(source, rate, spans, frequency, t60, damp, gain)
    np.testing.assert_allclose(actual, expected, atol=1e-13)


def test_repedal_cannot_resurrect_a_damped_sympathetic_note():
    piano = GrandPiano(8000, room_mix=0, body_mix=0, normalize=False)
    piano.UNDAMPED_ALWAYS = 0
    source = np.zeros(16000)
    source[:400] = np.sin(2*np.pi*261.63*np.arange(400)/8000)
    response = piano._sympathetic(source, [(0, .1), (1, 1.4)], len(source))
    initial = np.sqrt(np.mean(response[640:800]**2))
    reopened = np.sqrt(np.mean(response[8400:9200]**2))
    assert reopened < initial * 1e-4


def test_quick_repedal_retains_only_the_residual_string_energy():
    rate = 8000
    source = np.zeros(4000)
    source[0] = 1
    held = pedal_resonator(source, rate, [(0, .5)], 261.63, 12, .1)
    caught = pedal_resonator(source, rate, [(0, .1), (.15, .5)], 261.63, 12, .1)
    ratio = np.linalg.norm(caught[1600:3200]) / np.linalg.norm(held[1600:3200])
    assert .02 < ratio < .05  # 50 ms under a 100 ms T60 damper is about -30 dB.


def test_validate_jobs_preserve_preview_and_full_requests(tmp_path, monkeypatch):
    jobs = []
    class FakePerformance:
        events = [(60, 90, 0, 100)]
        pedal_spans = []
        duration = 100
        warnings = []
    monkeypatch.setattr(validate_render, "load_performance", lambda path: FakePerformance())
    monkeypatch.setattr(validate_render, "GrandPiano", lambda rate: None)
    def record(piano, events, spans, destination, tail):
        jobs.append((destination.name, events[0][3]))
        return {}
    monkeypatch.setattr(validate_render, "measure_render", record)
    monkeypatch.setattr(sys, "argv", ["validate_render", "--numbers", "9", "15", "--full", "12",
                                     "--seconds", "3", "--out-dir", str(tmp_path)])
    validate_render.main()
    assert jobs == [("demo9_3s_improved.wav", 3), ("demo15_3s_improved.wav", 3),
                    ("demo12_full_improved.wav", 100)]


def test_validation_accepts_output_directory_outside_project(tmp_path):
    class FakePiano:
        sample_rate = 8000
        def render(self, events, spans, tail):
            audio = np.zeros((100, 2))
            audio[10, :] = .1
            return audio
    destination = tmp_path / "result.wav"
    result = validate_render.measure_render(FakePiano(), [], [], destination, 0)
    assert result["path"] == str(destination.resolve())
    assert destination.exists()
