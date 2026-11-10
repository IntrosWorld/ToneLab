"""Audible failure checks: tuning, articulation, dynamics, choking and PCM safety."""
import json
from pathlib import Path
import wave

import numpy as np
from scipy.signal import resample_poly

from tone_lab import dsp
from tone_lab.score import ROOT, Gesture, load_guitar
from tone_lab.guitar import CleanGuitar, PRESETS
from tone_lab.bass.model import BassNote, FingerBass, load_bass
from tone_lab.drums.model import AcousticKit, DrumHit, load_drums


def rms(x):
    return np.sqrt(np.mean(x*x))


def measured_frequency(audio, expected):
    x = audio[round(.18*dsp.SR):round(.78*dsp.SR)]
    n = 2**int(np.ceil(np.log2(len(x)*16)))
    spectrum = np.abs(np.fft.rfft(x*np.hanning(len(x)), n))
    frequencies = np.fft.rfftfreq(n, 1/dsp.SR)
    choices = np.flatnonzero((frequencies > expected*.97)&(frequencies < expected*1.03))
    k = choices[np.argmax(spectrum[choices])]
    return frequencies[k]


def test_gp1_exact_short_phrase_and_effects():
    events, report = load_guitar(ROOT/"ref/tab1.gp")
    assert len(events) == 47 and report["raw_notes"] == 61
    assert sum(e.articulation == "pull_off" for e in events) == 4
    assert sum(e.articulation == "bend_up" for e in events) == 8
    assert sum(e.articulation == "dead_slide_in" for e in events) == 2
    assert 12.2 < max(e.start+e.duration for e in events) < 12.3
    assert load_bass(ROOT/"ref/tab1.gp", 12.3)[0] == []


def test_gp2_diagnostic_has_real_bass_and_drums():
    notes, _ = load_bass(ROOT/"ref/tab2.gp")
    hits, _ = load_drums(ROOT/"ref/tab2.gp")
    assert len(notes) == 25
    assert len(hits) > 40
    assert {h.piece for h in hits} == {"kick", "snare", "hihat", "crash"}
    assert max(n.start for n in notes) < 12.3


def test_bass_recorded_waveform_is_in_tune():
    model = FingerBass()
    for midi, string in ((28, 0), (40, 0), (43, 3)):
        expected = 440*2**((midi-69)/12)
        audio = model.note(BassNote(0, 1.1, midi, 100, string), np.random.default_rng(4))
        observed = measured_frequency(audio, expected)
        assert abs(1200*np.log2(observed/expected)) < 2., (midi, observed, expected)


def test_clean_guitar_high_fret_stays_in_tune():
    model = CleanGuitar()
    event = Gesture(0, 1.1, 79, string_index=5)
    audio, _ = model.note(event, PRESETS["clean_neck"], np.random.default_rng(4))
    expected = 440*2**((79-69)/12)
    assert abs(1200*np.log2(measured_frequency(audio, expected)/expected)) < 2.


def test_bass_dynamic_range_survives_synthesis():
    model = FingerBass()
    soft = model.note(BassNote(0, .7, 33, 48, 1), np.random.default_rng(4))
    hard = model.note(BassNote(0, .7, 33, 116, 1), np.random.default_rng(4))
    assert 20*np.log10(rms(hard)/rms(soft)) > 9
    assert rms(hard[-1000:]) < rms(hard[:10000])*1e-5


def test_snare_ghost_notes_and_repeated_hits():
    kit = AcousticKit()
    rng = np.random.default_rng(4)
    soft = kit.hit(DrumHit(0, "snare", 42), rng)
    hard = kit.hit(DrumHit(0, "snare", 118), rng)
    second = kit.hit(DrumHit(0, "snare", 118), rng)
    assert 20*np.log10(rms(hard)/rms(soft)) > 10
    assert np.corrcoef(hard[:8000], second[:8000])[0, 1] < .96


def test_closed_hihat_chokes_open_tail():
    kit = AcousticKit()
    open_hit = DrumHit(0, "hihat", 110, 1.)
    full = kit.hit(open_hit, np.random.default_rng(4))
    choked = kit.hit(open_hit, np.random.default_rng(4), .212)
    assert len(choked) == round(.212*dsp.SR)
    assert abs(choked[-1]) < 1e-12
    assert rms(full[12000:18000]) > 1e-4


def test_all_kit_pieces_have_audible_finite_output():
    kit = AcousticKit()
    for piece in ("kick", "snare", "hihat", "crash", "ride", "tom_high", "tom_mid", "tom_low", "tom_floor"):
        hit = kit.hit(DrumHit(0, piece, 100), np.random.default_rng(4), .15)
        assert np.isfinite(hit).all() and rms(hit) > .001, piece


def test_bass_is_exactly_centred_and_deterministic():
    model = FingerBass()
    notes = [BassNote(.1, .3, 33, 100, 1), BassNote(.3, .3, 35, 90, 1)]
    x = model.render(notes, .9)
    assert np.array_equal(x[:, 0], x[:, 1])
    assert np.array_equal(x, model.render(notes, .9))
    assert np.isfinite(x).all()


def test_chorus_has_width_and_preserves_mono_fundamental():
    t = np.arange(dsp.SR*2)/dsp.SR
    mono = .2*np.sin(2*np.pi*220*t)+.05*np.sin(2*np.pi*660*t)
    x = dsp.chorus(mono, **PRESETS["chorus_classic"]["chorus"])
    assert rms(x[:, 0]-x[:, 1]) > .01
    assert rms(x.mean(axis=1)) > .70*rms(mono)
    assert np.corrcoef(x.T)[0, 1] > .5


def test_master_and_pcm_preserve_silence_and_peak_headroom(tmp_path):
    t = np.arange(dsp.SR)/dsp.SR
    audio = .2*np.sin(2*np.pi*9700*t)
    audio[14000] = 4.
    audio[:10000] = 0
    mastered, _ = dsp.master(audio)
    assert np.max(np.abs(resample_poly(mastered, 4, 1, axis=0))) <= 10**(-1.5/20)+1e-8
    assert np.max(np.abs(mastered[:9000])) < 1e-12
    path = tmp_path/"test.wav"
    dsp.write_wav(path, mastered)
    with wave.open(str(path)) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (2, 3, 48000)
    assert np.max(np.abs(dsp.read_wav(path)-mastered)) < 3e-7


def test_all_finished_renders_are_finite_and_have_headroom():
    path = Path(__file__).resolve().parents[1]/"out/manifest.json"
    if not path.exists():
        return  # This invariant also runs before the optional listening files exist.
    data = json.loads(path.read_text())
    for name, report in data["renders"].items():
        assert report["finite"], name
        assert report["true_peak_4x_dbtp"] <= -1.49, name
        assert report["dc_max"] < .001, name
