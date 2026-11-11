"""Fixed-point representation and RTL agreement.

The representation is the part of a hardware port most likely to quietly ruin
the sound, and it is where the earlier `modal_piano` in this workspace came
apart.  These tests pin it down.
"""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import hw_model as hw          # noqa: E402
import piano_physics as phys   # noqa: E402


def recovered_frequency(pitch):
    """Frequency the hardware actually produces, after quantisation."""
    f0 = float(phys.fundamental(pitch))
    k, _ = hw.coefficients(f0, float(phys.decay_rate(f0)))
    kq = hw.to_fixed(k, hw.COEF_Q, hw.COEF_W) / (1 << hw.COEF_Q)
    return 2.0 * math.asin(min(1.0, kq / 2.0)) * hw.SAMPLE_RATE / (2.0 * math.pi)


def recovered_t60(pitch):
    f0 = float(phys.fundamental(pitch))
    _, loss = hw.coefficients(f0, float(phys.decay_rate(f0)))
    lq = hw.to_fixed(loss, hw.LOSS_Q, hw.LOSS_W) / (1 << hw.LOSS_Q)
    if lq <= 0.0:
        return float("inf")
    return math.log(1000.0) / (-math.log(1.0 - lq) * hw.SAMPLE_RATE)


@pytest.mark.parametrize("pitch", [21, 28, 40, 60, 76, 96, 108])
def test_quantised_pitch_is_inaudibly_close(pitch):
    want = float(phys.fundamental(pitch))
    got = recovered_frequency(pitch)
    cents = 1200 * math.log2(got / want)
    assert abs(cents) < 0.01


@pytest.mark.parametrize("pitch", [21, 28, 40, 60, 76, 96, 108])
def test_quantised_decay_is_close(pitch):
    want = float(phys.t60(float(phys.fundamental(pitch))))
    got = recovered_t60(pitch)
    assert abs(got - want) / want < 0.02


def test_naive_direct_form_would_fail():
    """Why the magic-circle form is used instead of a direct-form biquad.

    Storing cos(theta) and r = exp(-rate/fs) in Q1.16 -- the obvious choice --
    rounds both to exactly 1.0 in the bass.  The bottom A then never oscillates
    and never decays.  This test documents the trap rather than the fix.
    """
    f0 = float(phys.fundamental(21))
    theta = 2 * math.pi * f0 / hw.SAMPLE_RATE
    r = math.exp(-float(phys.decay_rate(f0)) / hw.SAMPLE_RATE)
    cos_q = round(math.cos(theta) * (1 << 16)) / (1 << 16)
    r_q = round(r * (1 << 16)) / (1 << 16)
    assert cos_q == 1.0      # no oscillation
    assert r_q == 1.0        # no decay

    # The representation actually used keeps both well away from degeneracy.
    k, loss = hw.coefficients(f0, float(phys.decay_rate(f0)))
    assert hw.to_fixed(k, hw.COEF_Q, hw.COEF_W) > 1000
    assert hw.to_fixed(loss, hw.LOSS_Q, hw.LOSS_W) > 100


def test_every_note_fits_the_slot_budget():
    for pitch in range(phys.LOWEST_MIDI, phys.HIGHEST_MIDI + 1):
        assert len(hw.components(pitch)) <= hw.SLOTS_PER_VOICE


def test_rom_is_consistent_with_itself():
    rom = hw.build_rom()
    assert rom["entries"] == len(rom["k"]) == len(rom["loss"])
    for layer in rom["amp"]:
        assert len(layer) == rom["entries"]
    for pitch_index, (base, count) in enumerate(zip(rom["base"], rom["count"])):
        assert base + count <= rom["entries"], pitch_index


# ---------------------------------------------------------------------------
# RTL agreement
# ---------------------------------------------------------------------------
SIM = ROOT / "out" / "build_strings" / ("piano_strings.exe" if sys.platform == "win32" else "piano_strings")


@pytest.mark.skipif(not SIM.exists(), reason="run scripts/verilate.sh first")
def test_rtl_matches_the_bit_accurate_model():
    """The RTL must reproduce the Python fixed-point model, not merely sound similar.

    The RTL emits each sample one period after computing it, so the streams are
    compared with that one-sample offset removed.
    """
    samples = 4000
    text = ROOT / "out" / "rtl_check.txt"
    subprocess.run(
        [str(SIM), "--pitch", "60", "--velocity", "100",
         "--seconds", str(samples / hw.SAMPLE_RATE), "--text", str(text)],
        cwd=str(ROOT), check=True, capture_output=True)

    # The core emits the left dry bus, which carries this note's microphone
    # gain and arrival delay, so both are applied to the reference.
    stereo = hw.stereo_rom()
    delay, _, pan_l, _ = stereo[60 - phys.LOWEST_MIDI]
    ref = hw.simulate_note(60, 100, samples) * (1 << hw.STATE_Q)
    ref = ref * (pan_l / (1 << hw.GAIN_Q))

    rtl = np.loadtxt(text)

    # Align: the arrival delay plus the one-cycle output pipeline.
    best = -1e9
    for lag in range(0, delay + 4):
        n = min(len(rtl) - lag, len(ref))
        if n < 1000:
            continue
        err = rtl[lag:lag + n] - ref[:n]
        sig = np.sqrt((ref[:n] ** 2).mean())
        noise = np.sqrt((err ** 2).mean())
        best = max(best, 20 * np.log10(sig / max(noise, 1e-12)))
    assert best > 90, f"best SNR {best:.1f} dB"


FULL_SIM = ROOT / "out" / "build_top" / ("piano_sim.exe" if sys.platform == "win32" else "piano_sim")


@pytest.mark.skipif(not FULL_SIM.exists(), reason="run scripts/verilate.sh first")
def test_a_chord_sounds_every_note():
    """Simultaneous note-ons must all be started.

    A chord arrives as several note-ons on consecutive clocks.  The core can
    only begin one note per audio sample, so without a queue every note but the
    first is dropped -- and because the testbench emits them in event order,
    what actually went missing was the bass.
    """
    import wave

    events = ROOT / "out" / "chord_events.txt"
    events.write_text("on 0.000000 40 100\non 0.000000 64 100\n", encoding="utf-8")
    wav = ROOT / "out" / "chord_check.wav"
    subprocess.run(
        [str(FULL_SIM), "--events", str(events), "--seconds", "0.4",
         "--wav", str(wav)],
        cwd=str(ROOT), check=True, capture_output=True)

    with wave.open(str(wav), "rb") as f:
        rate, channels = f.getframerate(), f.getnchannels()
        raw = f.readframes(f.getnframes())
    b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
    v = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8)
         | (b[:, 2].astype(np.int32) << 16))
    v = (v ^ 0x800000) - 0x800000
    mono = v.reshape(-1, channels).astype(float).mean(axis=1)

    n = 1 << 14
    seg = mono[:n] * np.hanning(min(n, len(mono)))[: len(mono[:n])]
    if len(seg) < n:
        seg = np.pad(seg, (0, n - len(seg)))
    spec = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(n, 1.0 / rate)
    peak = spec.max()

    for pitch in (40, 64):
        target = float(phys.fundamental(pitch))
        band = (freqs > target * 0.97) & (freqs < target * 1.03)
        level = 20 * np.log10(spec[band].max() / peak)
        assert level > -30, f"MIDI {pitch} missing from the chord ({level:.1f} dB)"



@pytest.mark.skipif(not FULL_SIM.exists(), reason="run scripts/verilate.sh first")
def test_note_off_during_the_load_queue_still_damps():
    """A note-off can arrive before its note-on has been loaded.

    The core starts one note per audio sample, so a queued note has no voice
    yet and a note-off has nothing to match.  The note would then load with its
    key still down and never damp -- it rings for the rest of the piece.  Real
    material triggers this: demo-6 has note-ons and note-offs two microseconds
    apart, well inside one 20.8 us sample, and two pitches rang on forever.
    """
    import wave

    events = ROOT / "out" / "quickoff_events.txt"
    events.write_text("on 0.000000 60 100\noff 0.000002 60 0\n", encoding="utf-8")
    wav = ROOT / "out" / "quickoff.wav"
    subprocess.run([str(FULL_SIM), "--events", str(events),
                    "--seconds", "1.5", "--wav", str(wav)],
                   cwd=str(ROOT), check=True, capture_output=True)

    with wave.open(str(wav), "rb") as f:
        rate, channels = f.getframerate(), f.getnchannels()
        raw = f.readframes(f.getnframes())
    b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
    v = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8)
         | (b[:, 2].astype(np.int32) << 16))
    mono = ((v ^ 0x800000) - 0x800000).reshape(-1, channels).astype(float).mean(axis=1)

    win = max(1, int(rate * 0.05))
    env = np.sqrt(np.convolve(mono ** 2, np.ones(win) / win, mode="same"))
    peak = env.max()
    assert peak > 0

    late = env[int(rate * 1.2):]
    level = 20 * np.log10(late.max() / peak)
    assert level < -40, f"note still ringing at {level:.1f} dB after its note-off"
