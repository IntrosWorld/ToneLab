"""48 kHz offline DSP; fixed instrument gain, gentle bus dynamics, PCM24 export."""
from __future__ import annotations

import math
from pathlib import Path
import wave

import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import butter, fftconvolve, lfilter, resample_poly, sosfilt

SR = 48_000


def db(value):
    return 20 * np.log10(np.maximum(np.asarray(value), 1e-12))


def filt(x, hz, kind="lowpass", order=2, sr=SR):
    return sosfilt(butter(order, hz, kind, fs=sr, output="sos"), x, axis=0)


def eq(x, hz, gain_db, q=.707, sr=SR):
    """RBJ peaking biquad; zero dB is exactly unity."""
    a = 10 ** (gain_db / 40)
    w = 2 * np.pi * hz / sr
    alpha = np.sin(w) / (2 * q)
    b = [1 + alpha*a, -2*np.cos(w), 1-alpha*a]
    denominator = [1+alpha/a, -2*np.cos(w), 1-alpha/a]
    return lfilter(np.array(b)/denominator[0], np.array(denominator)/denominator[0], x, axis=0)


def compressor(x, threshold_db=-22., ratio=2., attack=.012, release=.13, knee=6.):
    """Linked soft-knee compression; no zero-state fade-in or per-note AGC."""
    a = np.asarray(x)
    power = a*a if a.ndim == 1 else np.mean(a*a, axis=1)
    coefficient = math.exp(-1/(.006*SR))
    rms = np.sqrt(lfilter([1-coefficient], [1, -coefficient], power) + 1e-16)
    over = db(rms) - threshold_db
    soft = np.where(over <= -knee/2, 0.,
                    np.where(over >= knee/2, over, (over+knee/2)**2/(2*knee)))
    target_db = -(1-1/ratio)*soft
    gain_db = np.empty(len(a))
    state = 0.
    ca, cr = math.exp(-1/(attack*SR)), math.exp(-1/(release*SR))
    for i, target in enumerate(target_db):
        coefficient = ca if target < state else cr
        state = coefficient*state + (1-coefficient)*target
        gain_db[i] = state
    gain = 10**(gain_db/20)
    return a * (gain if a.ndim == 1 else gain[:, None])


def saturate(x, drive=.7):
    """Oversampled mild preamp rounding, normalized for unity small-signal gain."""
    up = resample_poly(x, 4, 1, axis=0)
    wet = (np.tanh(drive*up + .015*(drive*up)**2))/drive
    return resample_poly(wet, 1, 4, axis=0)[:len(x)]


def stereo(x):
    return np.column_stack([x, x]) if x.ndim == 1 else x.copy()


def fractional_delay(x, delay_seconds):
    """Four-point cubic interpolation with zero-padding before sample zero."""
    pos = np.arange(len(x), dtype=float) - np.asarray(delay_seconds)*SR
    index = np.floor(pos).astype(np.int64)
    mu = pos-index
    def at(offset):
        ids = index+offset
        return np.where((ids >= 0) & (ids < len(x)), x[np.clip(ids, 0, len(x)-1)], 0.)
    a, b, c, d = at(-1), at(0), at(1), at(2)
    return b + .5*mu*(c-a + mu*(2*a-5*b+4*c-d + mu*(3*(b-c)+d-a)))


def chorus(mono, mix=.29, depth=.0018, rate=.36, voices=2):
    t = np.arange(len(mono))/SR
    wet = np.zeros((len(mono), 2))
    band = filt(filt(mono, 170, "highpass"), 6700)
    for v in range(voices):
        phase = 2*np.pi*v/voices
        delay = .016 + .006*(v % 2) + depth*(
            .85*np.sin(2*np.pi*rate*(1+.11*v)*t+phase)
            + .15*np.sin(2*np.pi*.73*rate*t+.81+phase))
        voice = fractional_delay(band, delay)
        pan = (-.82 if v % 2 == 0 else .82)
        wet[:, 0] += np.sqrt((1-pan)/2)*voice
        wet[:, 1] += np.sqrt((1+pan)/2)*voice
    return stereo(mono) + mix*np.sqrt(2/voices)*wet


def room(x, wet=.10, t60=.8, seed=640):
    """Generated diffuse studio room, with early reflections and a dark tail.

    This is a sound-design room, not a measured or FEA-derived impulse response.
    No prerecorded audio is used.
    """
    dry = stereo(x)
    n = int((t60+.15)*SR)
    t = np.arange(n)/SR
    rng = np.random.default_rng(seed)
    result = dry.copy()
    for channel in range(2):
        ir = filt(filt(rng.standard_normal(n), 220, "highpass"), 4700)
        ir *= np.exp(-np.log(1000)*t/t60)*(1-np.exp(-t/.018))
        ir[t < .031+channel*.004] = 0
        ir *= .56/max(np.linalg.norm(ir), 1e-12)
        for delay, gain in ((.012, .30), (.024, .24), (.039, -.18), (.059, .14), (.087, .09)):
            ir[int((delay+channel*.0017)*SR)] += gain
        send = .72*dry[:, channel] + .28*dry[:, 1-channel]
        result[:, channel] += wet*fftconvolve(send, ir)[:len(dry)]
    return result


def fade_end(x, seconds=.04):
    x = x.copy()
    n = min(len(x), max(1, round(seconds*SR)))
    ramp = .5+.5*np.cos(np.linspace(0, np.pi, n))
    x[-n:] *= ramp if x.ndim == 1 else ramp[:, None]
    return x


def active_rms(x):
    """400 ms blocks; ignore blocks >35 dB below the loudest block."""
    a = stereo(x)
    block = int(.4*SR)
    pad = (-len(a)) % block
    power = np.pad(np.mean(a*a, axis=1), (0, pad)).reshape(-1, block).mean(axis=1)
    if not len(power) or power.max() < 1e-20:
        return 0.
    return float(np.sqrt(power[power > power.max()*10**(-35/10)].mean()))


def master(x, target_db=-20., ceiling_db=-1.5):
    """One loudness gain per file; linked offline lookahead safety only if needed."""
    x = filt(stereo(x), 15, "highpass")
    level = active_rms(x)
    if level < 1e-12:
        return x, {"gain_db": 0., "limiter_reduction_db": 0.}
    gain = 10**(target_db/20)/level
    x *= gain
    ceiling = 10**(ceiling_db/20)
    env = np.max(np.abs(x), axis=1)
    # Two-sided window is intentional: offline processing gives 4 ms lookahead.
    window = int(.008*SR) | 1
    bound = maximum_filter1d(env, size=window, mode="nearest")
    reductions = np.minimum(1., ceiling/np.maximum(bound, 1e-12))
    # Smoothing is backed by a final oversampled peak check; never hard clip PCM.
    smoothed = uniform_filter1d(reductions, size=int(.003*SR)|1, mode="nearest")
    reduction_db = float(-db(smoothed.min()))
    x *= smoothed[:, None]
    x = fade_end(x)
    peak = float(np.max(np.abs(resample_poly(x, 4, 1, axis=0))))
    if peak > ceiling:
        x *= ceiling/peak
    return x, {"gain_db": float(db(gain)), "limiter_reduction_db": reduction_db}


def metrics(x):
    a = stereo(x)
    peak = float(np.max(np.abs(a)))
    rms = float(np.sqrt(np.mean(a*a)))
    left, right = a[:, 0], a[:, 1]
    denom = np.linalg.norm(left)*np.linalg.norm(right)
    mid, side = (left+right)/2, (left-right)/2
    spectrum = np.abs(np.fft.rfft(mid))**2
    freq = np.fft.rfftfreq(len(mid), 1/SR)
    total = max(spectrum.sum(), 1e-20)
    return {
        "seconds": len(a)/SR, "sample_rate": SR, "channels": 2,
        "finite": bool(np.isfinite(a).all()), "peak_dbfs": float(db(peak)),
        "true_peak_4x_dbtp": float(db(np.max(np.abs(resample_poly(a, 4, 1, axis=0))))),
        "rms_dbfs": float(db(rms)), "active_rms_dbfs": float(db(active_rms(a))),
        "crest_db": float(db(peak/max(rms, 1e-12))),
        "dc_max": float(np.max(np.abs(np.mean(a, axis=0)))),
        "stereo_correlation": float(np.dot(left, right)/max(denom, 1e-20)),
        "side_mid_db": float(db(np.linalg.norm(side)/max(np.linalg.norm(mid), 1e-20))),
        "band_energy_fraction": {f"{lo}-{hi}": float(spectrum[(freq>=lo)&(freq<hi)].sum()/total)
                                 for lo,hi in ((20,120),(120,400),(400,2000),(2000,6000),(6000,18000))},
    }


def write_wav(path: Path, audio):
    a = stereo(audio)
    if not np.isfinite(a).all() or np.max(np.abs(a), initial=0) > 1:
        raise ValueError(f"refusing non-finite or clipped PCM: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(24)
    dither = (rng.random(a.shape)-rng.random(a.shape))/(2**23)
    pcm = np.rint(np.clip(a+dither, -1, 1)*(2**23-1)).astype("<i4")
    raw = pcm.reshape(-1).view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(3)
        wav.setframerate(SR)
        wav.writeframes(raw)


def read_wav(path: Path):
    from scipy.io import wavfile
    rate, audio = wavfile.read(path)
    if rate != SR:
        raise ValueError(f"expected {SR} Hz, got {rate}")
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(float)/(-float(np.iinfo(audio.dtype).min))
    return stereo(audio)
