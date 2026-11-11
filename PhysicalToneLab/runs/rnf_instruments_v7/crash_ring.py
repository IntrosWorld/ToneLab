"""Crash, metallic ring (after the user heard the new crash as white noise with 'the tin type high note missing').

Measured (crash_tonality.py, held-out 17 inch): in a real crash the 10 strongest partials hold 37-82 % of each band's
energy (spectral flatness -14..-20 dB); in our crash only 9-40 % above 1.2 kHz (flatness -4..-9 dB) = nearly noise.
The flow is a smooth gain, it cannot create narrow partials, and the tone_lab source does not have them.
Change: the ringing partials (frequency, level, decay rate) are MEASURED from one real training hit (18 inch, shoulder,
ff) and added as decaying sine waves; the flow output (the noisy bed) is turned down per band and time by the share the
partials hold in that real hit, so the total stays the same.  This is a measured modal fit, not a network.
  python crash_ring.py fit | check | sample
"""
import sys
sys.dont_write_bytecode = True
import json
import numpy as np
import soundfile as sf
from scipy.signal import stft, istft, resample_poly
import crash_flow as cf

SR, HERE = cf.SR, cf.HERE; MODES = HERE / 'data/crash_ring_modes.npz'; REF = 10; EDGES = [200, 400, 800, 1600, 3200, 6400, 11000, 16000]; PER_BAND = 24; SPACING = 6.


def amp(x, f, a, b):
    seg = x[int(a * SR):int(b * SR)]; w = np.hanning(len(seg)); t = np.arange(len(seg)) / SR; return 2 * np.abs((seg * w) @ np.exp(-2j * np.pi * np.outer(t, f))) / w.sum()


def measure(x):
    seg = x[int(.5 * SR):int(1.5 * SR)]; S = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))); fq = np.fft.rfftfreq(len(seg), 1 / SR); F = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        i = np.where((fq >= lo) & (fq < hi))[0]; got = []
        for j in i[np.argsort(S[i])[::-1]]:
            if 0 < j < len(S) - 1 and S[j] >= S[j - 1] and S[j] >= S[j + 1] and all(abs(fq[j] - g) >= SPACING for g in got):
                got.append(fq[j])
                if len(got) == PER_BAND: break
        F += got
    F = np.array(sorted(F)); T = np.arange(.1, 2.5, .2); A = np.stack([amp(x, F, t, t + .4) for t in T]); db = 20 * np.log10(A + 1e-9); tc = T + .2
    slope, icpt = np.polyfit(tc, db, 1); rate = np.clip(-slope, .5, 40.); a0 = 10 ** ((db.mean(0) + rate * tc.mean()) / 20); return F, a0, rate


def ring(F, a0, rate, L, seed=1):
    t = np.arange(L) / SR; ph = np.random.default_rng(seed).uniform(0, 2 * np.pi, len(F)); y = np.zeros(L)
    for f, a, r, p in zip(F, a0, rate, ph): y += a * 10 ** (-r * t / 20) * np.sin(2 * np.pi * f * t + p)
    return y * np.minimum(1., t / .004)


def band_power(x):
    f, t, Z = stft(x, SR, 'hann', 2048, 1536); P = np.abs(Z) ** 2; return t, np.stack([P[(f >= lo) & (f < hi)].sum(0) for lo, hi in zip(EDGES[:-1], EDGES[1:])])


def fit():
    d = np.load(cf.PAIRS); r = d['R'][REF].astype(float); e = d['E'][REF].astype(float); F, a0, rate = measure(r); y = ring(F, a0, rate, len(r)); t, pr = band_power(r); _, py = band_power(y)
    k = np.ones(9) / 9; share = np.stack([np.convolve(a, k, 'same') / (np.convolve(b, k, 'same') + 1e-12) for a, b in zip(py, pr)]).clip(0, .97); n = int(.1 * SR)
    np.savez(MODES, F=F, a0=a0, rate=rate, share=share, t=t, ref_rms=np.sqrt((e[:n] ** 2).mean())); print('partials', len(F), 'cymbal size', int(d['size'][REF]))
    for i, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        j = (F >= lo) & (F < hi); print(f'{lo}-{hi} Hz: decay dB/s median {np.median(rate[j]):.1f}, share of real band energy at .3/1/2 s:', [round(float(share[i, np.argmin(abs(t - s))]), 2) for s in (.3, 1., 2.)], 'strongest', np.round(F[j][np.argsort(a0[j])[::-1][:4]]).astype(int))


def render(model, e, seed=1):
    """e: one engine crash hit in training units (engine * scale), starting at the onset.  Returns bed (reduced) + ring."""
    m = np.load(MODES); n = int(.1 * SR); L = len(e); bed = cf.process(model, e); f, t, Z = stft(bed, SR, 'hann', 2048, 1536)
    g = np.sqrt(1 - np.stack([np.interp(t, m['t'], s, right=s[-1]) for s in m['share']])); c = np.log(np.sqrt(np.array(EDGES[:-1]) * np.array(EDGES[1:]))); G = np.stack([np.interp(np.log(f.clip(1)), c, g[:, i]) for i in range(len(t))], 1)
    bed = istft(Z * G, SR, 'hann', 2048, 1536)[1][:L]; bed = np.pad(bed, (0, L - len(bed))); o = max(0, int(np.argmax(np.abs(e) > .1 * np.abs(e).max())) - 64); k = np.sqrt((e[o:o + n] ** 2).mean()) / float(m['ref_rms'])
    y = bed.copy(); r = ring(m['F'], m['a0'], m['rate'], L - o, seed) * k; y[o:] += r; return y


def check():
    import crash_tonality as ct
    d = np.load(cf.PAIRS); model = cf.load(); idx = np.where(d['size'] == 17)[0]; out = {}
    sets = dict(real=[d['R'][i].astype(float) for i in idx], flow_only=[cf.process(model, d['E'][i].astype(float)) for i in idx], flow_plus_ring=[render(model, d['E'][i].astype(float), seed=int(i)) for i in idx])
    for n, xs in sets.items():
        s = np.mean([ct.stats(x) for x in xs], 0); out[n] = np.round(s, 1).tolist()
        for w, r in zip(ct.WINS, s): print(n, w, 'flatness dB', np.round(r[:, 1]).astype(int), 'top10 %', np.round(r[:, 2]).astype(int))
    edges = [200, 500, 1000, 2000, 4000, 8000, 16000]; T = (0., .1, .3, .6, 1.0, 1.5, 2.2)
    def prof(x):
        o = []
        for t in T:
            seg = x[int(t * SR):int(t * SR) + 4096]; S = np.abs(np.fft.rfft(seg * np.hanning(4096))) ** 2; f = np.fft.rfftfreq(4096, 1 / SR); o.append([10 * np.log10(S[(f >= a) & (f < b)].sum() + 1e-12) for a, b in zip(edges[:-1], edges[1:])])
        return np.array(o)
    pr = np.mean([prof(x) for x in sets['real']], 0)
    for n in ('flow_only', 'flow_plus_ring'):
        e = np.mean([prof(x) for x in sets[n]], 0) - pr; out[n + '_band_error_db'] = np.round(e).astype(int).tolist(); out[n + '_mean_abs_db'] = round(float(np.abs(e).mean()), 1); print(n, 'band level minus real (rows = time, cols = bands)'); print(np.round(e).astype(int)); print('mean abs dB', out[n + '_mean_abs_db'])
    (HERE / 'crash_ring_summary.json').write_text(json.dumps(out, indent=1))


def hit(tm, k, model, s, velocity=108, seed=1):
    """One finished crash hit at 44.1 kHz in engine units."""
    e = cf.engine_hit(tm, k, velocity, seed) * s; return render(model, np.pad(e, (0, max(0, int(3.2 * SR) - len(e)))), seed) / s


def sample():
    tm, k = cf.kit(); model = cf.load(); s = float(np.load(cf.PAIRS)['scale']); x = hit(tm, k, model, s); out = HERE.parent / 'rnf_song_ascend/drum_samples'; y = np.zeros(int(13.5 * SR))
    for j in range(4): a = int((.2 + 3.2 * j) * SR); n = min(len(x), len(y) - a); y[a:a + n] += x[:n]
    sf.write(out / '4d_crash_with_measured_ring.flac', y * (.9 / np.abs(y).max()), SR, subtype='PCM_16'); print('written', len(x) / SR, 's per hit')


if __name__ == '__main__':
    dict(fit=fit, check=check, sample=sample)[sys.argv[1]]()
