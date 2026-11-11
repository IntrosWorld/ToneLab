"""Electric-guitar tones from real hardware: the full resonant flow (v6 architecture) trained separately on three
real distortion pedals of EGFxSet (real Stratocaster, clean vs Blues Driver / Tube Screamer / RAT, CC BY 4.0).

Same task, metric, models and controls as rnf_resonant_amp_v6 (complex harmonic envelopes, error-to-signal ratio),
but here both the guitar and the effect are real recordings.  Notes are split by fret:
  test = frets 3, 8, 13, 18   validation = frets 5, 15   train = the rest.
Writes only inside this folder.

  python tones.py extract <tone>            # paired envelopes for one pedal
  python tones.py fit <tone> <arm> <seed> [steps]
  python tones.py test                      # all tones, all trained checkpoints, test frets (run once at the end)
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent; V7 = HERE.parent; V6 = V7.parent / 'rnf_resonant_amp_v6'; DATA = V7 / 'data/egfx'
sys.path.insert(0, str(V6))
import extract as ex, models, train as tr
TONES = ('BluesDriver', 'TubeScreamer', 'RAT'); ARMS = ('resonant', 'shuffled', 'wiener_hammerstein', 'blackbox_large')
SPLIT = lambda fret: 2 if fret in (3, 8, 13, 18) else 1 if fret in (5, 15) else 0


def read(path):
    x, sr = sf.read(path, dtype='float64', always_2d=True); x = x.mean(1); return resample_poly(x, 147, 160) if sr == 48000 else x


def lag_of(x, y, span=2000):
    """Delay of y after x in samples, from the cross-correlation of the low-passed (fundamental-dominated) signals."""
    n = 1 << 17; lp = lambda v: np.fft.irfft(np.fft.rfft(v[:n], n) * (np.fft.rfftfreq(n, 1 / ex.SR) < 400), n); a, b = lp(x), lp(y)
    c = np.fft.irfft(np.fft.rfft(b) * np.fft.rfft(a).conj()); c = np.concatenate([c[-span:], c[:span]]); return int(np.argmax(np.abs(c))) - span


def extract(tone):
    out = HERE / tone; out.mkdir(parents=True, exist_ok=True); names = sorted(p.name for p in (DATA / 'Clean').glob('*.wav') if (DATA / tone / p.name).exists())
    lag = 0          # measured from note attacks: clean and pedal files are aligned to within ~3 samples (re-amped recordings)
    print(tone, 'files', len(names), flush=True)
    rows = dict(B=[], A=[], f0=[], K=[], split=[], fid=[], t=[]); stats = dict(frames=0, pitched=0, kept=0, target_resid=[])
    for fi, n in enumerate(names):
        string, fret = map(int, n[:-4].split('-')); x, y = read(DATA / 'Clean' / n), read(DATA / tone / n); y = np.roll(y, -lag); m = min(len(x), len(y)); x, y = x[:m] * (.25 / max(np.abs(x).max(), 1e-9)), y[:m]
        keep = []
        for s in np.arange(0, m - ex.WIN, ex.HOP):
            fr = x[s:s + ex.WIN]; stats['frames'] += 1
            if np.sqrt(np.mean(fr ** 2)) < .004: continue
            f0 = ex.yin(fr * ex.W)
            if f0 is None or not 60 < f0 < 1400: continue
            keep.append((s, ex.refine(fr, f0)))
        if not keep: continue
        S = np.array([k[0] for k in keep]); F = np.array([k[1] for k in keep]); X = np.stack([x[s:s + ex.WIN] for s in S]); Y = np.stack([y[s:s + ex.WIN] for s in S]); stats['pitched'] += len(S)
        B, A = ex.harmonics(X, F, ex.H), ex.harmonics(Y, F, ex.H); K = np.minimum(ex.H, np.floor(ex.TOP / F)).astype(int); mask = np.arange(1, ex.H + 1)[None] <= K[:, None]; B, A = B * mask, A * mask
        ok = ((X - ex.resynth(B, F)) ** 2 * ex.W ** 2).sum(1) / np.maximum((X ** 2 * ex.W ** 2).sum(1), 1e-12) < .03
        stats['target_resid'] += (((Y - ex.resynth(A, F)) ** 2 * ex.W ** 2).sum(1) / np.maximum((Y ** 2 * ex.W ** 2).sum(1), 1e-12))[ok].tolist(); stats['kept'] += int(ok.sum())
        for k, v in zip(rows, (B[ok], A[ok], F[ok], K[ok], np.full(ok.sum(), SPLIT(fret)), np.full(ok.sum(), fi), S[ok] / ex.SR)): rows[k].append(v)
    np.savez(str(out / 'envelopes.npz'), **{k: np.concatenate(v) for k, v in rows.items()}); sp = np.concatenate(rows['split'])
    rep = dict(tone=tone, files=len(names), delay_samples=lag, frames=stats['frames'], pitched=stats['pitched'], kept=stats['kept'], target_harmonic_residual_median=float(np.median(stats['target_resid'])), split_frames=[int((sp == i).sum()) for i in range(3)])
    (out / 'extract_report.json').write_text(json.dumps(rep, indent=1)); print(json.dumps(rep), flush=True)


def fit(tone, arm, seed, steps=1500):
    tr.HERE = HERE / tone; tr.fit(arm, int(seed), int(steps))


def test():
    import torch
    res = {}
    for tone in TONES:
        tr.HERE = HERE / tone
        if not (tr.HERE / 'envelopes.npz').exists(): continue
        B, A, f0, mask = tr.data(2); row = dict(test_frames=len(B), doing_nothing=float(((B * (B.conj() * A).sum() / B.abs().square().sum() - A).abs().square().sum() / A.abs().square().sum())))
        for arm in ARMS:
            vals = []; pars = None
            for ck in sorted((tr.HERE / 'checkpoints').glob(f'{arm}_seed*.pt')):
                c = torch.load(ck); m = models.build(arm, c['seed']); m.load_state_dict(c['state']); vals.append(tr.esr(m, B, A, f0, mask)); pars = c['parameters']
            if vals: row[arm] = dict(test=vals, mean=float(np.mean(vals)), parameters=pars)
        res[tone] = row; print(tone, json.dumps(row), flush=True)
    (HERE / 'results.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    dict(extract=lambda: extract(sys.argv[2]), fit=lambda: fit(*sys.argv[2:]), test=test)[sys.argv[1]]()
