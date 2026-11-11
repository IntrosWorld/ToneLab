"""Paired complex harmonic envelopes from the existing clean/amped pairs.  Reads originals, writes only here.

Each kept frame: clean guitar is a single steady pitch (harmonic residual small).  For it we store
B_k (clean) and A_k (amped) = complex amplitude of harmonic k, phase referenced to the frame centre.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
import numpy as np

HERE = Path(__file__).resolve().parent
PAIRS = HERE.parents[2] / 'NeuralInstrumentStudio_Experiments/energy_v3/data/captured_amp'
SR, WIN, HOP, H, TOP = 44100, 4096, 1024, 40, 8000.
W = np.hanning(WIN + 1)[:-1]; N = np.arange(WIN) - WIN / 2


def yin(frame):
    lo, hi = int(SR / 1400), int(SR / 60)
    f = np.fft.rfft(frame, 2 * WIN); ac = np.fft.irfft(f * f.conj())[:hi + 2]
    c = np.cumsum(frame ** 2); e = c[-1]; tau = np.arange(hi + 2)
    d = e - np.concatenate([[0], c[:-1]])[tau] * 0 + (c[WIN - 1 - tau]) - 2 * ac      # energy of both shifted parts
    d = (c[WIN - 1 - tau]) + (e - np.where(tau > 0, c[tau - 1], 0.)) - 2 * ac
    cm = d[1:] * np.arange(1, hi + 2) / np.maximum(np.cumsum(d[1:]), 1e-12); cm = np.concatenate([[1.], cm])
    below = np.where(cm[lo:hi] < .12)[0]
    if len(below) == 0: return None
    t = below[0] + lo
    while t + 1 < hi and cm[t + 1] < cm[t]: t += 1
    a, b, c2 = cm[t - 1], cm[t], cm[t + 1]; den = a - 2 * b + c2
    return SR / (t + (.5 * (a - c2) / den if abs(den) > 1e-12 else 0.))


def harmonics(frames, f0, k):
    """frames [F, WIN], f0 [F] -> complex amplitudes [F, k] (cos amplitude convention)."""
    out = []
    for i in range(0, len(f0), 64):
        ph = 2 * np.pi * f0[i:i + 64, None, None] * np.arange(1, k + 1)[None, :, None] * N[None, None] / SR
        out.append(2 * np.einsum('fn,fkn->fk', frames[i:i + 64] * W, np.exp(-1j * ph)) / W.sum())
    return np.concatenate(out)


def resynth(C, f0):
    out = []
    for i in range(0, len(f0), 64):
        ph = 2 * np.pi * f0[i:i + 64, None, None] * np.arange(1, C.shape[1] + 1)[None, :, None] * N[None, None] / SR
        out.append(np.einsum('fk,fkn->fn', C[i:i + 64], np.exp(1j * ph)).real)
    return np.concatenate(out)


def refine(frame, f0):
    grid = f0 * (1 + np.linspace(-.012, .012, 25)); k = int(min(H, TOP // f0, 12))
    amp = np.abs(harmonics(np.repeat(frame[None], len(grid), 0), grid, k)); w = np.arange(1, k + 1)
    return grid[np.argmax((amp ** 2 * w).sum(1))]


def one(args):
    fi, row = args
    d = np.load(PAIRS / row['file']); x, y = d['x'].astype(np.float64), d['y'].astype(np.float64)
    starts = np.arange(0, len(x) - WIN, HOP); keep = []
    for s in starts:
        fr = x[s:s + WIN]
        if np.sqrt(np.mean(fr ** 2)) < .004: continue
        f0 = yin(fr * W)
        if f0 is None or not 60 < f0 < 1400: continue
        keep.append((s, refine(fr, f0)))
    if not keep: return None
    S = np.array([k[0] for k in keep]); F = np.array([k[1] for k in keep])
    X = np.stack([x[s:s + WIN] for s in S]); Y = np.stack([y[s:s + WIN] for s in S])
    B = harmonics(X, F, H); A = harmonics(Y, F, H); K = np.minimum(H, np.floor(TOP / F)).astype(int); mask = np.arange(1, H + 1)[None] <= K[:, None]
    B = B * mask; A = A * mask
    resid = ((X - resynth(B, F)) ** 2 * W ** 2).sum(1) / np.maximum((X ** 2 * W ** 2).sum(1), 1e-12); ok = resid < .03
    residy = ((Y - resynth(A, F)) ** 2 * W ** 2).sum(1) / np.maximum((Y ** 2 * W ** 2).sum(1), 1e-12)
    stat = dict(file=row['id'], split=row['split'], frames=len(starts), pitched=len(S), kept=int(ok.sum()), target_harmonic_residual_median=float(np.median(residy[ok])) if ok.any() else None)
    return stat, dict(B=B[ok], A=A[ok], f0=F[ok], K=K[ok], t=S[ok] / SR, fid=np.full(ok.sum(), fi), split=np.full(ok.sum(), dict(train=0, validation=1, test=2)[row['split']]))


def main():
    from multiprocessing import Pool
    manifest = json.loads((PAIRS / 'manifest.json').read_text()); jobs = [(fi, row) for fi, row in enumerate(manifest['pairs']) if row['id'] != 'silence']
    with Pool(6) as pool: res = [r for r in pool.map(one, jobs) if r]
    stats = [r[0] for r in res]
    for s in stats: print(s, flush=True)
    np.savez(str(HERE / 'envelopes.npz'), **{k: np.concatenate([r[1][k] for r in res]) for k in res[0][1]})
    (HERE / 'extract_report.json').write_text(json.dumps(stats, indent=1))
    sp = np.concatenate([r[1]['split'] for r in res]); print('frames train/val/test', [int((sp == i).sum()) for i in range(3)])


if __name__ == '__main__':
    main()
