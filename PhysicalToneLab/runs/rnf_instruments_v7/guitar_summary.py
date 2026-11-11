"""Held-out summary for the guitar model: attack-band error and decay, engine vs model vs real."""
import sys
sys.dont_write_bytecode = True
import json
import numpy as np
import guitar_flow as gf, piano_decay as pd
E, R, M, V = gf.matched(); d = np.load(gf.PAIRS); m = gf.load(); SR = 44100; W = int(.25 * SR); f = np.fft.rfftfreq(W, 1 / SR); edges = [80, 160, 320, 640, 1280, 2560, 5120, 10000]; held = set(sorted(set(M.tolist()))[2::4])
def prof(x): S = np.abs(np.fft.rfft(x[:W] * np.hanning(W))) ** 2; return np.array([10 * np.log10(S[(f >= a) & (f < b)].sum() + 1e-12) for a, b in zip(edges[:-1], edges[1:])])
de, dm = [], []
for i in [i for i in range(len(M)) if M[i] in held]:
    y = pd.process(m, E[i].astype(np.float64), int(M[i]), int(V[i])); pr = prof(R[i]); ok = np.array(edges[:-1]) >= .8 * 440 * 2 ** ((M[i] - 69) / 12); de.append(np.where(ok, prof(E[i]) - pr, np.nan)); dm.append(np.where(ok, prof(y) - pr, np.nan))
de, dm = np.array(de), np.array(dm); out = dict(bands_from_hz=edges[1:-1], engine_minus_real_mean=np.round(np.nanmean(de, 0), 1)[1:].tolist(), model_minus_real_mean=np.round(np.nanmean(dm, 0), 1)[1:].tolist(),
    engine_mean_abs=np.round(np.nanmean(np.abs(de), 0), 1)[1:].tolist(), model_mean_abs=np.round(np.nanmean(np.abs(dm), 0), 1)[1:].tolist(), engine_overall=round(float(np.nanmean(np.abs(de[:, 1:]))), 2), model_overall=round(float(np.nanmean(np.abs(dm[:, 1:]))), 2), held_notes=len(de))
print(json.dumps(out)); (gf.HERE / 'guitar_summary.json').write_text(json.dumps(out, indent=1))
