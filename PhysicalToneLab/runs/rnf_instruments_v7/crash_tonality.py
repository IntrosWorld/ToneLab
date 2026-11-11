"""How 'metallic' (ringing partials) vs 'noisy' is each crash?  Real 17 inch hits vs tone_lab source vs source + flow.
Per band, in windows after the hit: spectral crest (peak / mean power, dB), spectral flatness (dB, 0 = noise), and the
share of band energy held by the 10 strongest 3 Hz-wide peaks.  Also the frame-to-frame envelope fluctuation.
"""
import sys
sys.dont_write_bytecode = True
import json
import numpy as np
import crash_flow as cf

SR = cf.SR; EDGES = [300, 600, 1200, 2500, 5000, 10000, 16000]; WINS = ((.05, .55), (.5, 1.5), (1.5, 2.8))


def stats(x):
    rows = []
    for a, b in WINS:
        seg = x[int(a * SR):int(b * SR)]; S = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2 + 1e-20; f = np.fft.rfftfreq(len(seg), 1 / SR); row = []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            P = S[(f >= lo) & (f < hi)]; k = max(1, int(round(3 * len(seg) / SR))); Q = np.convolve(P, np.ones(k), 'same'); top = np.sort(Q)[::-1]
            picked, used = 0., np.zeros(len(Q), bool)
            for i in np.argsort(Q)[::-1]:
                if used[max(0, i - k):i + k].any(): continue
                picked += Q[i]; used[i] = True
                if used.sum() >= 10: break
            row.append([10 * np.log10(P.max() / P.mean()), 10 * np.log10(np.exp(np.log(P).mean()) / P.mean()), 100 * picked / P.sum()])
        rows.append(row)
    return np.array(rows)


if __name__ == '__main__':
    d = np.load(cf.PAIRS); m = cf.load(); idx = np.where(d['size'] == 17)[0]; out = dict(bands_from_hz=EDGES[:-1], windows_s=WINS)
    sets = dict(real=[d['R'][i].astype(float) for i in idx], source=[d['E'][i].astype(float) for i in idx], model=[cf.process(m, d['E'][i].astype(float)) for i in idx])
    for n, xs in sets.items():
        s = np.mean([stats(x) for x in xs], 0); out[n] = np.round(s, 1).tolist()
        for w, r in zip(WINS, s): print(n, w, 'crest dB', np.round(r[:, 0]).astype(int), 'flatness dB', np.round(r[:, 1]).astype(int), 'top10 %', np.round(r[:, 2]).astype(int))
    (cf.HERE / 'crash_tonality.json').write_text(json.dumps(out, indent=1))
