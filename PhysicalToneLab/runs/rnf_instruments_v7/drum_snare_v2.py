"""Snare, second training (after the user heard the snare as white noise in the Ascend render).

Measured on 20 held-out snare hits with drum_flow.pt: 5-12 kHz is 8-13 dB above the real snare for the first 150 ms and
1.5-5 kHz 4-9 dB above, while 150-400 Hz is 4 dB below.  Cause: each real hit was level-matched to the engine hit by total
energy, and the engine snare is almost all hiss (5-12 kHz 27 dB above real).  That made the target need +20 dB or more of
body, which hits the model's +20 dB boost limit, so the balance stayed too bright.
Change: real snare hits are level-matched to the engine hit in the body band only (150-1500 Hz, first 80 ms), so the model
has to CUT hiss (allowed down to -39 dB) instead of boosting body.  Same model class, same loss, snare hits only.
Kick and hi-hat keep the approved drum_flow.pt.  Writes to snare_v2/ and data/drum_pairs_snare_v2.npz only.
Attempt 1 (same model class, only the level matching changed) made NO difference (snare_v2/attempt1_no_change_summary.json):
hiss still 8-14 dB over real.  So the boost limit was not the cause.  The engine snare's hiss is ~47 dB too strong
relative to its body, and the model's cut limit was -39 dB (-4.5 nepers).  Attempt 2 (SnareFlow below): cut limit -69 dB.
Attempt 2 (cut limit -69 dB) ALSO made no difference (attempt2_no_change_summary.json): same held error 2.2287.
Checked in the training metric itself: hiss error vs real is +7 dB at the hit, ~0 at 30-80 ms, +5 at 80-160 ms, +9 after.
The real snare's hiss does not die away exponentially relative to the engine's, and the model only had gain - rate * age
(one straight line in dB per frequency).  Attempt 3: an extra small network of (frequency, time since the hit) added to
the log-gain, so the model can learn any decay shape.  Still phase-preserving, still cannot create frequencies.
  python drum_snare_v2.py prep | fit | check
"""
import sys
sys.dont_write_bytecode = True
import json
import numpy as np
import torch
import drum_flow as df

Orig = df.DrumFlow


class SnareFlow(Orig):
    def __init__(self):
        super().__init__(); self.tf = df.mlp(2, 48, 1); torch.nn.init.zeros_(self.tf[-1].weight); torch.nn.init.zeros_(self.tf[-1].bias)

    def forward(self, fr, piece, age):
        x = torch.cat([torch.log(df.FREQ.clamp(df.LO, df.TOP) / 1000.)[None, :, None].expand(len(fr), -1, -1), torch.nn.functional.one_hot(piece, 3).float()[:, None].expand(-1, len(df.FREQ), -1)], -1)
        G = self.gain(x)[..., 0].clamp(-8., 2.3) - self.rate(x)[..., 0].clamp(-3, 40) * age[:, None]
        G = G + self.tf(torch.stack([x[..., 0], torch.sqrt(4 * age)[:, None].expand(-1, len(df.FREQ))], -1))[..., 0].clamp(-6, 3)
        G = torch.where(df.FREQ[None] < df.FLOOR_HZ[piece][:, None], G.clamp(max=0.), G)
        return torch.fft.irfft(torch.fft.rfft(fr) * df.BAND * torch.exp(G.clamp(-14, 2.3)), df.N)


V7 = df.HERE; SUB = V7 / 'snare_v2'; NEW = V7 / 'data/drum_pairs_snare_v2.npz'; OLD = df.PAIRS; SR = df.SR


def body(x, n=int(.08 * 44100)):
    f = np.fft.rfftfreq(n, 1 / SR); S = np.abs(np.fft.rfft(x[:, :n] * np.hanning(n))) ** 2; return np.sqrt(S[:, (f >= 150) & (f < 1500)].sum(1))


def prep():
    d = np.load(OLD); i = d['piece'] == 1; E, R = d['E'][i].astype(np.float64), d['R'][i].astype(np.float64); k = body(E) / np.maximum(body(R), 1e-9); R = R * k[:, None]
    np.savez(NEW, E=E.astype(np.float32), R=R.astype(np.float32), piece=d['piece'][i], src=d['src'][i], length=d['length'][i], scale=d['scale']); print('snare hits', int(i.sum()), 'real level change dB median', round(float(np.median(20 * np.log10(k))), 1))


def use():
    df.PAIRS = NEW; df.HERE = SUB; SUB.mkdir(exist_ok=True); df.DrumFlow = SnareFlow


def load():
    m = SnareFlow(); m.load_state_dict(torch.load(SUB / 'checkpoints/drum_flow.pt')['state']); return m.eval()


def check():
    d = np.load(NEW); E, R = d['E'], d['R']; h = df.split(d); idx = [i for i in np.where(h)[0] if d['length'][i] > .35]; new = load(); old = Orig(); old.load_state_dict(torch.load(V7 / 'checkpoints/drum_flow.pt')['state']); old.eval(); out = dict(held_hits=len(idx), every_ms=40)
    def tprof(x, lo, hi):
        o = []
        for a in np.arange(0, .32, .04):
            seg = x[int(a * SR):int((a + .04) * SR)]; S = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 4096)) ** 2; f = np.fft.rfftfreq(4096, 1 / SR); o.append(10 * np.log10(S[(f >= lo) & (f < hi)].sum() + 1e-12))
        return np.array(o)
    Y = {n: [df.process(m, E[i].astype(np.float64), 'snare') for i in idx] for n, m in (('old', old), ('new', new))}
    ref = np.mean([tprof(R[i], 150, 1500) for i in idx], 0)[0]; refs = {n: np.mean([tprof(y, 150, 1500) for y in Y[n]], 0)[0] for n in Y}
    for lo, hi in ((150, 400), (400, 1500), (1500, 5000), (5000, 12000)):
        r = np.mean([tprof(R[i], lo, hi) for i in idx], 0) - ref; row = dict(real=np.round(r).astype(int).tolist())
        for n in Y: row[f'{n}_model_minus_real'] = np.round(np.mean([tprof(y, lo, hi) for y in Y[n]], 0) - refs[n] - r).astype(int).tolist()          # each relative to its own 150-1500 Hz onset level
        out[f'{lo}-{hi}Hz'] = row; print(lo, hi, json.dumps(row))
    (SUB / 'snare_v2_summary.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    if sys.argv[1] == 'prep': prep()
    elif sys.argv[1] == 'fit': use(); df.fit()
    else: check()
