"""Crash cymbal: per-hit flow toward real crash cymbals (after the user identified the crash as the white-noise sound).

Measured: the drum_set_fea crash is almost only 8-16 kHz (1 kHz is 16 dB down, 200 Hz 42 dB down) and stays that way for
1.5 s; real crashes are roughly flat from 200 Hz to 8 kHz and lose their top end quickly (8-16 kHz: -11 dB at the hit,
-52 dB after 1.5 s).  A gain model cannot create the missing body from 32 modes, so the SOURCE is the project's other
physical crash (tone_lab AcousticKit: 512 plate modes 440 Hz-18 kHz, read only), and the flow is trained from it to:
Real data: University of Iowa Musical Instrument Samples (free, unrestricted), suspended crash cymbals 13/17/18/20 inch,
stick, normal and shoulder strikes, mf and ff (fetch_crash.py).  The 17 inch cymbal is held out.
Model: log-gain = g(f) - rate(f) * age + h(f, age); phase-preserving, cannot create frequencies.
  python crash_flow.py prep | fit | check | sample
"""
import sys
sys.dont_write_bytecode = True
import json, time
import numpy as np
import soundfile as sf
import torch
from torch import nn
from scipy.signal import resample_poly
import drum_flow as df

HERE = df.HERE; ENG = HERE.parents[2] / 'electric_guitar_lead/distortion_guitar_fea'; PAIRS = HERE / 'data/crash_pairs.npz'; CK = HERE / 'checkpoints/crash_flow.pt'
SR, N, HOP, SECONDS = 44100, 2048, 512, 3.0; FREQ, BAND, bands, CENTRE = df.FREQ, df.BAND, df.bands, df.CENTRE


def kit():
    sys.path.insert(0, str(ENG)); from tone_lab.drums import model as tm
    return tm, tm.AcousticKit()


def engine_hit(tm, k, velocity=108, seed=1):
    x = np.asarray(k.hit(tm.DrumHit(0., 'crash', int(velocity)), np.random.default_rng(seed)), dtype=np.float64); x = x.mean(1) if x.ndim == 2 else x; return resample_poly(x, 147, 160)       # 48 kHz -> 44.1 kHz


def cut(x, L):
    o = max(0, int(np.argmax(np.abs(x) > .1 * np.abs(x).max())) - 64); x = x[o:o + L]; return np.pad(x, (0, L - len(x)))


class CrashFlow(nn.Module):
    def __init__(self):
        super().__init__(); self.gain, self.rate, self.tf = df.mlp(1, 32, 1), df.mlp(1, 32, 1), df.mlp(2, 48, 1)
        for net in (self.gain, self.rate, self.tf): nn.init.zeros_(net[-1].weight); nn.init.zeros_(net[-1].bias)

    def forward(self, fr, age):
        u = torch.log(FREQ.clamp(df.LO, df.TOP) / 1000.)[:, None]; G = self.gain(u)[None, :, 0].clamp(-8, 3.5) - self.rate(u)[None, :, 0].clamp(-2, 20) * age[:, None]
        G = G + self.tf(torch.stack([u[None, :, 0].expand(len(fr), -1), torch.sqrt(age)[:, None].expand(-1, len(FREQ))], -1))[..., 0].clamp(-6, 3)
        G = torch.where(FREQ[None] < 150., G.clamp(max=0.), G); return torch.fft.irfft(torch.fft.rfft(fr) * BAND * torch.exp(G.clamp(-14, 3.5)), N)          # never more than +30 dB


def prep():
    tm, k = kit(); L = int(SECONDS * SR); E, R, S = [], [], []; n = int(.1 * SR)
    for i, f in enumerate(sorted((HERE / 'data/crash').glob('*crash.stick.*.aif'))):
        if '.pp.' in f.name: continue                                        # pp recordings peak at 0.002: mostly room/recorder noise
        x, sr = sf.read(f, dtype='float64', always_2d=True); assert sr == SR; r = cut(x.mean(1), L); e = cut(engine_hit(tm, k, 108, seed=i), L); r *= np.sqrt((e[:n] ** 2).mean() / (r[:n] ** 2).mean())
        E.append(e); R.append(r); S.append(int(f.name[:2]))
    E, R = np.stack(E), np.stack(R); s = .1 / np.sqrt(np.mean(E ** 2)); np.savez(PAIRS, E=(E * s).astype(np.float32), R=(R * s).astype(np.float32), size=np.array(S), scale=s); print('hits', len(S), 'sizes', sorted(set(S)), 'held out: 17 inch,', S.count(17), 'hits')


def frames(held):
    d = np.load(PAIRS); out = [[] for _ in range(3)]
    for i in np.where((d['size'] == 17) == held)[0]:
        for s in range(-N // 2, int(SECONDS * SR) - N, HOP):
            a = max(0, s); e = np.zeros(N); r = np.zeros(N); e[a - s:] = d['E'][i, a:s + N]; r[a - s:] = d['R'][i, a:s + N]
            if max(np.abs(e).max(), np.abs(r).max()) > 1e-3:
                for o, v in zip(out, (e, r, max(0., (s + N / 2) / SR))): o.append(v)
    return torch.from_numpy(np.stack(out[0])).float(), torch.from_numpy(np.stack(out[1])).float(), torch.tensor(out[2]).float()


def loss_of(mdl, e, r, a):
    tg = bands(r); ob = bands(mdl(e, a)); top = torch.maximum(tg, ob.detach()); bw = ((top - tg.max(-1, keepdim=True).values) / 5.8 + 1).clamp(0, 1) * (CENTRE[None] >= 150.); return ((ob - tg).abs() * bw).sum(-1) / bw.sum(-1)


def fit(steps=3000, seed=1):
    torch.set_num_threads(6); torch.manual_seed(seed); E, R, A = frames(False); held = frames(True); model = CrashFlow(); g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), 3e-3); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps, pct_start=.1); best = (9., None, 0); clock = time.time()
    def score(mdl):
        with torch.no_grad(): return float(torch.cat([loss_of(mdl, *[t[i:i + 512] for t in held]) for i in range(0, len(held[0]), 512)]).mean())
    start = score(lambda fr, a: torch.fft.irfft(torch.fft.rfft(fr) * BAND, N)); print('fit frames', len(E), 'held frames', len(held[0]), 'engine unprocessed, held:', round(start, 4), flush=True)
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(E), (256,), generator=g); err = loss_of(model, E[idx], R[idx], A[idx]); w = 1 / (.3 + A[idx]); loss = (err * w).sum() / w.sum(); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step(); sched.step()
        if step % 300 == 0 or step == steps:
            v = score(model)
            if v < best[0]: best = (v, {k_: t.clone() for k_, t in model.state_dict().items()}, step)
            print(f'crash step {step} loss {loss.item():.4f} held {v:.4f} best {best[0]:.4f} {time.time() - clock:.0f}s', flush=True)
    torch.save(dict(state=best[1], held=best[0]), CK); (HERE / 'crash_flow_result.json').write_text(json.dumps(dict(engine_unprocessed_held=start, held_error=best[0], best_step=best[2]), indent=1))


def load():
    m = CrashFlow(); m.load_state_dict(torch.load(CK)['state']); return m.eval()


@torch.no_grad()
def process(model, x):
    x = np.concatenate([np.zeros(N), x, np.zeros(2 * N)]); starts = np.arange(0, len(x) - N, HOP); win = np.hanning(N + 1)[:-1]; y = np.zeros(len(x))
    fr = torch.from_numpy(np.stack([x[s:s + N] for s in starts])).float(); age = torch.from_numpy(np.clip((starts + N / 2 - N) / SR, 0, SECONDS)).float(); out = model(fr, age).numpy() * win
    for s, o in zip(starts, out): y[s:s + N] += o
    return y[N:-2 * N] / 2.


def check():
    d = np.load(PAIRS); m = load(); idx = np.where(d['size'] == 17)[0]; edges = [200, 500, 1000, 2000, 4000, 8000, 16000]; T = (0., .1, .3, .6, 1.0, 1.5, 2.2); out = dict(bands_from_hz=edges[:-1], seconds=T, held_hits=len(idx))
    def prof(x):
        o = []
        for t in T:
            seg = x[int(t * SR):int(t * SR) + 4096]; S = np.abs(np.fft.rfft(seg * np.hanning(4096))) ** 2; f = np.fft.rfftfreq(4096, 1 / SR); o.append([10 * np.log10(S[(f >= a) & (f < b)].sum() + 1e-12) for a, b in zip(edges[:-1], edges[1:])])
        return np.array(o)
    pe = np.mean([prof(d['E'][i]) for i in idx], 0); pr = np.mean([prof(d['R'][i]) for i in idx], 0); pm = np.mean([prof(process(m, d['E'][i].astype(np.float64))) for i in idx], 0)
    out['source_minus_real_db'] = np.round(pe - pr).astype(int).tolist(); out['model_minus_real_db'] = np.round(pm - pr).astype(int).tolist(); out['source_mean_abs'] = round(float(np.abs(pe - pr).mean()), 1); out['model_mean_abs'] = round(float(np.abs(pm - pr).mean()), 1)
    for k_ in ('source_minus_real_db', 'model_minus_real_db'): print(k_); print(np.array(out[k_]))
    print('mean abs dB: source', out['source_mean_abs'], 'model', out['model_mean_abs']); (HERE / 'crash_summary.json').write_text(json.dumps(out, indent=1))


def sample():
    """Four hits each: the drum_set_fea crash (used so far), the tone_lab crash (new source), and the new source through the flow."""
    tm, k = kit(); dm, model = df.engine(); m = load(); s = float(np.load(PAIRS)['scale']); out = HERE.parent / 'rnf_song_ascend/drum_samples'; out.mkdir(parents=True, exist_ok=True)
    old = resample_poly(dm.synth_hit(dm.DrumHit(0., 'crash', 108, 0.), model, np.random.default_rng(1)), 147, 160); new = engine_hit(tm, k); flow = process(m, new * s) / s
    for name, x in (('4_crash_old_physical_model', old), ('4b_crash_new_physical_model', new), ('4c_crash_new_model_plus_learned_stage', flow)):
        y = np.zeros(int(11.5 * SR))
        for j in range(4): a = int((.2 + 2.5 * j) * SR); n = min(len(x), len(y) - a); y[a:a + n] += x[:n]
        sf.write(out / f'{name}.flac', y * (.9 / np.abs(y).max()), SR, subtype='PCM_16'); S = np.abs(np.fft.rfft(x)) ** 2; f = np.fft.rfftfreq(len(x), 1 / SR); print(name, 'share of energy above 8 kHz: %.0f %%, below 2 kHz: %.0f %%' % (100 * S[f > 8000].sum() / S.sum(), 100 * S[f < 2000].sum() / S.sum()))


if __name__ == '__main__':
    dict(prep=prep, fit=fit, check=check, sample=sample)[sys.argv[1]]()
