"""Piano, per-note flow with a damping correction: log-gain = g(f, partial, pitch, velocity) - rate(...) * note age.

rate is a change of decay rate per partial (the modal damping correction of the original cell).  Phase is untouched,
so no new frequencies can appear.  Low end (< 150 Hz) is trained to stay as the engine.   fit | render
"""
import sys
sys.dont_write_bytecode = True
import json, time
import numpy as np
import soundfile as sf
import torch
from torch import nn
import piano_clean as pc
from piano_clean import HERE, BASE, SR, N, HOP, BAND, bands, target


class DecayFlow(pc.CleanFlow):
    def __init__(self):
        super().__init__(eq_only=True); self.rate = pc.mlp(4, 32, 1); nn.init.zeros_(self.rate[-1].weight); nn.init.zeros_(self.rate[-1].bias)

    def forward(self, fr, midi, vel, age):
        x = self.features(midi, vel); G = self.gain(x)[..., 0].clamp(-4.5, 4.5) - (self.rate(x)[..., 0].clamp(-3, 6) * age[:, None]).clamp(-9, 9)
        return torch.fft.irfft(torch.fft.rfft(fr) * BAND * torch.exp(G), N)


def frames(split):
    d = np.load(HERE / 'data/piano_pairs.npz'); AE, AR, M, D = d['E'], d['R'], d['midi'], d['dyn']; pitches = sorted(set(M.tolist())); held = set(pitches[2::4]); out = [[] for _ in range(5)]
    for i in [i for i, p in enumerate(M) if (p in held) == (split == 'held')]:
        for s in range(0, AE.shape[1] - N, HOP):
            if max(np.abs(AE[i, s:s + N]).max(), np.abs(AR[i, s:s + N]).max()) > 3e-3:
                for o, v in zip(out, (AE[i, s:s + N], AR[i, s:s + N], M[i], 112. if D[i] == 'ff' else 76., s / SR)): o.append(v)
    return torch.from_numpy(np.stack(out[0])), torch.from_numpy(np.stack(out[1])), torch.tensor(out[2]).float(), torch.tensor(out[3]).float(), torch.tensor(out[4]).float()


def score(model, E, R, mi, ve, ag):
    with torch.no_grad(): return float(np.mean([float((bands(model(E[i:i + 256], mi[i:i + 256], ve[i:i + 256], ag[i:i + 256])) - target(E[i:i + 256], R[i:i + 256])).abs().mean()) for i in range(0, len(E), 256)]))


def fit(steps=2500, seed=1):
    torch.set_num_threads(8); torch.manual_seed(seed); E, R, mi, ve, ag = frames('fit'); held = frames('held'); model = DecayFlow()
    opt = torch.optim.Adam(model.parameters(), 3e-3); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps, pct_start=.1); g = torch.Generator().manual_seed(seed); best = (9., None, 0); clock = time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(E), (128,), generator=g); loss = (bands(model(E[idx], mi[idx], ve[idx], ag[idx])) - target(E[idx], R[idx])).abs().mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step(); sched.step()
        if step % 250 == 0 or step == steps:
            v = score(model, *held)
            if v < best[0]: best = (v, {k: t.clone() for k, t in model.state_dict().items()}, step)
            print(f'decay step {step} loss {loss.item():.4f} held {v:.4f} best {best[0]:.4f} {time.time() - clock:.0f}s', flush=True)
    torch.save(dict(state=best[1], held=best[0]), HERE / 'checkpoints/piano_decay.pt'); (HERE / 'piano_decay_result.json').write_text(json.dumps(dict(held_error=best[0], best_step=best[2], parameters=sum(p.numel() for p in model.parameters())), indent=1))


def load():
    m = DecayFlow(); m.load_state_dict(torch.load(HERE / 'checkpoints/piano_decay.pt')['state']); return m.eval()


@torch.no_grad()
def process(model, x, midi, vel):
    x = np.concatenate([np.zeros(N), x, np.zeros(2 * N)]); starts = np.arange(0, len(x) - N, HOP); y = np.zeros(len(x)); win = np.hanning(N + 1)[:-1]
    for b in range(0, len(starts), 256):
        st = starts[b:b + 256]; n = len(st); age = torch.from_numpy(np.clip((st - N) / SR, 0, 3.)).float()          # trained on the first 3 s; held constant after
        out = model(torch.from_numpy(np.stack([x[s:s + N] for s in st])).float(), torch.full((n,), float(midi)), torch.full((n,), float(vel)), age).numpy() * win
        for s, o in zip(st, out): y[s:s + N] += o
    return y[N:-2 * N] / 2.


def render():
    sys.path.insert(0, str(BASE / 'grand_piano_improved/tools')); import render_model as rm
    d = np.load(HERE / 'data/piano_pairs.npz'); se, sr_ = float(d['engine_scale']), float(d['real_scale']); model = load(); E, R, M, D = d['E'], d['R'], d['midi'], d['dyn']
    piano = rm.GrandPiano(sample_rate=SR, room_mix=0., normalize=False); ev = rm.demo_events(); total = int((max(e[3] for e in ev) + 4) * SR); mix = np.zeros(total)
    for pitch, vel, start, release in ev:
        y = process(model, piano.render([(pitch, vel, 0., release - start)], tail=3.).mean(1) * se, pitch, vel) / sr_; a = int(start * SR); n = min(len(y), total - a); mix[a:a + n] += y[:n]
    sf.write(HERE / 'audition/piano_8_flow_with_decay_correction.wav', mix * (.7 / np.abs(mix).max()), SR, subtype='PCM_24')
    edges = [20, 60, 120, 250, 500, 1000, 2000, 4000, 10000]; S = np.abs(np.fft.rfft(mix)) ** 2; f = np.fft.rfftfreq(len(mix), 1 / SR); e = np.array([S[(f >= a) & (f < b)].sum() for a, b in zip(edges[:-1], edges[1:])])
    seg = mix[int(10.2 * SR):int(14 * SR)]; C = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2; fc = np.fft.rfftfreq(len(seg), 1 / SR); C = np.sort(C[(fc > 60) & (fc < 10000)])[::-1]
    print('band dB', dict(zip(edges[:-1], np.round(10 * np.log10(e / e.sum()), 1))), 'chord energy between partials', round(float(C[int(.02 * len(C)):].sum() / C.sum()), 4), 'nan', bool(np.isnan(mix).any()))
    drop = lambda x: 20 * np.log10(np.sqrt((x[int(2 * SR):int(2.9 * SR)] ** 2).mean()) / np.sqrt((x[int(.1 * SR):int(.5 * SR)] ** 2).mean()))
    def bright(x): S = np.abs(np.fft.rfft(x[:SR])) ** 2; f = np.fft.rfftfreq(SR, 1 / SR); return 10 * np.log10(S[(f > 500) & (f < 4000)].sum() / S[(f > 25) & (f < 500)].sum() + 1e-12)
    for i in range(len(M)):
        if M[i] in sorted(set(M.tolist()))[2::4] and D[i] == 'ff':
            o = process(model, E[i].astype(np.float64), M[i], 112); print(f'held midi {M[i]}: level change over 2 s  engine {drop(E[i]):6.1f}  flow {drop(o):6.1f}  REAL {drop(R[i]):6.1f} dB | brightness  engine {bright(E[i]):6.1f}  flow {bright(o):6.1f}  REAL {bright(R[i]):6.1f} dB')


if __name__ == '__main__':
    dict(fit=fit, render=render)[sys.argv[1]]()
