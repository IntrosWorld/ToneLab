"""Piano, third per-note version (after 'low notes sound bad').  Differences from piano_decay.py:
  * every band is trained toward the real notes (the engine's very strong fundamentals are no longer protected)
  * decay-rate correction limited to >= -0.5 /s, so the model can hardly make a partial grow over time
  * no manual bass shelf afterwards.          python piano_low.py fit | check
"""
import sys, os
sys.dont_write_bytecode = True
import json, time
import numpy as np
import torch
import piano_clean as pc, piano_decay as pd
from piano_clean import HERE, N, BAND, bands


class LowFlow(pd.DecayFlow):
    def forward(self, fr, midi, vel, age):
        x = self.features(midi, vel); G = self.gain(x)[..., 0].clamp(-4.5, 4.5) - self.rate(x)[..., 0].clamp(-.5, 6) * age[:, None]
        return torch.fft.irfft(torch.fft.rfft(fr) * BAND * torch.exp(G.clamp(-12, 4.5)), N)


CKPT = os.environ.get('PIANO_LOW_CKPT', 'piano_low')


def load():
    m = LowFlow(); m.load_state_dict(torch.load(HERE / f'checkpoints/{CKPT}.pt')['state']); return m.eval()


def fit(steps=2500, seed=1):
    torch.set_num_threads(8); torch.manual_seed(seed); E, R, mi, ve, ag = pd.frames('fit'); held = pd.frames('held'); model = LowFlow()
    opt = torch.optim.Adam(model.parameters(), 3e-3); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps, pct_start=.1); g = torch.Generator().manual_seed(seed); best = (9., None, 0); clock = time.time()
    def score():
        with torch.no_grad(): return float(np.mean([float((bands(model(held[0][i:i + 256], held[2][i:i + 256], held[3][i:i + 256], held[4][i:i + 256])) - bands(held[1][i:i + 256])).abs().mean()) for i in range(0, len(held[0]), 256)]))
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(E), (128,), generator=g); err = (bands(model(E[idx], mi[idx], ve[idx], ag[idx])) - bands(R[idx])).abs().mean(-1); wgt = 1 / (.08 + ag[idx]) if CKPT != 'piano_low' else torch.ones_like(err); loss = (err * wgt).sum() / wgt.sum()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step(); sched.step()
        if step % 250 == 0 or step == steps:
            v = score()
            if v < best[0]: best = (v, {k: t.clone() for k, t in model.state_dict().items()}, step)
            print(f'low step {step} loss {loss.item():.4f} held {v:.4f} best {best[0]:.4f} {time.time() - clock:.0f}s', flush=True)
    torch.save(dict(state=best[1], held=best[0]), HERE / f'checkpoints/{CKPT}.pt'); (HERE / f'{CKPT}_result.json').write_text(json.dumps(dict(held_error_all_bands_vs_real=best[0], best_step=best[2]), indent=1))


def check():
    d = np.load(HERE / 'data/piano_pairs.npz'); E, R, M, D = d['E'], d['R'], d['midi'], d['dyn']; sr = 44100; m = load(); edges = [25, 60, 120, 250, 500, 1000, 2000, 4000, 10000]; f = np.fft.rfftfreq(sr, 1 / sr)
    W = int(.3 * sr); f = np.fft.rfftfreq(W, 1 / sr); prof = lambda x, t0: ' '.join(f'{10 * np.log10((np.abs(np.fft.rfft(x[int(t0 * sr):int(t0 * sr) + W] * np.hanning(W))) ** 2)[(f >= a) & (f < b)].sum() + 1e-14):6.1f}' for a, b in zip(edges[:-1], edges[1:]))
    print('absolute band level dB    ', ' '.join(f'{a:>6}' for a in edges[:-1]))
    for p in [33, 36, 39, 45, 48]:
        i = [k for k in range(len(M)) if M[k] == p and D[k] == 'ff'][0]; y = pd.process(m, E[i].astype(np.float64), p, 112)
        for t0 in (0., .3): print(f'midi {p} t={t0}  engine', prof(E[i], t0)); print(f'              model ', prof(y, t0)); print('              REAL  ', prof(R[i], t0))


if __name__ == '__main__':
    dict(fit=fit, check=check)[sys.argv[1]]()
