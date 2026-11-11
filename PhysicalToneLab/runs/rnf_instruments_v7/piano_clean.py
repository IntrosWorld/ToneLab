"""Piano, second design: the flow acts on EACH NOTE before mixing, knows the note's pitch and velocity, and
contains only phase-preserving terms (it rescales what the engine produced; it cannot create new frequencies).

  terms per step:  linear gain g(f, partial number, pitch, velocity)
                   level-dependent damping        1 / (1 + h * e1)          (louder -> more loss)
                   saturable loss                 exp(-h * a / (1 + e2))    (quieter -> more loss: changes decay)
Below 150 Hz the training target is the engine itself (keep its low end); above, the real Iowa notes.

  python piano_clean.py fit <arm>     # arm = flow | eq_only
  python piano_clean.py render
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json, time
import numpy as np
import soundfile as sf
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
SR, N, HOP, LO, TOP, KEEP, STEPS = 44100, 4096, 1024, 25., 10000., 150., 4
FREQ = torch.fft.rfftfreq(N, 1 / SR); BAND = ((FREQ >= LO) & (FREQ <= TOP)).float(); WIN = torch.hann_window(N)


def mlp(i, h, o): return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, o))


class CleanFlow(nn.Module):
    def __init__(self, eq_only=False, rank=3):
        super().__init__(); self.eq_only = eq_only; self.gain = mlp(4, 32, 1); self.ka, self.kb, self.sa, self.sb = mlp(4, 16, rank), mlp(4, 16, rank), mlp(4, 16, rank), mlp(4, 16, rank)
        self.log_h = nn.Parameter(torch.full((STEPS,), -1.4))
        for net, bias in ((self.gain, 0.), (self.ka, -4.), (self.sa, -4.)): nn.init.zeros_(net[-1].weight); nn.init.constant_(net[-1].bias, bias)

    def features(self, midi, vel):
        f0 = 440. * 2 ** ((midi - 69) / 12); f = FREQ.clamp(LO, TOP)[None]
        return torch.stack([torch.log(f / 1000.).expand(len(midi), -1), torch.log(f / f0[:, None]).clamp(-2, 5), ((midi - 60) / 24)[:, None].expand(-1, len(FREQ)), ((vel - 94) / 18)[:, None].expand(-1, len(FREQ))], -1)

    def forward(self, fr, midi, vel):
        x = self.features(midi, vel); Z = torch.fft.rfft(fr) * BAND; G = self.gain(x)[..., 0].clamp(-4.5, 4.5)
        if not self.eq_only:
            sp = nn.functional.softplus; ka, kb, sa, sb = sp(self.ka(x)), sp(self.kb(x)), sp(self.sa(x)), sp(self.sb(x)); mag2 = (2 * Z.abs() / N).square() * BAND
            for s in range(STEPS):
                h = self.log_h[s].exp(); p = mag2 * torch.exp(2 * G)
                e1 = torch.einsum('fkr,fjr,fj->fk', ka, kb, p); e2 = torch.einsum('fjr,fj->f', sb, p)[:, None] * 1e3
                G = G - torch.log1p(h * e1) - h * sa.sum(-1) / (1 + e2)
        return torch.fft.irfft(Z * torch.exp(G), N)


def mel_bank():
    f = FREQ.numpy(); mel = lambda h: 2595 * np.log10(1 + h / 700); edges = 700 * (10 ** (np.linspace(mel(LO), mel(TOP), 98) / 2595) - 1); W = np.zeros((96, len(f)))
    for i in range(96): W[i] = np.clip(np.minimum((f - edges[i]) / (edges[i + 1] - edges[i]), (edges[i + 2] - f) / (edges[i + 2] - edges[i + 1])), 0, None)
    return torch.from_numpy(W).float(), torch.from_numpy(edges[1:-1] < KEEP)


MEL, LOW = mel_bank()
def bands(fr): return torch.log((torch.fft.rfft(fr * WIN).abs().square() / N) @ MEL.T + 1e-8)
def target(E, R): return torch.where(LOW, bands(E), bands(R))                 # engine low end, real above


def frames(split):
    d = np.load(HERE / 'data/piano_pairs.npz'); AE, AR, M, D = d['E'], d['R'], d['midi'], d['dyn']; pitches = sorted(set(M.tolist())); held = set(pitches[2::4]); E, R, mi, ve = [], [], [], []
    for i in [i for i, p in enumerate(M) if (p in held) == (split == 'held')]:
        for s in range(0, AE.shape[1] - N, HOP):
            if max(np.abs(AE[i, s:s + N]).max(), np.abs(AR[i, s:s + N]).max()) > 3e-3: E.append(AE[i, s:s + N]); R.append(AR[i, s:s + N]); mi.append(M[i]); ve.append(112. if D[i] == 'ff' else 76.)
    return torch.from_numpy(np.stack(E)), torch.from_numpy(np.stack(R)), torch.tensor(mi).float(), torch.tensor(ve).float()


def score(model, E, R, mi, ve):
    with torch.no_grad(): return float(np.mean([float((bands(model(E[i:i + 256], mi[i:i + 256], ve[i:i + 256])) - target(E[i:i + 256], R[i:i + 256])).abs().mean()) for i in range(0, len(E), 256)]))


def fit(arm, steps=2000, seed=1):
    torch.set_num_threads(8); torch.manual_seed(seed); E, R, mi, ve = frames('fit'); held = frames('held'); model = CleanFlow(arm == 'eq_only')
    opt = torch.optim.Adam(model.parameters(), 3e-3); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps, pct_start=.1); g = torch.Generator().manual_seed(seed)
    ident = lambda fr, m, v: torch.fft.irfft(torch.fft.rfft(fr) * BAND, N); start = score(ident, *held); best = (9., None, 0); clock = time.time()
    print(arm, 'fit frames', len(E), 'held frames', len(held[0]), 'engine unprocessed, held:', round(start, 4), flush=True)
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(E), (128,), generator=g); loss = (bands(model(E[idx], mi[idx], ve[idx])) - target(E[idx], R[idx])).abs().mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step(); sched.step()
        if step % 200 == 0 or step == steps:
            v = score(model, *held)
            if v < best[0]: best = (v, {k: t.clone() for k, t in model.state_dict().items()}, step)
            print(f'{arm} step {step} loss {loss.item():.4f} held {v:.4f} best {best[0]:.4f} {time.time() - clock:.0f}s', flush=True)
    (HERE / 'checkpoints').mkdir(exist_ok=True); torch.save(dict(state=best[1], held=best[0], engine_unprocessed_held=start), HERE / f'checkpoints/piano_clean_{arm}.pt')
    (HERE / f'piano_clean_{arm}_result.json').write_text(json.dumps(dict(arm=arm, engine_unprocessed_held=start, held_error=best[0], best_step=best[2], parameters=sum(p.numel() for p in model.parameters())), indent=1))


@torch.no_grad()
def process(model, x, midi, vel):
    x = np.concatenate([np.zeros(N), x, np.zeros(2 * N)]); starts = np.arange(0, len(x) - N, HOP); y = np.zeros(len(x)); win = np.hanning(N + 1)[:-1]
    for b in range(0, len(starts), 256):
        st = starts[b:b + 256]; n = len(st); out = model(torch.from_numpy(np.stack([x[s:s + N] for s in st])).float(), torch.full((n,), float(midi)), torch.full((n,), float(vel))).numpy() * win
        for s, o in zip(st, out): y[s:s + N] += o
    return y[N:-2 * N] / 2.


def load(arm):
    c = torch.load(HERE / f'checkpoints/piano_clean_{arm}.pt'); m = CleanFlow(arm == 'eq_only'); m.load_state_dict(c['state']); return m.eval()


def perform(events, arm='flow', tail=3.):
    """Each note is rendered alone by the engine (dry, mono), passed through the flow, then mixed."""
    sys.path.insert(0, str(BASE / 'grand_piano_improved/tools')); import render_model as rm
    d = np.load(HERE / 'data/piano_pairs.npz'); se, sr_ = float(d['engine_scale']), float(d['real_scale']); model = load(arm) if arm else None
    piano = rm.GrandPiano(sample_rate=SR, room_mix=0., normalize=False); total = int((max(e[3] for e in events) + tail + 1) * SR); mix = np.zeros(total); cache = {}
    for pitch, vel, start, release in events:
        key = (pitch, vel, round(release - start, 3))
        if key not in cache:
            x = piano.render([(pitch, vel, 0., release - start)], tail=tail).mean(1); cache[key] = process(model, x * se, pitch, vel) / sr_ if model else x
        y = cache[key]; a = int(start * SR); n = min(len(y), total - a); mix[a:a + n] += y[:n]
    return mix


def render():
    sys.path.insert(0, str(BASE / 'grand_piano_improved/tools')); import render_model as rm
    ev = rm.demo_events(); out = HERE / 'audition'; info = {}; edges = [20, 60, 120, 250, 500, 1000, 2000, 4000, 10000]
    for name, arm in (('piano_5_engine_notes_mixed', None), ('piano_6_clean_flow_per_note', 'flow'), ('piano_7_eq_only_per_note', 'eq_only')):
        if arm and not (HERE / f'checkpoints/piano_clean_{arm}.pt').exists(): continue
        y = perform(ev, arm); sf.write(out / f'{name}.wav', y * (.7 / np.abs(y).max()), SR, subtype='PCM_24')
        S = np.abs(np.fft.rfft(y)) ** 2; f = np.fft.rfftfreq(len(y), 1 / SR); e = np.array([S[(f >= a) & (f < b)].sum() for a, b in zip(edges[:-1], edges[1:])])
        seg = y[int(10.2 * SR):int(14 * SR)]; C = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2; fc = np.fft.rfftfreq(len(seg), 1 / SR); C = np.sort(C[(fc > 60) & (fc < 10000)])[::-1]
        info[name] = dict(band_db={f'{a}-{b}': round(float(v), 1) for a, b, v in zip(edges[:-1], edges[1:], 10 * np.log10(e / e.sum() + 1e-12))}, chord_energy_between_partials=round(float(C[int(.02 * len(C)):].sum() / C.sum()), 4), nan=bool(np.isnan(y).any()))
    (out / 'piano_clean_notes.json').write_text(json.dumps(info, indent=1)); print(json.dumps(info, indent=1))


if __name__ == '__main__':
    dict(render=render, fit=lambda: fit(sys.argv[2]))[sys.argv[1]]()
