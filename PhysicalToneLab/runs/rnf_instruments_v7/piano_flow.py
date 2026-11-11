"""Resonant flow for piano: physical-engine note -> real recorded note.   Reads engines/originals, writes only here.

  python piano_flow.py prep            # pair each real Iowa note with the engine's note of the same pitch and dynamic
  python piano_flow.py fit <arm>       # arm = resonant | eq_only
  python piano_flow.py render          # engine demo phrase through the trained flow

Real and engine notes are not the same waveform, so the loss compares band energies over the life of the note
(80 bands, 60 Hz - 10 kHz, every 11.6 ms), on pitches held out from training.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json, re, time
import numpy as np
import soundfile as sf
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / 'rnf_resonant_amp_v6'))
import models
SR, N, HOP, LO, TOP, SECONDS = 44100, 2048, 512, 60., 10000., 3.0
NAMES = dict(C=0, Db=1, D=2, Eb=3, E=4, F=5, Gb=6, G=7, Ab=8, A=9, Bb=10, B=11)


def onset(x):
    env = np.abs(x); return max(0, int(np.argmax(env > .1 * env.max())) - 256)


def prep():
    sys.path.insert(0, str(BASE / 'grand_piano_improved/tools')); import render_model as rm
    piano = rm.GrandPiano(sample_rate=SR, room_mix=0., normalize=False); L = int(SECONDS * SR); E, R, midi, dyn = [], [], [], []
    for f in sorted((HERE / 'data/piano').glob('Piano.*.aiff')):
        m = re.match(r'Piano\.(ff|mf)\.([A-G]b?)(\d)\.aiff', f.name)
        try: x, sr = sf.read(f, dtype='float64', always_2d=True)
        except Exception as e: print('unreadable', f.name, e); continue
        if not m or sr != SR: continue
        p = NAMES[m.group(2)] + 12 * (int(m.group(3)) + 1); x = x.mean(1); x = x[onset(x):][:L]; x = np.pad(x, (0, L - len(x)))
        e = piano.render([(p, 112 if m.group(1) == 'ff' else 76, 0., SECONDS)], tail=.3).mean(1); e = e[onset(e):][:L]; e = np.pad(e, (0, L - len(e)))
        E.append(e); R.append(x); midi.append(p); dyn.append(m.group(1)); print(f.name, p, 'real rms', round(float(np.sqrt(np.mean(x ** 2))), 4), 'engine rms', round(float(np.sqrt(np.mean(e ** 2))), 4), flush=True)
    E, R = np.stack(E), np.stack(R); se, sr_ = .1 / np.sqrt(np.mean(E ** 2)), .1 / np.sqrt(np.mean(R ** 2))
    np.savez(HERE / 'data/piano_pairs.npz', E=(E * se).astype(np.float32), R=(R * sr_).astype(np.float32), midi=np.array(midi), dyn=np.array(dyn), engine_scale=se, real_scale=sr_)
    print('notes', len(midi), 'pitches', sorted(set(midi)))


class FrameFlow(nn.Module):
    """The v6 resonant flow acting on all frequency bins of a frame (same equations as render_audio.py, differentiable)."""
    def __init__(self, eq_only=False, seed=1, mixing=True):
        super().__init__(); torch.manual_seed(seed); self.m = models.SpectralFlow(seed=seed); self.eq_only = eq_only; self.mixing = mixing
        f = torch.fft.rfftfreq(N, 1 / SR); self.register_buffer('band', ((f >= LO) & (f <= TOP)).float()); self.register_buffer('u', torch.log(f.clamp(LO, TOP) / 1000.)[:, None])

    def forward(self, fr):
        m, u, band = self.m, self.u, self.band; Z = torch.fft.rfft(fr) * m.pre(u[:, 0]) * m.drive.exp() * band
        if not self.eq_only:
            a = m.fin(u); a = torch.complex(a[:, :m.nf], a[:, m.nf:]).T; q = m.q(u); R = m.r2 + m.r3; q = torch.complex(q[:, :R], q[:, R:]).T
            lin = m.lin(u); lin = torch.complex(lin[:, 0], lin[:, 1]); ka, kb, kp = nn.functional.softplus(m.ka(u)), nn.functional.softplus(m.kb(u)), m.kp(u); r2, r3 = m.r2, m.r3
            for s in range(m.steps):
                h = m.log_h[s].exp(); p = (2 * Z.abs() / N).square() * band
                if not self.mixing:
                    e = torch.einsum('kr,jr,fj->fk', ka, kb, p); ph = torch.einsum('kr,jr,fj->fk', kp, kb, p)
                    Z = Z * torch.exp(h * lin) * torch.polar(torch.ones_like(ph), h * torch.tanh(ph)) / (1 + h * e) * band; continue
                w = torch.fft.irfft(Z[:, None] * a, N)
                prod = torch.cat([w[:, :r2] * w[:, r2:2 * r2], w[:, 2 * r2:2 * r2 + r3] * w[:, 2 * r2 + r3:2 * r2 + 2 * r3] * w[:, 2 * r2 + 2 * r3:]], 1)
                t = (q * torch.fft.rfft(prod)).sum(1); e = torch.einsum('kr,jr,fj->fk', ka, kb, p); ph = torch.einsum('kr,jr,fj->fk', kp, kb, p)
                Z = (Z * torch.exp(h * lin) * torch.polar(torch.ones_like(ph), h * torch.tanh(ph)) + h * t) / (1 + h * e) * band
                en = (2 * Z.abs() / N).square().sum(-1, keepdim=True); Z = Z * torch.sqrt((4. / en.clamp_min(1e-12)).clamp(max=1.))        # safety limit
        return torch.fft.irfft(Z * m.post(u[:, 0]), N)


def mel_bank():
    f = np.fft.rfftfreq(N, 1 / SR); mel = lambda h: 2595 * np.log10(1 + h / 700); edges = 700 * (10 ** (np.linspace(mel(LO), mel(TOP), 82) / 2595) - 1); W = np.zeros((80, len(f)))
    for i in range(80): W[i] = np.clip(np.minimum((f - edges[i]) / (edges[i + 1] - edges[i]), (edges[i + 2] - f) / (edges[i + 2] - edges[i + 1])), 0, None)
    return torch.from_numpy(W).float()


MEL = mel_bank(); WIN = torch.hann_window(N)
def bands(fr): return torch.log((torch.fft.rfft(fr * WIN).abs().square() / N) @ MEL.T + 1e-7)


def frames(split):
    d = np.load(HERE / 'data/piano_pairs.npz'); pitches = sorted(set(d['midi'].tolist())); held = set(pitches[2::4])
    AE, AR = d['E'], d['R']; sel = [i for i, p in enumerate(d['midi']) if (p in held) == (split == 'held')]; E, R = [], []
    for i in sel:
        for s in range(0, AE.shape[1] - N, HOP):
            e, r = AE[i, s:s + N], AR[i, s:s + N]
            if max(np.abs(e).max(), np.abs(r).max()) > 3e-3: E.append(e); R.append(r)
    return torch.from_numpy(np.stack(E)), torch.from_numpy(np.stack(R)), sorted(held)


def score(model, E, R):
    with torch.no_grad(): return float(np.mean([float((bands(model(E[i:i + 256])) - bands(R[i:i + 256])).abs().mean()) for i in range(0, len(E), 256)]))


def fit(arm, steps=1200, seed=1):
    torch.set_num_threads(8); E, R, _ = frames('fit'); Eh, Rh, held = frames('held'); model = FrameFlow(arm == 'eq_only', seed, mixing=arm != 'no_mixing')
    opt = torch.optim.Adam(model.parameters(), 2e-3); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps, pct_start=.1); g = torch.Generator().manual_seed(seed)
    band = lambda v: torch.fft.irfft(torch.fft.rfft(v) * model.band, N); start = float((bands(band(Eh)) - bands(Rh)).abs().mean()); best = (9., None, 0); clock = time.time(); log = []
    print(arm, 'fit frames', len(E), 'held frames', len(Eh), 'held pitches', held, 'engine unprocessed, held:', round(start, 4), flush=True)
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(E), (96,), generator=g); loss = (bands(model(E[idx])) - bands(R[idx])).abs().mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step(); sched.step()
        if step % 100 == 0 or step == steps:
            v = score(model, Eh, Rh); log.append((step, float(loss), v))
            if v < best[0]: best = (v, {k: t.clone() for k, t in model.state_dict().items()}, step)
            print(f'{arm} step {step} loss {loss.item():.4f} held {v:.4f} best {best[0]:.4f} {time.time() - clock:.0f}s', flush=True)
    (HERE / 'checkpoints').mkdir(exist_ok=True); torch.save(dict(state=best[1], held=best[0], step=best[2], engine_unprocessed_held=start, log=log, held_pitches=held), HERE / f'checkpoints/piano_{arm}.pt')
    (HERE / f'piano_{arm}_result.json').write_text(json.dumps(dict(arm=arm, engine_unprocessed_held=start, held_error=best[0], best_step=best[2], held_pitches=held, parameters=sum(p.numel() for p in model.parameters())), indent=1))


@torch.no_grad()
def process(model, x):
    x = np.concatenate([np.zeros(N), x, np.zeros(2 * N)]); starts = np.arange(0, len(x) - N, HOP); y = np.zeros(len(x)); win = np.hanning(N + 1)[:-1]
    for b in range(0, len(starts), 256):
        st = starts[b:b + 256]; out = model(torch.from_numpy(np.stack([x[s:s + N] for s in st])).float()).numpy() * win
        for s, o in zip(st, out): y[s:s + N] += o
    return y[N:-2 * N] / 2.


def render():
    sys.path.insert(0, str(BASE / 'grand_piano_improved/tools')); import render_model as rm
    d = np.load(HERE / 'data/piano_pairs.npz'); se, sr_ = float(d['engine_scale']), float(d['real_scale'])
    piano = rm.GrandPiano(sample_rate=SR, room_mix=0., normalize=False); x = piano.render(rm.demo_events(), tail=3.).mean(1); (HERE / 'audition').mkdir(exist_ok=True); info = {}
    peak = lambda v: v * (.7 / np.abs(v).max())
    sf.write(HERE / 'audition/piano_1_engine_only.wav', peak(x), SR, subtype='PCM_24')
    for arm in ('resonant', 'eq_only', 'no_mixing'):
        p = HERE / f'checkpoints/piano_{arm}.pt'
        if not p.exists(): continue
        c = torch.load(p); model = FrameFlow(arm == 'eq_only', mixing=arm != 'no_mixing'); model.load_state_dict(c['state']); y = process(model, x * se) / sr_
        sf.write(HERE / ('audition/piano_' + dict(resonant='2_resonant_flow', eq_only='3_eq_only_control', no_mixing='4_flow_without_frequency_mixing')[arm] + '.wav'), peak(y), SR, subtype='PCM_24')
        S = np.abs(np.fft.rfft(y)) ** 2; f = np.fft.rfftfreq(len(y), 1 / SR); info[arm] = dict(held_error=c['held'], centroid_hz=round(float((S * f).sum() / S.sum())), nan=bool(np.isnan(y).any()), crest_db=round(float(20 * np.log10(np.abs(y).max() / np.sqrt(np.mean(y ** 2)))), 1))
    S = np.abs(np.fft.rfft(x)) ** 2; f = np.fft.rfftfreq(len(x), 1 / SR); info['engine'] = dict(centroid_hz=round(float((S * f).sum() / S.sum())), crest_db=round(float(20 * np.log10(np.abs(x).max() / np.sqrt(np.mean(x ** 2)))), 1)); print(json.dumps(info, indent=1))


if __name__ == '__main__':
    dict(prep=prep, render=render, fit=lambda: fit(sys.argv[2]))[sys.argv[1]]()
