"""Chorus learned from a real pedal: a modulated-delay layer (time-varying linear term) trained on waveforms.

Data: EGFxSet Clean vs Chorus, bridge pickup, 138 notes (real Stratocaster, real chorus pedal, CC BY 4.0).
Measured first (not assumed): the pedal output is dry + one delayed copy of about equal level whose delay swings by
about +-1.2 ms at 2.17 Hz.  The swing is at a different position in every recording (free-running pedal).
A first attempt on short-frame harmonic envelopes with one shared swing position failed (error 0.36 vs 0.43 for a
fixed filter); kept in the README.  This version works on the waveform:
    y(t) = (c * x)(t) + (g * x)(t - d0 - lfo(t))
    c, g     filters whose frequency responses come from small networks of log frequency (the flow's coupling networks)
    lfo(t)   three-term Fourier series with learned rate: the wave shape is learned, not fixed to a sine
    d0       base delay
    plus one number per recording: where the pedal's swing was when the note started (a nuisance initial condition;
    for held-out notes it is found by search, nothing else is fitted to them)
Split by fret as for the pedals: test 3, 8, 13, 18; validation 5, 15.     python chorus.py fit | test | render
"""
import sys
sys.dont_write_bytecode = True
import json, math
import numpy as np
import soundfile as sf
import torch
from torch import nn
import tones as T
from tones import HERE, DATA

SR, TAPS, LEAD = 44100, 256, 2048; OUT = HERE / 'chorus'; torch.set_num_threads(6)


def notes(split, seconds=4.3):
    X, Y, names = [], [], []
    for p in sorted((DATA / 'Clean').glob('*.wav')):
        if T.SPLIT(int(p.stem.split('-')[1])) != split: continue
        x, y = T.read(p), T.read(DATA / 'Chorus' / p.name); L = int(seconds * SR); k = 1 / np.sqrt((x[:SR] ** 2).mean()); X.append(x[:L] * k); Y.append(y[:L] * k); names.append(p.stem)
    return torch.from_numpy(np.stack(X)).float(), torch.from_numpy(np.stack(Y)).float(), names


def net(): return nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 2))


class Chorus(nn.Module):
    def __init__(self, modulated=True, d0=8.):
        super().__init__(); self.c, self.g = net(), net(); self.modulated = modulated
        for n in (self.c, self.g): nn.init.zeros_(n[-1].weight); nn.init.zeros_(n[-1].bias)
        self.d0 = nn.Parameter(torch.tensor(float(d0))); self.rate = nn.Parameter(torch.tensor(2.17)); self.depth = nn.Parameter(torch.tensor(1.2)); self.shape = nn.Parameter(torch.zeros(2, 2))     # ms; shape: sin/cos of 2x and 3x rate

    def fir(self, which, sr):
        f = torch.fft.rfftfreq(TAPS, 1 / sr); o = which(torch.log(f.clamp(70., 8000.) / 1000.)[:, None]); H = torch.polar(o[:, 0].clamp(-3, 3).exp(), o[:, 1] * (f / 8000.).clamp(max=1.))
        return torch.roll(torch.fft.irfft(H, TAPS), TAPS // 2) * torch.hann_window(TAPS, periodic=True)

    def filt(self, x, which, sr):
        return nn.functional.conv1d(x[:, None], self.fir(which, sr).flip(0)[None, None], padding=TAPS // 2)[:, 0, :x.shape[1]]

    def delay_ms(self, t, phase):
        a = 2 * math.pi * self.rate * t + phase[:, None]; return self.d0 + self.depth * torch.sin(a) + self.shape[0, 0] * torch.sin(2 * a) + self.shape[1, 0] * torch.cos(2 * a) + self.shape[0, 1] * torch.sin(3 * a) + self.shape[1, 1] * torch.cos(3 * a)

    def forward(self, x, t0, phase, sr=SR):
        """x [n, L]; t0 [n] seconds of the first sample; phase [n] swing position of each recording."""
        dry = self.filt(x, self.c, sr)
        if not self.modulated: return dry
        wet = self.filt(x, self.g, sr); L = x.shape[1]; idx = torch.arange(L)[None]; pos = idx - self.delay_ms(t0[:, None] + idx / sr, phase) * 1e-3 * sr
        i = pos.floor().long(); mu = pos - i; a = wet.gather(1, i.clamp(0, L - 1)); b = wet.gather(1, (i + 1).clamp(0, L - 1)); return dry + (a + mu * (b - a)) * (i >= 0)


def esr(p, y): return (p - y)[:, LEAD:].square().sum(-1) / y[:, LEAD:].square().sum(-1)


@torch.no_grad()
def best_phase(m, X, Y, seconds=1.5, steps=32):
    """Swing position of each recording by search over one cycle (first `seconds` of the note)."""
    L = int(seconds * SR); out = torch.zeros(len(X)); err = torch.full((len(X),), 9.)
    for ph in torch.arange(steps) * 2 * math.pi / steps:
        e = esr(m(X[:, :L], torch.zeros(len(X)), torch.full((len(X),), float(ph))), Y[:, :L]); better = e < err; out[better] = ph; err[better] = e[better]
    return out, err


def fit():
    X, Y, _ = notes(0); XV, YV, _ = notes(1); torch.manual_seed(1); (OUT / 'checkpoints').mkdir(parents=True, exist_ok=True); g = torch.Generator().manual_seed(1)
    # base delay: the error is periodic in it for any single note, so search it jointly over notes of different pitch
    sub = torch.randperm(len(X), generator=g)[:20]; best = (9., 0.)
    for d0 in np.arange(.5, 30., .1):
        m = Chorus(d0=d0); _, e = best_phase(m, X[sub], Y[sub], seconds=.6, steps=12); e = float(e.mean())
        if e < best[0]: best = (e, float(d0))
    print('base-delay search: mean error', round(best[0], 4), 'at', round(best[1], 2), 'ms', flush=True)
    m = Chorus(d0=best[1]); ph, e0 = best_phase(m, X, Y); phase = nn.Parameter(ph.clone()); print('start: median note error', round(float(e0.median()), 4), flush=True)
    steps = 1500; opt = torch.optim.Adam([dict(params=m.parameters(), lr=3e-3), dict(params=[phase], lr=1e-2)]); sched = torch.optim.lr_scheduler.OneCycleLR(opt, [3e-3, 1e-2], total_steps=steps, pct_start=.1); L = int(.6 * SR); keep = (9., None)
    for step in range(1, steps + 1):
        i = torch.randint(0, len(X), (12,), generator=g); s = int(torch.randint(0, X.shape[1] - L, (1,), generator=g)); p = m(X[i, s:s + L], torch.full((12,), s / SR), phase[i]); loss = esr(p, Y[i, s:s + L]).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 250 == 0 or step == steps:
            _, ev = best_phase(m, XV, YV); v = float(ev.median())
            if v < keep[0]: keep = (v, {k: t.clone() for k, t in m.state_dict().items()})
            print(f'chorus step {step} loss {loss.item():.4f} val median {v:.4f} d0 {float(m.d0):.3f} rate {float(m.rate):.4f} depth {float(m.depth):.3f}', flush=True)
    torch.save(dict(state=keep[1], val=keep[0], base_delay_search=best), OUT / 'checkpoints/chorus.pt')
    s = Chorus(modulated=False); opt = torch.optim.Adam(s.parameters(), 3e-3)
    for step in range(1, 601):
        i = torch.randint(0, len(X), (12,), generator=g); a = int(torch.randint(0, X.shape[1] - L, (1,), generator=g)); loss = esr(s(X[i, a:a + L], None, None), Y[i, a:a + L]).mean(); opt.zero_grad(); loss.backward(); opt.step()
    torch.save(dict(state=s.state_dict()), OUT / 'checkpoints/static.pt'); print('static filter final loss', round(loss.item(), 4))


def load(name='chorus'):
    m = Chorus(modulated=name == 'chorus'); m.load_state_dict(torch.load(OUT / f'checkpoints/{name}.pt')['state']); return m.eval()


@torch.no_grad()
def process(m, x, sr=SR, phase=0.):
    return m(torch.from_numpy(np.ascontiguousarray(x)).float()[None], torch.zeros(1), torch.tensor([float(phase)]), sr)[0].numpy().astype(np.float64)


@torch.no_grad()
def test():
    X, Y, names = notes(2); m, s = load(), load('static'); ph, _ = best_phase(m, X, Y, steps=48); full = lambda p: esr(p, Y).numpy()
    em = full(m(X, torch.zeros(len(X)), ph)); es = full(s(X, None, None)); ec = full(X * ((X * Y).sum(-1) / (X * X).sum(-1))[:, None])
    half = int(1.5 * SR); late = (lambda p: ((p - Y)[:, half:].square().sum(-1) / Y[:, half:].square().sum(-1)).numpy())(m(X, torch.zeros(len(X)), ph))
    tt = torch.linspace(0, 1 / float(m.rate), 400)[None]; d = m.delay_ms(tt, torch.zeros(1))[0].numpy()
    res = dict(held_out_notes=len(X), waveform_error_median=dict(clean_as_output=round(float(np.median(ec)), 3), static_filter=round(float(np.median(es)), 3), modulated_delay=round(float(np.median(em)), 3)),
               waveform_error_mean=dict(clean_as_output=round(float(ec.mean()), 3), static_filter=round(float(es.mean()), 3), modulated_delay=round(float(em.mean()), 3)), modulated_delay_worst_note=[names[int(em.argmax())], round(float(em.max()), 3)],
               modulated_delay_after_1p5s_median=round(float(np.median(late)), 3), parameters=sum(p.numel() for p in m.parameters()),
               learned=dict(base_delay_ms=round(float(m.d0), 3), rate_hz=round(float(m.rate), 4), delay_swing_ms=[round(float(d.min()), 2), round(float(d.max()), 2)], depth_ms=round(float(m.depth), 3), shape_ms=np.round(m.shape.numpy(), 3).tolist()),
               note='swing position of each held-out note found by search on its first 1.5 s; error is over the whole 4.3 s, and separately over the part after 1.5 s that the search did not see')
    print(json.dumps(res, indent=1)); (OUT / 'results.json').write_text(json.dumps(res, indent=1))


@torch.no_grad()
def render():
    m = load(); aud = HERE / 'audition'; aud.mkdir(exist_ok=True); X, Y, names = notes(2, 3.5); pick = [names.index(n) for n in ('6-3', '5-8', '4-3', '3-8', '2-13', '1-3', '5-13', '3-18')]; X, Y = X[pick], Y[pick]
    ph, _ = best_phase(m, X, Y, steps=48); P = m(X, torch.zeros(len(X)), ph); k = .89 / max(float(v.abs().max()) for v in (X, Y, P))
    for name, v in (('1_clean', X), ('2_real_pedal', Y), ('3_learned_model', P)): sf.write(aud / f'heldout_Chorus_{name}.flac', v.numpy().reshape(-1) * k, SR, subtype='PCM_16')
    print('wrote held-out chorus files; not listened to by the assistant')


if __name__ == '__main__':
    dict(fit=fit, test=test, render=render)[sys.argv[1]]()
