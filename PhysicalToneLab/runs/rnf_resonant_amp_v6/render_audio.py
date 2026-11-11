"""Run the SAME trained resonant flow on the full audio signal (attacks, noise and chords included).

The trained equations are filter -> multiply pointwise -> filter, plus level-dependent damping, with every
coupling a function of absolute frequency.  On a tracked note they act on its harmonics; here they act on all
frequency bins of short overlapping frames.  No retraining, same weights.  Writes only to audition/.

  python render_audio.py test    # waveform error against the real-amp target on the test recordings
  python render_audio.py song    # Sun Raha Hai lead
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
import numpy as np
import soundfile as sf
import torch
from torch import nn
from scipy.signal import resample_poly
import models, extract

HERE = Path(__file__).resolve().parent
OLD = HERE.parents[2] / 'NeuralInstrumentStudio_Experiments/energy_v3'
SR, N, HOP, LO, TOP = 44100, 4096, 1024, 70., 8000.          # frames are taken at 2x sample rate (88.2 kHz): 46 ms, hop 11.6 ms


def scales():
    d = np.load(HERE / 'envelopes.npz'); tr = d['split'] == 0
    return np.sqrt(np.mean(np.abs(d['B'][tr]) ** 2) * models.H), np.sqrt(np.mean(np.abs(d['A'][tr]) ** 2) * models.H)


def load(seed=2):
    c = torch.load(HERE / f'checkpoints/resonant_seed{seed}.pt'); m = models.build('resonant', seed); m.load_state_dict(c['state']); return m.eval()


@torch.no_grad()
def step_caps(m):
    """Largest state energy per flow step on the training frames (99.9th percentile).  Used as a safety limit."""
    import train
    B, A, f0, mask = train.data(0); u = models.feat(f0)[..., None]; z = B * m.pre(u[..., 0]) * m.drive.exp() * mask
    a = m.fin(u); a = torch.complex(a[..., :m.nf], a[..., m.nf:]); q = m.q(u); R = m.r2 + m.r3; q = torch.complex(q[..., :R], q[..., R:])
    lin = m.lin(u); lin = torch.complex(lin[..., 0], lin[..., 1]); ka, kb, kp = nn.functional.softplus(m.ka(u)), nn.functional.softplus(m.kb(u)), m.kp(u); r2, r3 = m.r2, m.r3; caps = []
    for s in range(m.steps):
        h = m.log_h[s].exp(); p = z.abs().square() * mask
        spec = torch.zeros(len(z), m.nf, m.n // 2 + 1, dtype=z.dtype); spec[:, :, 1:models.H + 1] = (a * z[..., None]).transpose(1, 2) * m.n / 2; w = torch.fft.irfft(spec, m.n)
        prod = torch.cat([w[:, :r2] * w[:, r2:2 * r2], w[:, 2 * r2:2 * r2 + r3] * w[:, 2 * r2 + r3:2 * r2 + 2 * r3] * w[:, 2 * r2 + 2 * r3:]], 1)
        t = (q * (torch.fft.rfft(prod)[:, :, 1:models.H + 1] * 2 / m.n).transpose(1, 2)).sum(-1)
        e = torch.einsum('fkr,fjr,fj->fk', ka, kb, p); ph = torch.einsum('fkr,fjr,fj->fk', kp, kb, p)
        z = (z * torch.exp(h * lin) * torch.polar(torch.ones_like(ph), h * torch.tanh(ph)) + h * t) / (1 + h * e) * mask
        caps.append(float(torch.quantile(z.abs().square().sum(-1), .999)))
    return caps


LIMITED = [0, 0]


@torch.no_grad()
def process(m, x):
    """x: mono 44.1 kHz in the level convention of the training pairs.  Returns amped audio."""
    sb, sa = scales(); x2 = resample_poly(x, 2, 1) / sb; pad = N; x2 = np.concatenate([np.zeros(pad), x2, np.zeros(pad + N)])
    f = torch.fft.rfftfreq(N, 1 / (2 * SR)); band = ((f >= LO) & (f <= TOP)).float(); u = torch.log(f.clamp(LO, TOP) / 1000.)[:, None]
    pre, post = m.pre(u[:, 0]) * m.drive.exp(), m.post(u[:, 0])
    a = m.fin(u); a = torch.complex(a[:, :m.nf], a[:, m.nf:]).T                                   # nf, bins
    q = m.q(u); R = m.r2 + m.r3; q = torch.complex(q[:, :R], q[:, R:]).T                         # R, bins
    lin = m.lin(u); lin = torch.complex(lin[:, 0], lin[:, 1])
    ka, kb, kp = nn.functional.softplus(m.ka(u)), nn.functional.softplus(m.kb(u)), m.kp(u)       # bins, rank
    caps = step_caps(m); win = torch.from_numpy(np.hanning(N + 1)[:-1]).float(); starts = np.arange(0, len(x2) - N, HOP); y = np.zeros(len(x2)); r2, r3 = m.r2, m.r3
    for b in range(0, len(starts), 256):
        st = starts[b:b + 256]; fr = torch.from_numpy(np.stack([x2[s:s + N] for s in st])).float()
        Z = torch.fft.rfft(fr) * pre * band                                                       # spectrum of the waveform
        for s in range(m.steps):
            h = m.log_h[s].exp(); p = (2 * Z.abs() / N).square() * band                           # sinusoid-amplitude power per bin
            w = torch.fft.irfft(Z[:, None] * a, N)                                                # frames, nf, N filtered waveforms
            prod = torch.cat([w[:, :r2] * w[:, r2:2 * r2], w[:, 2 * r2:2 * r2 + r3] * w[:, 2 * r2 + r3:2 * r2 + 2 * r3] * w[:, 2 * r2 + 2 * r3:]], 1)
            t = (q * torch.fft.rfft(prod)).sum(1)
            e = torch.einsum('kr,jr,fj->fk', ka, kb, p); ph = torch.einsum('kr,jr,fj->fk', kp, kb, p)
            Z = (Z * torch.exp(h * lin) * torch.polar(torch.ones_like(ph), h * torch.tanh(ph)) + h * t) / (1 + h * e) * band
            en = (2 * Z.abs() / N).square().sum(-1); g = torch.sqrt((caps[s] / en.clamp_min(1e-12)).clamp(max=1.)); Z = Z * g[:, None]   # safety limit: largest training energy
            if s == m.steps - 1: LIMITED[0] += int((g < 1).sum()); LIMITED[1] += len(g)
        out = (torch.fft.irfft(Z * post, N) * win).numpy()
        for s, o in zip(st, out): y[s:s + N] += o
    y = y[pad:pad + 2 * len(x)] / 2. * sa                                                         # Hann at hop N/4 sums to 2
    return resample_poly(y, 1, 2)[:len(x)]


def describe(a):
    S = np.abs(np.fft.rfft(a)) ** 2; f = np.fft.rfftfreq(len(a), 1 / SR)
    return dict(rms=round(float(np.sqrt(np.mean(a ** 2))), 4), peak=round(float(np.abs(a).max()), 3), centroid_hz=round(float((S * f).sum() / S.sum())),
                crest_db=round(float(20 * np.log10(np.abs(a).max() / np.sqrt(np.mean(a ** 2)))), 1))


def esr(y, t):
    """best integer lag within +-64 samples, then error-to-signal ratio"""
    best = min(((np.roll(y, k) - t)[200:-200] ** 2).sum() for k in range(-64, 65)); return float(best / (t[200:-200] ** 2).sum())


def test():
    m = load(); manifest = json.loads((extract.PAIRS / 'manifest.json').read_text())['pairs']; rows = {}
    for row in manifest:
        if row['split'] != 'test': continue
        d = np.load(extract.PAIRS / row['file']); x, t = d['x'].astype(np.float64), d['y'].astype(np.float64); y = process(m, x)
        lp = lambda v: resample_poly(resample_poly(v, 2, 5), 5, 2)[:len(v)]                        # compare below ~8.8 kHz too
        rows[row['id']] = dict(waveform_error=round(esr(y, t), 3), waveform_error_below_8k=round(esr(lp(y), lp(t)), 3), clean_as_output=round(esr(x * (x @ t) / (x @ x), t), 3),
                               target=describe(t), resonant_flow=describe(y)); rows[row['id']]['frames_limited'] = list(LIMITED); print(row['id'], rows[row['id']], flush=True)
    (HERE / 'audio_rate_test.json').write_text(json.dumps(rows, indent=1))


def song():
    m = load(); x, sr = sf.read(OLD / 'artifact_study/clean_release_di.wav', dtype='float64', always_2d=True); assert sr == SR; x = x.mean(1)
    ref, _ = sf.read(OLD / 'artifact_study/clean_release_dry.wav', dtype='float64', always_2d=True); ref = ref.mean(1); n = min(len(x), len(ref))
    info = {}
    for gain_db in (0, -6):                                                                       # song DI is louder than the training DI
        y = process(m, x * 10 ** (gain_db / 20)); name = f'song_4_resonant_flow_full_audio{"" if gain_db == 0 else f"_input{gain_db}dB"}.wav'
        sf.write(HERE / 'audition' / name, y, SR, subtype='PCM_24'); info[name] = dict(describe(y), frames_limited=list(LIMITED), waveform_error_vs_previous_amp=round(esr(y[:n], ref[:n]), 3))
    info['previous_amp'] = describe(ref[:n]); (HERE / 'audition/song_4_notes.json').write_text(json.dumps(info, indent=1)); print(json.dumps(info, indent=1))


if __name__ == '__main__':
    dict(test=test, song=song)[sys.argv[1]]()
