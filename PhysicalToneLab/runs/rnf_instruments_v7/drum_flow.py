"""Drums: per-hit flow, engine hit (drum_set_fea) -> real hit, for kick, snare and hi-hat.

Real data: IDMT-SMT-Drums V2 (Zenodo 7544164, CC BY-NC-ND 4.0, research use, not redistributed): the 42 RealDrum
'#train' files = isolated kick / snare / hi-hat tracks of a real acoustic kit; only those are fetched (11 MB).
Model: phase-preserving gain per frequency that changes with time since the hit: log-gain = g(f, piece) - rate(f, piece) * age.
Short frames (46 ms, hop 5.8 ms) because drums are transients.  Crash is left as the engine makes it (no real data).
Reads the engine, writes only in this folder.            python drum_flow.py fetch | prep | fit | check | render
"""
import sys
sys.dont_write_bytecode = True
import json, re, struct, zlib, zipfile, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from torch import nn
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent; BASE = HERE.parents[2]; ENG = BASE / 'electric_guitar_lead/distortion_guitar_fea/drum_set_fea'
DATA, PAIRS = HERE / 'data/drums', HERE / 'data/drum_pairs.npz'
SR, N, HOP, LO, TOP, SECONDS = 44100, 2048, 256, 30., 16000., .75
PIECES = dict(KD='kick', SD='snare', HH='hihat'); ORDER = ['kick', 'snare', 'hihat']
FREQ = torch.fft.rfftfreq(N, 1 / SR); BAND = ((FREQ >= LO) & (FREQ <= TOP)).float(); WIN = torch.hann_window(N)
_e = 700 * (10 ** (np.linspace(2595 * np.log10(1 + LO / 700), 2595 * np.log10(1 + TOP / 700), 66) / 2595) - 1); _f = FREQ.numpy()
MEL = torch.from_numpy(np.stack([np.clip(np.minimum((_f - _e[i]) / (_e[i + 1] - _e[i]), (_e[i + 2] - _f) / (_e[i + 2] - _e[i + 1])), 0, None) for i in range(64)])).float()
CENTRE = torch.from_numpy(_e[1:-1]).float(); FLOOR_HZ = torch.tensor([0., 150., 300.])      # kick, snare, hi-hat: below this the target is the engine itself
def bands(fr): return torch.log((torch.fft.rfft(fr * WIN).abs().square() / N) @ MEL.T + 1e-8)
def mlp(i, h, o): return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, o))


class DrumFlow(nn.Module):
    def __init__(self):
        super().__init__(); self.gain, self.rate = mlp(4, 32, 1), mlp(4, 32, 1)
        for net in (self.gain, self.rate): nn.init.zeros_(net[-1].weight); nn.init.zeros_(net[-1].bias)

    def forward(self, fr, piece, age):
        x = torch.cat([torch.log(FREQ.clamp(LO, TOP) / 1000.)[None, :, None].expand(len(fr), -1, -1), nn.functional.one_hot(piece, 3).float()[:, None].expand(-1, len(FREQ), -1)], -1)
        G = self.gain(x)[..., 0].clamp(-4.5, 2.3) - self.rate(x)[..., 0].clamp(-3, 40) * age[:, None]       # rate < 0 lengthens (limited), > 0 shortens
        G = torch.where(FREQ[None] < FLOOR_HZ[piece][:, None], G.clamp(max=0.), G)                        # below the floor the model may only cut
        return torch.fft.irfft(torch.fft.rfft(fr) * BAND * torch.exp(G.clamp(-12, 2.3)), N)                    # never more than +20 dB


def fetch():
    sys.path.insert(0, str(HERE)); import fetch_guitarset as fg
    fg.URL = 'https://zenodo.org/records/7544164/files/{}?download=1'; Z = 'IDMT-SMT-DRUMS-V2.zip'; DATA.mkdir(parents=True, exist_ok=True)
    for i in zipfile.ZipFile(fg.Remote(Z)).infolist():
        if 'RealDrum' not in i.filename or 'train' not in i.filename or not i.filename.lower().endswith('.wav'): continue
        head = fg.fetch(Z, i.header_offset, i.header_offset + 29); nlen, elen = struct.unpack('<HH', head[26:30]); a = i.header_offset + 30 + nlen + elen
        raw = fg.fetch(Z, a, a + i.compress_size - 1); data = raw if i.compress_type == 0 else zlib.decompress(raw, -15); assert zlib.crc32(data) == i.CRC; (DATA / i.filename.split('/')[-1].replace('#', '_')).write_bytes(data)
    print('files', len(list(DATA.glob('*.wav'))))


def engine():
    sys.path.insert(0, str(ENG)); import drum_model as dm
    return dm, json.loads((ENG / 'drum_set_fea_modes.json').read_text(encoding='utf-8'))            # existing mode file, read only


def cut(x, L):
    o = max(0, int(np.argmax(np.abs(x) > .15 * np.abs(x).max())) - 64); x = x[o:o + L]; return np.pad(x, (0, L - len(x)))


def onsets(x):
    hop = 64; env = np.sqrt(np.convolve(x ** 2, np.ones(256) / 256, 'same')[::hop]); rise = np.maximum(np.log(env[3:] + 1e-4) - np.log(env[:-3] + 1e-4), 0) * (env[3:] > .12 * env.max()); out = []
    for i in np.argsort(rise)[::-1]:
        if rise[i] < .7: break
        if all(abs(i - j) > .12 * SR / hop for j in out): out.append(i)
    return sorted(int(i * hop) for i in out)


def prep():
    dm, model = engine(); L = int(SECONDS * SR); E, R, P, F, LEN = [], [], [], [], []; rng = np.random.default_rng(11)
    for fi, f in enumerate(sorted(DATA.glob('*.wav'))):
        m = re.search(r'_(KD|SD|HH)_train', f.name); x, sr = sf.read(f, dtype='float64', always_2d=True); x = x.mean(1)
        if not m or sr != SR: continue
        piece = PIECES[m.group(1)]; on = onsets(x)
        for k, a in enumerate(on):
            b = min(a + L, (on[k + 1] - 128) if k + 1 < len(on) else len(x)); n = b - a
            if n < int(.1 * SR): continue
            r = np.zeros(L); r[:n] = x[a:b]; r[max(0, n - 256):n] *= np.linspace(1, 0, min(256, n)); r = cut(r, L)
            e = cut(resample_poly(dm.synth_hit(dm.DrumHit(0., piece, 108, 0.), model, rng), 147, 160), L); e[n:] = 0; e[max(0, n - 256):n] *= np.linspace(1, 0, min(256, n))
            E.append(e); R.append(r); P.append(ORDER.index(piece)); F.append(fi); LEN.append(n / SR)
    E, R = np.stack(E), np.stack(R); n = int(.08 * SR); R *= (np.sqrt((E[:, :n] ** 2).mean(1)) / np.maximum(np.sqrt((R[:, :n] ** 2).mean(1)), 1e-6))[:, None]      # level-match each real hit to its engine hit
    s = .1 / np.sqrt(np.mean(E ** 2)); np.savez(PAIRS, E=(E * s).astype(np.float32), R=(R * s).astype(np.float32), piece=np.array(P), src=np.array(F), length=np.array(LEN), scale=s)
    print('hits', len(P), 'per piece', {ORDER[i]: int((np.array(P) == i).sum()) for i in range(3)}, 'median length s', {ORDER[i]: round(float(np.median(np.array(LEN)[np.array(P) == i])), 2) for i in range(3)})


def split(d):
    files = sorted(set(d['src'].tolist())); held = set(files[2::4]); return np.array([f in held for f in d['src']])


def frames(want_held):
    d = np.load(PAIRS); AE, AR, P = d['E'], d['R'], d['piece']; h = split(d); out = [[] for _ in range(4)]
    for i in np.where(h == want_held)[0]:
        n = int(d['length'][i] * SR)
        for s in range(-N // 2, n, HOP):                                  # frames centred from the hit onset onward
            a = max(0, s); e = np.zeros(N); r = np.zeros(N); e[a - s:a - s + max(0, min(N, AE.shape[1] - s) - (a - s))] = AE[i, a:s + N][:N - (a - s)]; r[a - s:a - s + len(AR[i, a:s + N])] = AR[i, a:s + N]
            if max(np.abs(e).max(), np.abs(r).max()) > 2e-3:
                for o, v in zip(out, (e, r, P[i], max(0., (s + N / 2) / SR))): o.append(v)
    return torch.from_numpy(np.stack(out[0])).float(), torch.from_numpy(np.stack(out[1])).float(), torch.tensor(out[2]).long(), torch.tensor(out[3]).float()


def load():
    m = DrumFlow(); m.load_state_dict(torch.load(HERE / 'checkpoints/drum_flow.pt')['state']); return m.eval()


def fit(steps=3000, seed=1):
    torch.set_num_threads(8); torch.manual_seed(seed); E, R, P, A = frames(False); held = frames(True); model = DrumFlow(); g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), 3e-3); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps, pct_start=.1); best = (9., None, 0); clock = time.time()

    def loss_of(mdl, e, r, p, a):
        tg = torch.where(CENTRE[None] < FLOOR_HZ[p][:, None], bands(torch.fft.irfft(torch.fft.rfft(e) * BAND, N)), bands(r)); ob = bands(mdl(e, p, a)); top = torch.maximum(tg, ob.detach()); bw = ((top - tg.max(-1, keepdim=True).values) / 5.8 + 1).clamp(0, 1); return ((ob - tg).abs() * bw).sum(-1) / bw.sum(-1)     # bands within 50 dB count
    def score(mdl):
        with torch.no_grad(): return float(torch.cat([loss_of(mdl, *[t[i:i + 512] for t in held]) for i in range(0, len(held[0]), 512)]).mean())
    start = score(lambda fr, p, a: torch.fft.irfft(torch.fft.rfft(fr) * BAND, N)); print('fit frames', len(E), 'held frames', len(held[0]), 'engine unprocessed, held:', round(start, 4), flush=True)
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(E), (256,), generator=g); err = loss_of(model, E[idx], R[idx], P[idx], A[idx]); w = 1 / (.1 + A[idx]); loss = (err * w).sum() / w.sum()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step(); sched.step()
        if step % 300 == 0 or step == steps:
            v = score(model)
            if v < best[0]: best = (v, {k: t.clone() for k, t in model.state_dict().items()}, step)
            print(f'drum step {step} loss {loss.item():.4f} held {v:.4f} best {best[0]:.4f} {time.time() - clock:.0f}s', flush=True)
    (HERE / 'checkpoints').mkdir(exist_ok=True); torch.save(dict(state=best[1], held=best[0]), HERE / 'checkpoints/drum_flow.pt'); (HERE / 'drum_flow_result.json').write_text(json.dumps(dict(engine_unprocessed_held=start, held_error=best[0], best_step=best[2]), indent=1))


@torch.no_grad()
def process(model, x, piece):
    """One hit starting at sample 0.  Hann frames at hop N/8 sum to 4."""
    x = np.concatenate([np.zeros(N), x, np.zeros(2 * N)]); starts = np.arange(0, len(x) - N, HOP); win = np.hanning(N + 1)[:-1]; y = np.zeros(len(x))
    fr = torch.from_numpy(np.stack([x[s:s + N] for s in starts])).float(); age = torch.from_numpy(np.clip((starts + N / 2 - N) / SR, 0, SECONDS)).float()
    out = model(fr, torch.full((len(starts),), ORDER.index(piece)), age).numpy() * win
    for s, o in zip(starts, out): y[s:s + N] += o
    return y[N:-2 * N] / 4.


def check():
    d = np.load(PAIRS); E, R, P = d['E'], d['R'], d['piece']; h = split(d); m = load(); edges = [30, 60, 120, 250, 500, 1000, 2000, 4000, 8000, 16000]; out = dict(bands_from_hz=edges[:-1])
    def prof(x, a, b):
        seg = x[int(a * SR):int(b * SR)]; S = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 8192)) ** 2; f = np.fft.rfftfreq(8192, 1 / SR); return np.array([10 * np.log10(S[(f >= lo) & (f < hi)].sum() + 1e-12) for lo, hi in zip(edges[:-1], edges[1:])])
    for pi, name in enumerate(ORDER):
        idx = [i for i in np.where(h)[0] if P[i] == pi and d['length'][i] > .2]; row = {}
        Y = {i: process(m, E[i].astype(np.float64), name) for i in idx}
        for label, (a, b) in dict(attack_0_50ms=(0., .05), body_60_200ms=(.06, .2)).items():
            de, dm_ = [], []
            for i in idx:
                pr = prof(R[i], a, b); ok = pr > pr.max() - 45; de.append(np.where(ok, prof(E[i], a, b) - pr, np.nan)); dm_.append(np.where(ok, prof(Y[i], a, b) - pr, np.nan))
            row[label] = dict(engine_minus_real=np.round(np.nanmean(de, 0), 1).tolist(), model_minus_real=np.round(np.nanmean(dm_, 0), 1).tolist(), engine_mean_abs=round(float(np.nanmean(np.abs(de))), 2), model_mean_abs=round(float(np.nanmean(np.abs(dm_))), 2))
        dec = lambda x: 20 * np.log10(np.sqrt((x[int(.12 * SR):int(.2 * SR)] ** 2).mean() + 1e-12) / np.sqrt((x[:int(.04 * SR)] ** 2).mean() + 1e-12))
        row['level_change_by_160ms_db'] = dict(engine=round(float(np.median([dec(E[i]) for i in idx])), 1), model=round(float(np.median([dec(Y[i]) for i in idx])), 1), real=round(float(np.median([dec(R[i]) for i in idx])), 1)); row['held_hits'] = len(idx); out[name] = row
        print(name, json.dumps(row), flush=True)
    (HERE / 'drum_summary.json').write_text(json.dumps(out, indent=1))


def render():
    """tab2 drum part through the engine's own mixing, with and without the per-hit flow (crash untouched)."""
    dm, model = engine(); flow = load(); s = float(np.load(PAIRS)['scale']); score = json.loads((ENG / 'tab2_drum_score.json').read_text(encoding='utf-8'))
    hits = [dm.DrumHit(e['start'], e['piece'], int(e['velocity']), float(e.get('open_amount', 0.))) for e in score['events']]; seconds = float(score['duration_s']); original = dm.synth_hit; info = {}

    def with_flow(hit, mdl, rng, maximum_tail=None):
        x = original(hit, mdl, rng, maximum_tail)
        if hit.piece not in ORDER: return x
        y = resample_poly(process(flow, resample_poly(x, 147, 160) * s, hit.piece) / s, 160, 147)[:len(x)]; y = np.pad(y, (0, len(x) - len(y))); fade = min(int(.01 * dm.SR), len(y)); y[-fade:] *= np.linspace(1, 0, fade); return y
    (HERE / 'audition').mkdir(exist_ok=True); edges = [30, 60, 120, 250, 500, 1000, 2000, 4000, 8000, 16000]
    for name, fn in (('drums_tab2_1_engine_only', original), ('drums_tab2_2_flow', with_flow)):
        dm.synth_hit = fn; y = dm.render_drums(model, hits, seconds); dm.synth_hit = original                 # in-memory only; the engine file is not touched
        sf.write(HERE / 'audition' / f'{name}.flac', y, dm.SR, subtype='PCM_16'); S = np.abs(np.fft.rfft(y.mean(1))) ** 2; f = np.fft.rfftfreq(len(y), 1 / dm.SR); e = np.array([S[(f >= a) & (f < b)].sum() for a, b in zip(edges[:-1], edges[1:])])
        info[name] = dict(band_db=dict(zip(map(str, edges[:-1]), np.round(10 * np.log10(e / e.sum()), 1).tolist())), nan=bool(np.isnan(y).any()))
    print(json.dumps(info)); (HERE / 'audition/drums_tab2_notes.json').write_text(json.dumps(dict(info, hits=len(hits), note='kick, snare, hi-hat through the per-hit flow; crash as the engine makes it; not listened to by the assistant'), indent=1))


if __name__ == '__main__':
    dict(fetch=fetch, prep=prep, fit=fit, check=check, render=render)[sys.argv[1]]()
