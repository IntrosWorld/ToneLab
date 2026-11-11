"""Acoustic guitar: per-note flow, engine note (acoustic_guitar_fea, steel-string model) -> real steel-string note.

Real data: GuitarSet (steel-string acoustic, microphone, CC BY 4.0), notes that sound alone in the solo takes.
  Same model family and lessons as the approved piano chain:
per note, pitch/velocity conditioned, phase-preserving (no new frequencies), every band trained toward the real note,
no partial may grow over time, attack weighted most.  Reads engine/originals, writes only in this folder.

  python guitar_flow.py prep | fit | check | render
"""
import sys
sys.dont_write_bytecode = True
import json, time
import numpy as np
import soundfile as sf
import torch
import piano_clean as pc, piano_decay as pd
from piano_low import LowFlow
from piano_clean import HERE, BASE, SR, N, HOP, bands

MODEL, FLOOR = LowFlow, 4.6            # model class; loss ignores bands more than FLOOR nepers (4.6 = 40 dB) below the frame's strongest
DATA = HERE / 'data/guitarset'; PAIRS = HERE / 'data/guitar_pairs.npz'; SECONDS, MAX_NOTES = 2.0, 900
_G = {}


def _engine_note(job):
    midi, vel, string, length = job
    if not _G:
        sys.path.insert(0, str(BASE / 'acoustic_guitar_fea/tools')); import acoustic_engine as ae; _G['g'] = ae.AcousticGuitar(sample_rate=SR)
    L = int(SECONDS * SR)
    try: e = _G['g'].render([dict(midi=midi, velocity=vel, start=0., duration=length, string_index=string)], tail=.3, normalize=False, room=False).mean(1)
    except Exception as err: return None
    o = max(0, int(np.argmax(np.abs(e) > .1 * np.abs(e).max())) - 256); e = e[o:o + L]; return np.pad(e, (0, L - len(e))).astype(np.float32)


def prep():
    """GuitarSet (steel-string acoustic, CC BY 4.0): notes that sound alone in the solo takes, microphone recording."""
    import io, zipfile
    from multiprocessing import Pool
    za = zipfile.ZipFile(DATA / 'annotation.zip'); audio = {q.name: q for q in (DATA / 'wav').glob('*.wav')}; L = int(SECONDS * SR); cand = []
    for name in sorted(n for n in za.namelist() if n.endswith('_solo.jams')):
        wav = name.split('/')[-1].replace('.jams', '_mic.wav')
        if wav not in audio: continue
        j = json.loads(za.read(name)); notes = []
        for si, ann in enumerate(a for a in j['annotations'] if a['namespace'] == 'note_midi'):
            data = ann['data']; rows = data if isinstance(data, list) else [dict(time=t, duration=d, value=v) for t, d, v in zip(data['time'], data['duration'], data['value'])]
            src = ann.get('annotation_metadata', {}).get('data_source'); string = int(src) if str(src).isdigit() else si
            notes += [(float(r['time']), float(r['duration']), float(r['value']), string) for r in rows]
        notes.sort()
        for k, (t, dur, val, string) in enumerate(notes):
            length = min(dur, SECONDS - .05)
            if dur < .3 or abs(val - round(val)) > .3: continue
            if any(o is not n and o[0] < t + length and o[0] + o[1] > t - .02 for n in [notes[k]] for o in notes[max(0, k - 8):k + 9]): continue
            cand.append((wav, t, length, int(round(val)), string))
    rng = np.random.default_rng(7); pick = [cand[i] for i in sorted(rng.choice(len(cand), min(MAX_NOTES, len(cand)), replace=False))]; print('isolated notes found', len(cand), 'used', len(pick), flush=True)
    R, lvl = [], []; cache = {}
    for wav, t, length, midi, string in pick:
        if wav not in cache:
            cache.clear(); x, sr = sf.read(audio[wav], dtype='float64', always_2d=True); assert sr == SR; cache[wav] = x.mean(1)
        x = cache[wav]; a = max(0, int(t * SR) - 256); n = min(int(length * SR) + 256, len(x) - a); r = np.zeros(L); r[:n] = x[a:a + n]; r[max(0, n - 1024):n] *= np.linspace(1, 0, min(1024, n))
        R.append(r); lvl.append(20 * np.log10(np.sqrt(np.mean(r[:int(.12 * SR)] ** 2)) + 1e-9))
    rank = np.argsort(np.argsort(lvl)) / max(1, len(lvl) - 1); vel = np.round(50 + 68 * rank).astype(int)                      # louder real note -> higher engine velocity
    with Pool(6) as pool: E = pool.map(_engine_note, [(m, int(v), s, l) for (w, t, l, m, s), v in zip(pick, vel)], chunksize=8)
    ok = [i for i, e in enumerate(E) if e is not None]; print('engine could play', len(ok), 'of', len(pick), flush=True); E = np.stack([E[i] for i in ok]); R = np.stack([R[i] for i in ok])
    for i, k in enumerate(ok):                                                                                               # engine excerpt cut to the real note's length
        n = int(pick[k][2] * SR) + 256; E[i, n:] = 0; E[i, max(0, n - 1024):n] *= np.linspace(1, 0, min(1024, n))
    se, sr_ = .1 / np.sqrt(np.mean(E ** 2)), .1 / np.sqrt(np.mean(R ** 2)); M = np.array([pick[k][3] for k in ok])
    np.savez(PAIRS, E=(E * se).astype(np.float32), R=(R * sr_).astype(np.float32), midi=M, vel=vel[ok], string=np.array([pick[k][4] for k in ok]), length=np.array([pick[k][2] for k in ok]), engine_scale=se, real_scale=sr_)
    print('notes', len(M), 'pitch range', M.min(), M.max(), 'median length s', float(np.median([pick[k][2] for k in ok])), 'per string', np.bincount(np.array([pick[k][4] for k in ok]), minlength=6).tolist())


def matched():
    """Pairs with every real note scaled to its engine note's level over the first 0.3 s (tone and decay are learned, not loudness)."""
    d = np.load(PAIRS); E, R = d['E'], d['R'].copy(); n = int(.3 * SR)
    R *= (np.sqrt((E[:, :n] ** 2).mean(1)) / np.maximum(np.sqrt((R[:, :n] ** 2).mean(1)), 1e-6))[:, None]
    return E, R, d['midi'], d['vel']


_e = 700 * (10 ** (np.linspace(2595 * np.log10(1 + pc.LO / 700), 2595 * np.log10(1 + pc.TOP / 700), 98) / 2595) - 1); CENTRE = torch.from_numpy(_e[1:-1]).float()
def target(E, R, midi):
    """Real note above 0.8 x fundamental; below it the engine itself (never chase room rumble)."""
    f0 = 440. * 2 ** ((midi - 69) / 12); return torch.where(CENTRE[None] < .8 * f0[:, None], bands(E), bands(R))


def frames(split):
    AE, AR, M, V = matched(); held = set(sorted(set(M.tolist()))[2::4]); out = [[] for _ in range(5)]
    for i in [i for i, p in enumerate(M) if (p in held) == (split == 'held')]:
        for s in range(0, AE.shape[1] - N, HOP):
            if min(np.abs(AE[i, s:s + N]).max(), np.abs(AR[i, s:s + N]).max()) > 2e-3:
                for o, v in zip(out, (AE[i, s:s + N], AR[i, s:s + N], M[i], float(V[i]), s / SR)): o.append(v)
    return torch.from_numpy(np.stack(out[0])), torch.from_numpy(np.stack(out[1])), torch.tensor(out[2]).float(), torch.tensor(out[3]).float(), torch.tensor(out[4]).float()


def load(name='guitar_flow'):
    m = MODEL(); m.load_state_dict(torch.load(HERE / f'checkpoints/{name}.pt')['state']); return m.eval()


def fit(steps=3000, seed=1, name='guitar_flow'):
    torch.set_num_threads(8); torch.manual_seed(seed); E, R, mi, ve, ag = frames('fit'); held = frames('held'); model = MODEL()
    opt = torch.optim.Adam(model.parameters(), 3e-3); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps, pct_start=.1); g = torch.Generator().manual_seed(seed); best = (9., None, 0); clock = time.time()
    def score(mdl):
        with torch.no_grad(): return float(np.mean([float((bands(mdl(held[0][i:i + 256], held[2][i:i + 256], held[3][i:i + 256], held[4][i:i + 256])) - target(held[0][i:i + 256], held[1][i:i + 256], held[2][i:i + 256])).abs().mean()) for i in range(0, len(held[0]), 256)]))
    start = score(lambda fr, m, v, a: torch.fft.irfft(torch.fft.rfft(fr) * pc.BAND, N)); print('fit frames', len(E), 'held frames', len(held[0]), 'engine unprocessed, held:', round(start, 4), flush=True)
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(E), (128,), generator=g); tg = target(E[idx], R[idx], mi[idx]); ob = bands(model(E[idx], mi[idx], ve[idx], ag[idx])); top = torch.maximum(tg, ob.detach()); bw = ((top - tg.max(-1, keepdim=True).values) / FLOOR + 1).clamp(0, 1); err = ((ob - tg).abs() * bw).sum(-1) / bw.sum(-1); w = 1 / (.08 + ag[idx]); loss = (err * w).sum() / w.sum()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step(); sched.step()
        if step % 300 == 0 or step == steps:
            v = score(model)
            if v < best[0]: best = (v, {k: t.clone() for k, t in model.state_dict().items()}, step)
            print(f'{name} step {step} loss {loss.item():.4f} held {v:.4f} best {best[0]:.4f} {time.time() - clock:.0f}s', flush=True)
    torch.save(dict(state=best[1], held=best[0]), HERE / f'checkpoints/{name}.pt'); (HERE / f'{name}_result.json').write_text(json.dumps(dict(engine_unprocessed_held=start, held_error=best[0], best_step=best[2]), indent=1))


def check():
    d = np.load(PAIRS); E, R, M, V = matched(); m = load(); edges = [60, 120, 250, 500, 1000, 2000, 4000, 10000]; W = int(.2 * SR); f = np.fft.rfftfreq(W, 1 / SR)
    prof = lambda x, t0: ' '.join(f'{10 * np.log10((np.abs(np.fft.rfft(x[int(t0 * SR):int(t0 * SR) + W] * np.hanning(W))) ** 2)[(f >= a) & (f < b)].sum() + 1e-14):6.1f}' for a, b in zip(edges[:-1], edges[1:]))
    drop = lambda x: 20 * np.log10(np.sqrt((x[int(.4 * SR):int(.6 * SR)] ** 2).mean() + 1e-14) / np.sqrt((x[int(.02 * SR):int(.2 * SR)] ** 2).mean() + 1e-14))
    held = sorted(set(M.tolist()))[2::4]; print('held-out pitches', held); print('absolute band level dB      ', ' '.join(f'{a:>6}' for a in edges[:-1]), '  level change by ~0.5 s')
    for p in held[::3]:
        ks = [k for k in range(len(M)) if M[k] == p]; i = max(ks, key=lambda k: float(d['length'][k])); y = pd.process(m, E[i].astype(np.float64), p, int(V[i]))
        for t0 in (0., .3):
            for name, v in (('engine', E[i]), ('model ', y), ('REAL  ', R[i])): print(f'midi {p} t={t0} {name}', prof(v, t0), f'   {drop(v):6.1f} dB' if t0 == 0 else '')


def render(which='demo'):
    sys.path.insert(0, str(BASE / 'acoustic_guitar_fea/tools')); import acoustic_engine as ae, render_model as rm
    d = np.load(PAIRS); se, sr_ = float(d['engine_scale']), float(d['real_scale']); model = load(); guitar = ae.AcousticGuitar(sample_rate=SR)
    events = dict(demo=rm.demo_events, arpeggio=rm.emotional_arpeggio_events)[which](); rows = [dict(e) if isinstance(e, dict) else dict(midi=e[0], velocity=e[1], start=e[2], duration=e[3]) for e in events]
    total = int((max(r['start'] + r['duration'] for r in rows) + 4.5) * SR); dry, wet = np.zeros((total, 2)), np.zeros((total, 2))
    for r in rows:
        a = int(r['start'] * SR); x = guitar.render([dict(r, start=0.)], tail=3., normalize=False, room=False); n = min(len(x), total - a); dry[a:a + n] += x[:n]
        y = np.stack([pd.process(model, x[:, c] * se, r['midi'], r['velocity']) / sr_ for c in range(2)], 1); wet[a:a + n] += y[:n]
    info = {}; edges = [60, 120, 250, 500, 1000, 2000, 4000, 10000]; f = np.fft.rfftfreq(total, 1 / SR); (HERE / 'audition').mkdir(exist_ok=True)
    for name, v in ((f'guitar_{which}_1_engine_only', dry), (f'guitar_{which}_2_flow', wet)):
        v = guitar._room(v) if hasattr(guitar, '_room') else v; S = np.abs(np.fft.rfft(v.mean(1), total)) ** 2 if len(v) == total else np.abs(np.fft.rfft(v.mean(1))) ** 2; ff = np.fft.rfftfreq(len(v), 1 / SR)
        e = np.array([S[(ff >= a) & (ff < b)].sum() for a, b in zip(edges[:-1], edges[1:])]); info[name] = dict(band_db=dict(zip(map(str, edges[:-1]), np.round(10 * np.log10(e / e.sum()), 1).tolist())), nan=bool(np.isnan(v).any()))
        sf.write(HERE / 'audition' / f'{name}.flac', v * (.8 / np.abs(v).max()), SR, subtype='PCM_16')
    print(json.dumps(info, indent=1)); (HERE / 'audition' / f'guitar_{which}_notes.json').write_text(json.dumps(dict(info, notes=len(rows), note='per-note render, stereo, engine room; not listened to by the assistant'), indent=1))


if __name__ == '__main__':
    dict(prep=prep, fit=fit, check=check, render=lambda: render(sys.argv[2] if len(sys.argv) > 2 else 'demo'))[sys.argv[1]]()
