"""Clean electric guitar: per-note flow, engine DI note (tone_lab CleanGuitar, bridge pickup) -> real clean Stratocaster note.

Real data: EGFxSet Clean, bridge pickup, 138 notes (6 strings x frets 0-22), CC BY 4.0, already in data/egfx/Clean.
Same per-note model and loss as the approved steel-string guitar model (phase-preserving, cannot create frequencies).
Reads the engine, writes only in this folder.     python electric_clean_flow.py prep | fit | check | render
"""
import sys
sys.dont_write_bytecode = True
import json
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import guitar_flow as gf, piano_decay as pd
from piano_clean import HERE, BASE, SR

gf.PAIRS = HERE / 'data/electric_clean_pairs.npz'; ENG = BASE / 'electric_guitar_lead/distortion_guitar_fea'; SECONDS, VEL, TUNING = 2.5, 100, (40, 45, 50, 55, 59, 64)
EDGES = [70, 140, 280, 560, 1120, 2240, 4480, 9000]; NAME = 'electric_clean_flow'


def engine():
    sys.path.insert(0, str(ENG)); from tone_lab import guitar as tg, dsp
    from tone_lab.score import load_guitar
    from render_solo import Gesture
    return tg, dsp, load_guitar, Gesture, dict(tg.PRESETS['clean_glass'], pickup=[1., 0., 0.])          # bridge pickup only, like the recordings


def cut(x, L):
    o = max(0, int(np.argmax(np.abs(x) > .1 * np.abs(x).max())) - 256); x = x[o:o + L]; return np.pad(x, (0, L - len(x)))


def prep():
    tg, dsp, _, Gesture, preset = engine(); g = tg.CleanGuitar(TUNING); L = int(SECONDS * SR); E, R, M, S = [], [], [], []; assert dsp.SR == 48000
    for f in sorted((HERE / 'data/egfx/Clean').glob('*.wav')):
        string, fret = map(int, f.stem.split('-')); idx = 6 - string; midi = TUNING[idx] + fret                 # EGFxSet string 1 = high E
        x, sr = sf.read(f, dtype='float64', always_2d=True); x = cut(resample_poly(x.mean(1), 147, 160) if sr == 48000 else x.mean(1), L)
        if f.stem in ('1-0', '6-0', '3-12'):                                                             # check the naming once per region
            S_ = np.abs(np.fft.rfft(x[2000:2000 + 32768] * np.hanning(32768))); fr = np.fft.rfftfreq(32768, 1 / SR); f0 = 440 * 2 ** ((midi - 69) / 12); pk = fr[(fr > .7 * f0) & (fr < 1.4 * f0)][np.argmax(S_[(fr > .7 * f0) & (fr < 1.4 * f0)])]
            print(f.stem, 'expected', round(f0, 1), 'measured', round(float(pk), 1)); assert abs(1200 * np.log2(pk / f0)) < 60
        e, _ = g.note(Gesture(0., SECONDS, float(midi), None, VEL, 'pick', 0., idx), preset, np.random.default_rng(2401))
        E.append(cut(resample_poly(e, 147, 160), L)); R.append(x); M.append(midi); S.append(idx)
    E, R = np.stack(E), np.stack(R); se, sr_ = .1 / np.sqrt(np.mean(E ** 2)), .1 / np.sqrt(np.mean(R ** 2))
    np.savez(gf.PAIRS, E=(E * se).astype(np.float32), R=(R * sr_).astype(np.float32), midi=np.array(M), vel=np.full(len(M), VEL), string=np.array(S), length=np.full(len(M), SECONDS), engine_scale=se, real_scale=sr_)
    print('notes', len(M), 'pitch range', min(M), max(M))


def check():
    E, R, M, V = gf.matched(); m = gf.load(NAME); W = int(.3 * SR); f = np.fft.rfftfreq(W, 1 / SR); held = set(sorted(set(M.tolist()))[2::4]); out = {}; hi = [i for i in range(len(M)) if M[i] in held]
    Y = {i: pd.process(m, E[i].astype(np.float64), int(M[i]), VEL) for i in hi}

    def prof(x, t0):
        a = int(t0 * SR); S = np.abs(np.fft.rfft(x[a:a + W] * np.hanning(W))) ** 2; return np.array([10 * np.log10(S[(f >= lo) & (f < up)].sum() + 1e-12) for lo, up in zip(EDGES[:-1], EDGES[1:])])
    dec = lambda x: 20 * np.log10(np.sqrt((x[int(1.5 * SR):int(1.9 * SR)] ** 2).mean() + 1e-12) / np.sqrt((x[int(.05 * SR):int(.45 * SR)] ** 2).mean() + 1e-12))
    for t0 in (0., 1.0):
        de, dm = [], []
        for i in hi:
            pr = prof(R[i], t0); ok = (np.array(EDGES[1:]) >= .8 * 440 * 2 ** ((M[i] - 69) / 12)) & (pr > pr.max() - 40)
            de.append(np.where(ok, prof(E[i], t0) - pr, np.nan)); dm.append(np.where(ok, prof(Y[i], t0) - pr, np.nan))
        de, dm = np.array(de), np.array(dm)
        out[f't={t0}'] = dict(engine_minus_real=np.round(np.nanmean(de, 0), 1).tolist(), model_minus_real=np.round(np.nanmean(dm, 0), 1).tolist(), engine_mean_abs=round(float(np.nanmean(np.abs(de))), 2), model_mean_abs=round(float(np.nanmean(np.abs(dm))), 2), held_notes=len(de))
    out['level_change_0.25_to_1.7s_db'] = dict(engine=round(float(np.median([dec(E[i]) for i in hi])), 1), model=round(float(np.median([dec(Y[i]) for i in hi])), 1), real=round(float(np.median([dec(R[i]) for i in hi])), 1))
    out['bands_from_hz'] = EDGES[:-1]; print(json.dumps(out)); (HERE / 'electric_clean_summary.json').write_text(json.dumps(out, indent=1))


def render_di(seconds=20.):
    """GP1 lead as bridge-pickup DI at 48 kHz: engine only, and with the per-note flow.  Returns (dsp, guitar module, engine DI, flow DI)."""
    tg, dsp, load_guitar, Gesture, preset = engine(); events, report = load_guitar(ENG / 'ref/tab1.gp', 0, seconds); d = np.load(gf.PAIRS); se = float(d['engine_scale']); flow = gf.load(NAME)
    tg.PRESETS['bridge_di'] = preset                                             # in-memory entry only; engine files are not touched
    g = tg.CleanGuitar(report['tuning']); total = max(e.start + e.duration for e in events) + 3.2; plain = g.render_di(events, 'bridge_di', total); original = g.note

    def with_flow(event, *a, **k):
        x, phase = original(event, *a, **k); y = pd.process(flow, resample_poly(x, 147, 160) * se, int(round(event.midi)), VEL) / se; y = resample_poly(y, 160, 147)[:len(x)]; return np.pad(y, (0, len(x) - len(y))), phase
    g.note = with_flow; flowed = g.render_di(events, 'bridge_di', total); g.note = original
    return dsp, g, plain, flowed, len(events)


def render():
    dsp, g, plain, flowed, n = render_di(); (HERE / 'audition').mkdir(exist_ok=True); info = {}
    for name, di in (('electric_clean_1_engine_only', plain), ('electric_clean_2_flow', flowed)):
        y = g.process(di, 'clean_glass'); y = y * (.89 / np.abs(y).max()); sf.write(HERE / 'audition' / f'{name}.flac', y, dsp.SR, subtype='PCM_16')
        S = np.abs(np.fft.rfft(y.mean(1))) ** 2; f = np.fft.rfftfreq(len(y), 1 / dsp.SR); e = np.array([S[(f >= a) & (f < b)].sum() for a, b in zip(EDGES[:-1], EDGES[1:])])
        info[name] = dict(band_db=dict(zip(map(str, EDGES[:-1]), np.round(10 * np.log10(e / e.sum()), 1).tolist())), nan=bool(np.isnan(y).any()))
    print(json.dumps(info)); (HERE / 'audition/electric_clean_notes.json').write_text(json.dumps(dict(info, notes=n, note='GP1 lead, bridge-pickup DI; engine clean amp chain (clean_glass) and room applied after the per-note flow; each file peak-normalised; not listened to by the assistant'), indent=1))


if __name__ == '__main__':
    dict(prep=prep, check=check, render=render, fit=lambda: gf.fit(name=NAME))[sys.argv[1]]()
