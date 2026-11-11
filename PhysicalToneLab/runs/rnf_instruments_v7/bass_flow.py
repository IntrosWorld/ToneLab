"""Electric bass (fingerstyle): per-note flow, engine DI note (bass_fea) -> real DI note.

Real data: IDMT-SMT-Bass (Zenodo 7188892, CC BY-NC-ND 4.0), PS/FS = fingerstyle normal notes, 3 basses x 3 pickup
settings x 4 strings x frets 0-12 (469 notes), fetched by fetch_bass.py.  Same per-note model and loss as the
approved guitar model.  Reads the engine, writes only in this folder.     python bass_flow.py prep | fit | check | render
"""
import sys
sys.dont_write_bytecode = True
import json, re
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
import guitar_flow as gf, piano_decay as pd, piano_clean as pc
from piano_clean import HERE, BASE, SR

gf.PAIRS = HERE / 'data/bass_pairs.npz'; ENG = BASE / 'electric_guitar_lead/distortion_guitar_fea/bass_fea'; SECONDS, OPEN, VEL = 2.5, [28, 33, 38, 43], 108
EDGES = [35, 70, 140, 280, 560, 1120, 2240, 4480, 9000]


class BassFlow(pc.CleanFlow):
    """Per-note gain + decay-rate correction.  Unlike piano/guitar the note may be lengthened: the engine's partials
    decay at ln(1000)/T60 with T60 = 3.05 (110/f)^0.32 s, and the correction may remove up to 90 % of that rate
    (so every partial still decays).  Engine bass notes are pure synthesis, so there is no noise to bring up."""
    def __init__(self):
        super().__init__(eq_only=True); self.rate = pc.mlp(4, 32, 1); torch.nn.init.zeros_(self.rate[-1].weight); torch.nn.init.zeros_(self.rate[-1].bias)
        self.register_buffer('floor', -.9 * 6.908 / (3.05 * (110 / pc.FREQ.clamp(40, None)) ** .32))

    def forward(self, fr, midi, vel, age):
        x = self.features(midi, vel); rate = torch.maximum(self.rate(x)[..., 0], self.floor[None]).clamp(max=6)
        G = self.gain(x)[..., 0].clamp(-4.5, 4.5) - rate * age[:, None]
        return torch.fft.irfft(torch.fft.rfft(fr) * pc.BAND * torch.exp(G.clamp(-12, 6)), pc.N)


gf.MODEL, gf.FLOOR = BassFlow, 6.9              # bass: keep bands down to 60 dB below the strongest in the loss


def engine():
    sys.path.insert(0, str(ENG)); import bass_model as bm
    return bm, json.loads((ENG / 'bass_fea_modes.json').read_text(encoding='utf-8'))          # existing mode file, read only


def engine_note(bm, model, midi, string, duration, seed=2401):
    x = bm.synth_note(bm.BassNote(0., duration, int(midi), VEL, int(string)), model, np.random.default_rng(seed)); return resample_poly(x, 147, 160)       # 48 kHz -> 44.1 kHz


def cut(x, L):
    o = max(0, int(np.argmax(np.abs(x) > .1 * np.abs(x).max())) - 256); x = x[o:o + L]; return np.pad(x, (0, L - len(x)))


def prep():
    bm, model = engine(); L = int(SECONDS * SR); E, R, M, S, B = [], [], [], [], []
    for f in sorted((HERE / 'data/bass').glob('*.wav')):
        m = re.match(r'BS_(\d)_EQ_(\d)_FS_NO_(\d)_(\d+)\.wav', f.name); x, sr = sf.read(f, dtype='float64', always_2d=True)
        if not m or sr != SR: continue
        string, fret = int(m.group(3)) - 1, int(m.group(4))
        if string > 3: continue                                  # one stray file labelled string 5
        midi = OPEN[string] + fret
        E.append(cut(engine_note(bm, model, midi, string, SECONDS), L)); R.append(cut(x.mean(1), L)); M.append(midi); S.append(string); B.append(int(m.group(1)) * 10 + int(m.group(2)))
    E, R = np.stack(E), np.stack(R); se, sr_ = .1 / np.sqrt(np.mean(E ** 2)), .1 / np.sqrt(np.mean(R ** 2))
    np.savez(gf.PAIRS, E=(E * se).astype(np.float32), R=(R * sr_).astype(np.float32), midi=np.array(M), vel=np.full(len(M), VEL), string=np.array(S), bass_eq=np.array(B), length=np.full(len(M), SECONDS), engine_scale=se, real_scale=sr_)
    print('notes', len(M), 'pitch range', min(M), max(M))


def check():
    E, R, M, V = gf.matched(); m = gf.load('bass_flow'); W = int(.3 * SR); f = np.fft.rfftfreq(W, 1 / SR); held = set(sorted(set(M.tolist()))[2::4]); out = {}; hi = [i for i in range(len(M)) if M[i] in held]
    Y = {i: pd.process(m, E[i].astype(np.float64), int(M[i]), VEL) for i in hi}

    def prof(x, t0):
        a = int(t0 * SR); S = np.abs(np.fft.rfft(x[a:a + W] * np.hanning(W))) ** 2; return np.array([10 * np.log10(S[(f >= lo) & (f < up)].sum() + 1e-12) for lo, up in zip(EDGES[:-1], EDGES[1:])])
    dec = lambda x: 20 * np.log10(np.sqrt((x[int(1.5 * SR):int(1.9 * SR)] ** 2).mean() + 1e-12) / np.sqrt((x[int(.05 * SR):int(.45 * SR)] ** 2).mean() + 1e-12))
    for t0 in (0., 1.0):
        de, dm = [], []
        for i in hi:
            pr = prof(R[i], t0); ok = (np.array(EDGES[:-1]) >= .8 * 440 * 2 ** ((M[i] - 69) / 12)) & (pr > pr.max() - 50)
            de.append(np.where(ok, prof(E[i], t0) - pr, np.nan)); dm.append(np.where(ok, prof(Y[i], t0) - pr, np.nan))
        de, dm = np.array(de), np.array(dm)
        out[f't={t0}'] = dict(engine_minus_real=np.round(np.nanmean(de, 0), 1).tolist(), model_minus_real=np.round(np.nanmean(dm, 0), 1).tolist(), engine_mean_abs=round(float(np.nanmean(np.abs(de))), 2), model_mean_abs=round(float(np.nanmean(np.abs(dm))), 2), held_notes=len(de))
    out['level_change_0.25_to_1.7s_db'] = dict(engine=round(float(np.median([dec(E[i]) for i in hi])), 1), model=round(float(np.median([dec(Y[i]) for i in hi])), 1), real=round(float(np.median([dec(R[i]) for i in hi])), 1))
    out['bands_from_hz'] = EDGES[:-1]; print(json.dumps(out)); (HERE / 'bass_summary.json').write_text(json.dumps(out, indent=1))


def render():
    """tab2 bass line through the engine's own body/cabinet/compressor chain, with and without the per-note flow."""
    bm, model = engine(); d = np.load(gf.PAIRS); se, sr_ = float(d['engine_scale']), float(d['real_scale']); flow = gf.load('bass_flow'); score = json.loads((ENG / 'tab2_bass_score.json').read_text(encoding='utf-8'))
    notes = [bm.BassNote(e['start'], e['duration'], int(e['midi']), int(e['velocity'])) for e in score['events']]; seconds = float(score['duration_s']); original = bm.synth_note; info = {}

    def with_flow(note, mdl, rng):
        x = original(note, mdl, rng); y = pd.process(flow, resample_poly(x, 147, 160) * se, note.midi, VEL) / sr_; y = resample_poly(y, 160, 147)[:len(x)]; return np.pad(y, (0, len(x) - len(y)))
    (HERE / 'audition').mkdir(exist_ok=True)
    for name, fn in (('bass_tab2_1_engine_only', original), ('bass_tab2_2_flow', with_flow)):
        bm.synth_note = fn                                                   # in-memory only; the engine file is not touched
        y = bm.render_bass(model, notes, seconds); bm.synth_note = original; sf.write(HERE / 'audition' / f'{name}.flac', y, bm.SR, subtype='PCM_16')
        S = np.abs(np.fft.rfft(y.mean(1))) ** 2; f = np.fft.rfftfreq(len(y), 1 / bm.SR); e = np.array([S[(f >= a) & (f < b)].sum() for a, b in zip(EDGES[:-1], EDGES[1:])])
        info[name] = dict(band_db=dict(zip(map(str, EDGES[:-1]), np.round(10 * np.log10(e / e.sum()), 1).tolist())), nan=bool(np.isnan(y).any()))
    print(json.dumps(info)); (HERE / 'audition/bass_tab2_notes.json').write_text(json.dumps(dict(info, notes=len(notes), note='engine body/cabinet/compressor chain applied after the per-note flow; not listened to by the assistant'), indent=1))


if __name__ == '__main__':
    dict(prep=prep, check=check, render=render, fit=lambda: gf.fit(name='bass_flow'))[sys.argv[1]]()
