"""Perform a MIDI file with the piano_9 chain: engine note -> per-note flow with decay correction -> mix -> the
same +11.7 dB low shelf below 250 Hz used for piano_9.  Also writes the engine-only mix of the same notes.
Reads the engine and MIDI, writes only to audition/.    python perform_midi.py <midi name> <seconds>
"""
import sys, os
sys.dont_write_bytecode = True
import json
from multiprocessing import Pool
import numpy as np
import soundfile as sf
import torch
import piano_decay as pd
from piano_clean import HERE, BASE, SR

SHELF_DB, TAIL = float(os.environ.get('PIANO_SHELF_DB', '11.7')), 3.0
MODEL, LABEL = os.environ.get('PIANO_MODEL', 'decay'), os.environ.get('PIANO_LABEL', 'flow_piano9_chain')
BASS_DB = float(os.environ['PIANO_BASS_DB']) if 'PIANO_BASS_DB' in os.environ else None
STATE = {}


def one(key):
    pitch, vel, dur = key
    if not STATE:
        sys.path.insert(0, str(BASE / 'grand_piano_improved/tools')); import render_model as rm
        torch.set_num_threads(1); d = np.load(HERE / 'data/piano_pairs.npz')
        STATE.update(piano=rm.GrandPiano(sample_rate=SR, room_mix=0., normalize=False, max_cache_mb=0), model=(__import__('piano_low').load() if MODEL == 'low' else pd.load()), low=(__import__('piano_low').load() if MODEL == 'hybrid' else None), se=float(d['engine_scale']), sr=float(d['real_scale']))
    x = STATE['piano'].render([(pitch, vel, 0., dur)], tail=TAIL).mean(1)
    y = pd.process(STATE['model'], x * STATE['se'], pitch, vel) / STATE['sr']
    if MODEL == 'hybrid':                       # low notes: new model; mid/high notes: the model the user approved (with its shelf); crossfade C3..G3
        w = float(np.clip((pitch - 48) / 7., 0., 1.)); f = np.fft.rfftfreq(len(y), 1 / SR); y = np.fft.irfft(np.fft.rfft(y) * (1 + (10 ** (11.7 / 20) - 1) / (1 + (f / 250.) ** 4)), len(y))
        low = pd.process(STATE['low'], x * STATE['se'], pitch, vel) / STATE['sr']
        if BASS_DB is not None:                   # put the engine's own fundamentals (< 150 Hz) back under the fixed low notes
            xs = x * STATE['se'] / STATE['sr']; fx = np.fft.rfftfreq(len(xs), 1 / SR); low = low + 10 ** (BASS_DB / 20) * np.fft.irfft(np.fft.rfft(xs) / (1 + (fx / float(os.environ.get('PIANO_BASS_HZ', '150'))) ** 4), len(xs))[:len(low)]
        y = w * y + (1 - w) * low
    return key, x.astype(np.float32), y.astype(np.float32)


def main(name, seconds):
    sys.path.insert(0, str(BASE / 'grand_piano_improved/tools')); import midi_events as me
    events, spans = me.load_events(BASE / 'grand_piano_improved/midi' / name); events, spans = me.clip_events(events, spans, seconds)
    notes = [(int(p), int(v), float(s), (int(p), int(v), max(.02, round(r - s, 2)))) for p, v, s, r in events]; keys = sorted({n[3] for n in notes})
    print('notes', len(notes), 'distinct renders', len(keys), flush=True); total = int((seconds + TAIL + 1) * SR); dry, wet = np.zeros(total), np.zeros(total); done = {}
    with Pool(6) as pool:
        for i, (key, x, y) in enumerate(pool.imap_unordered(one, keys, chunksize=4)):
            done[key] = (x, y)
            if i % 100 == 0: print(i, '/', len(keys), flush=True)
    for p, v, s, key in notes:
        x, y = done[key]; a = int(s * SR); n = min(len(x), total - a); dry[a:a + n] += x[:n]; n = min(len(y), total - a); wet[a:a + n] += y[:n]
    f = np.fft.rfftfreq(total, 1 / SR); wet = np.fft.irfft(np.fft.rfft(wet) * (1 + (10 ** (SHELF_DB / 20) - 1) / (1 + (f / 250.) ** 4)), total)
    stem = name.replace('.mid', ''); out = HERE / 'audition'; info = dict(midi=name, seconds=seconds, notes=len(notes), sustain_pedal_spans=len(spans), note='pedal spans are not applied here; notes are rendered one by one, dry, mono')
    edges = [20, 60, 120, 250, 500, 1000, 2000, 4000, 10000]
    for label, v in (('engine_only', dry), (LABEL, wet)):
        S = np.abs(np.fft.rfft(v)) ** 2; e = np.array([S[(f >= a) & (f < b)].sum() for a, b in zip(edges[:-1], edges[1:])])
        info[label] = dict(band_db=dict(zip(map(str, edges[:-1]), np.round(10 * np.log10(e / e.sum()), 1).tolist())), peak_before_normalising=float(np.abs(v).max()), nan=bool(np.isnan(v).any()))
        sf.write(out / f'{stem}_{int(seconds)}s_{label}.wav', v * (.8 / np.abs(v).max()), SR, subtype='PCM_24')
    (out / f'{stem}_{int(seconds)}s_{LABEL}_notes.json').write_text(json.dumps(info, indent=1)); print(json.dumps(info, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], float(sys.argv[2]))
