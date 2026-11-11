"""Perform a MIDI file, stereo, with listening settings on top of the trained per-note models.

Per note (each stereo channel separately, same gains, so the engine's stereo image is kept):
  notes below C3: piano_low model;  above G3: piano_decay model + its low shelf;  crossfade between
  + engine content below BASS_HZ at BASS_DB under the low notes          (setting, not learned)
  + engine content above AIR_HZ at AIR_DB on every note                   (setting, not learned)
After mixing: the engine's own room (GrandPiano._room) at ROOM mix.
Reads the engine and MIDI; writes only to audition/.   python perform_midi2.py <midi> <seconds> <label>
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

E = lambda k, d: float(os.environ.get(k, d))
BASS_DB, BASS_HZ, AIR_DB, AIR_HZ, ROOM, TAIL = E('PIANO_BASS_DB', 14), E('PIANO_BASS_HZ', 220), E('PIANO_AIR_DB', 6), E('PIANO_AIR_HZ', 2500), E('PIANO_ROOM', .22), 3.0
STATE = {}


def one(key):
    pitch, vel, dur = key
    if not STATE:
        sys.path.insert(0, str(BASE / 'grand_piano_improved/tools')); import render_model as rm, piano_low
        torch.set_num_threads(1); d = np.load(HERE / 'data/piano_pairs.npz')
        STATE.update(piano=rm.GrandPiano(sample_rate=SR, room_mix=0., normalize=False, max_cache_mb=0), hi=pd.load(), lo=piano_low.load(), se=float(d['engine_scale']), sr=float(d['real_scale']))
    x2 = STATE['piano'].render([(pitch, vel, 0., dur)], tail=TAIL); w = float(np.clip((pitch - 48) / 7., 0., 1.)); out = []
    for c in range(2):
        x = x2[:, c] * STATE['se']; n = len(x); f = np.fft.rfftfreq(n, 1 / SR); X = np.fft.rfft(x); filt = lambda H: np.fft.irfft(X * H, n)
        hi = pd.process(STATE['hi'], x, pitch, vel); hi = np.fft.irfft(np.fft.rfft(hi) * (1 + (10 ** (11.7 / 20) - 1) / (1 + (f / 250.) ** 4)), n)
        lo = pd.process(STATE['lo'], x, pitch, vel) + 10 ** (BASS_DB / 20) * filt(1 / (1 + (f / BASS_HZ) ** 4))
        air = 10 ** (AIR_DB / 20) * filt(1 - 1 / (1 + (f / AIR_HZ) ** 4))
        out.append((w * hi + (1 - w) * lo + w * air) / STATE['sr'])            # no added highs on low notes
    return key, np.stack(out, 1).astype(np.float32)


def main(name, seconds, label):
    sys.path.insert(0, str(BASE / 'grand_piano_improved/tools')); import midi_events as me, render_model as rm
    events, spans = me.load_events(BASE / 'grand_piano_improved/midi' / name); events, spans = me.clip_events(events, spans, seconds)
    notes = [(float(s), (int(p), int(v), max(.02, round(r - s, 2)))) for p, v, s, r in events]; keys = sorted({n[1] for n in notes}); total = int((seconds + TAIL + 1) * SR); mix = np.zeros((total, 2)); done = {}
    print('notes', len(notes), 'distinct renders', len(keys), flush=True)
    with Pool(6) as pool:
        for key, y in pool.imap_unordered(one, keys, chunksize=2): done[key] = y
    for s, key in notes:
        y = done[key]; a = int(s * SR); n = min(len(y), total - a); mix[a:a + n] += y[:n]
    if ROOM > 0: mix = rm.GrandPiano(sample_rate=SR, room_mix=0., normalize=False)._room(mix, mix=ROOM)
    mono = mix.mean(1); f = np.fft.rfftfreq(total, 1 / SR); S = np.abs(np.fft.rfft(mono)) ** 2; edges = [20, 60, 120, 250, 500, 1000, 2000, 4000, 10000]; e = np.array([S[(f >= a) & (f < b)].sum() for a, b in zip(edges[:-1], edges[1:])])
    info = dict(midi=name, seconds=seconds, settings=dict(bass_db=BASS_DB, bass_hz=BASS_HZ, air_db=AIR_DB, air_hz=AIR_HZ, room_mix=ROOM), band_db=dict(zip(map(str, edges[:-1]), np.round(10 * np.log10(e / e.sum()), 1).tolist())),
                left_right_correlation=round(float(np.corrcoef(mix[:, 0], mix[:, 1])[0, 1]), 3), nan=bool(np.isnan(mix).any()), note='bass, air and room are listening settings, not learned; not listened to by the assistant')
    stem = f"{name.replace('.mid', '')}_{int(seconds)}s_{label}"; sf.write(HERE / 'audition' / f'{stem}.wav', mix * (.8 / np.abs(mix).max()), SR, subtype='PCM_24')
    (HERE / 'audition' / f'{stem}_notes.json').write_text(json.dumps(info, indent=1)); print(json.dumps(info))


if __name__ == '__main__':
    main(sys.argv[1], float(sys.argv[2]), sys.argv[3])
