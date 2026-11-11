"""Thedooo - Ascend (Guitar Pro tab), first 120 s, every track through its physical model + learned stage.

  track 0  Distortion Guitar (lead)    tone_lab CleanGuitar DI -> clean per-note flow -> resonant-flow amp (v6 weights)
  track 3  Distortion Guitar (rhythm)  same chain
  track 1,2 Piano right / left hand    grand_piano_improved -> approved piano chain H (perform_midi2)
  track 4  12-string steel guitar      acoustic_guitar_fea -> steel guitar flow  (6-string engine; octave strings added, see steel())
  track 5  Drums                       drum_set_fea -> per-hit drum flow (kick, snare, hi-hat; crash and tom as the engine makes them)
There is no bass track in this tab.  Engines and models are read only; everything is written to this folder.
Mix levels, panning, rooms and the safety steps around the amp are hand settings, not learned.
  python render_song.py piano | steel | guitars | drums | mix
"""
import sys, os
sys.dont_write_bytecode = True
os.environ.update(PIANO_LOW_CKPT='piano_low_attack', PIANO_AIR_DB='12', PIANO_AIR_HZ='2000', PIANO_BASS_DB='15', PIANO_BASS_HZ='300')        # approved piano chain H
from pathlib import Path
import json
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent; RUNS = HERE.parent; V7 = RUNS / 'rnf_instruments_v7'; V6 = RUNS / 'rnf_resonant_amp_v6'; BASE = RUNS.parents[1]
ENG = BASE / 'electric_guitar_lead/distortion_guitar_fea'; GP = HERE / 'tab/Ascend.gp'
SECONDS, TAIL, OUT_SR = 120., 4., 48000; STEMS = HERE / 'stems'; STEMS.mkdir(exist_ok=True); TOTAL = int((SECONDS + TAIL) * OUT_SR)
sys.path[:0] = [str(V7), str(ENG), str(ENG / 'tools')]


def notes_of(track):
    from render_gp_score import parse_gpif, merge_ties, DYNAMICS
    return [dict(e, velocity=int(DYNAMICS.get(e['dynamic'], 96))) for e in merge_ties(parse_gpif(GP, track, SECONDS)[2])]


def save(name, y, sr):
    y = np.asarray(y, dtype=np.float64); y = resample_poly(y, OUT_SR, sr, axis=0) if sr != OUT_SR else y; y = y[:TOTAL]; y = np.pad(y, ((0, TOTAL - len(y)), (0, 0))); assert not np.isnan(y).any()
    np.save(STEMS / f'{name}.npy', y.astype(np.float32)); print(name, 'saved; peak', round(float(np.abs(y).max()), 3), 'rms', round(float(np.sqrt((y ** 2).mean())), 4), flush=True)


BASS_DETUNE = float(os.environ.get('PIANO_BASS_DETUNE', .2))


def calm_bass():
    """Piano engine setting changed IN MEMORY for this render (engine files untouched), after the user heard a vibration in
    held bass notes.  Measured: on a 3 s G1 the engine's partials 2-6 swing by 20-40 dB (deep nulls) because its two
    unison strings are detuned by ~1.9 cents in the bass.  Detune is scaled by BASS_DETUNE below C3, back to the engine's
    own value by G3.  Hand setting, not learned."""
    sys.path.insert(0, str(BASE / 'grand_piano_improved/tools')); import piano_physics as phys
    if getattr(phys, '_calmed', False): return
    original = phys.unison_detune_cents; phys._calmed = True
    phys.unison_detune_cents = lambda midi, index, count: original(midi, index, count) * (BASS_DETUNE + (1 - BASS_DETUNE) * min(1., max(0., (midi - 48) / 7.)))


def one_calm(key):
    """Same for the engine's second beating source: the two polarisations of each string (detune ratio 4e-4)."""
    import perform_midi2 as pm
    calm_bass(); import piano_physics as phys
    if not hasattr(phys, '_pol'): phys._pol = phys.POLARISATION_DETUNE_RATIO
    phys.POLARISATION_DETUNE_RATIO = phys._pol * (BASS_DETUNE + (1 - BASS_DETUNE) * min(1., max(0., (key[0] - 48) / 7.))); return pm.one(key)


def piano():
    from multiprocessing import Pool
    import perform_midi2 as pm
    sys.path.insert(0, str(BASE / 'grand_piano_improved/tools')); import render_model as rm
    SR = pm.SR; ev = [e for t in (1, 2) for e in notes_of(t)]
    if os.environ.get('PIANO_PEDAL', '1') == '1':
        # Sustain pedal (asked for by the user; the tab has no pedal marks).  Hand rule: the pedal is changed at every new
        # left-hand chord (3 s apart here) and every note rings until the next change; the last pedal is held to the end of
        # the last written note.  Done by lengthening the notes, so the engine's own string decay and dampers do the rest.
        # Not modelled: other strings ringing in sympathy while the dampers are up.
        left = sorted({round(e['start'], 2) for e in notes_of(2)}); change = [left[0]] + [t for a, t in zip(left, left[1:]) if t - a > 1.]; change.append(max(e['start'] + e['duration'] for e in ev)); longer = 0
        for e in ev:
            nxt = min([c for c in change if c > e['start'] + .05], default=change[-1]); longer += nxt > e['start'] + e['duration'] + .02; e['duration'] = max(e['duration'], nxt - e['start'])
        print('pedal changes at', np.round(change, 2).tolist(), '; notes lengthened', longer, 'of', len(ev), flush=True)
    notes = [(e['start'], (int(e['midi']), e['velocity'], max(.02, round(e['duration'], 2)))) for e in ev]; keys = sorted({n[1] for n in notes}); total = int((SECONDS + TAIL) * SR); mix = np.zeros((total, 2)); done = {}
    print('piano notes', len(notes), 'distinct renders', len(keys), flush=True)
    with Pool(2, initializer=calm_bass) as pool:
        for key, y in pool.imap_unordered(one_calm, keys, chunksize=2): done[key] = y
    for s, key in notes:
        y = done[key]; a = int(s * SR); n = min(len(y), total - a); mix[a:a + n] += y[:n]
    mix = rm.GrandPiano(sample_rate=SR, room_mix=0., normalize=False)._room(mix, mix=pm.ROOM); save('piano', mix, SR)


def steel():
    """12-string: the engine is a 6-string, so each note on the four low courses also gets its octave string (+12, 8 ms later, -3 dB)
    and each note on the two high courses a second unison string 8 ms later (-3 dB).  Arrangement step, not learned."""
    import guitar_flow as gf, piano_decay as pd
    sys.path.insert(0, str(BASE / 'acoustic_guitar_fea/tools')); import acoustic_engine as ae
    SR = gf.SR; d = np.load(gf.PAIRS); se, sr_ = float(d['engine_scale']), float(d['real_scale']); model = gf.load(); guitar = ae.AcousticGuitar(sample_rate=SR); ev = notes_of(4); total = int((SECONDS + TAIL) * SR); mix = np.zeros((total, 2)); cache = {}; moved = 0
    def one(midi, vel, dur):
        k = (midi, vel, round(dur, 2))
        if k not in cache:
            x = guitar.render([dict(midi=midi, velocity=vel, start=0., duration=dur)], tail=3., normalize=False, room=False); cache[k] = np.stack([pd.process(model, x[:, c] * se, midi, vel) / sr_ for c in range(2)], 1)
        return cache[k]
    for e in ev:
        midi, low = int(e['midi']), e['string'] <= 3                                   # GP string 0 = lowest course
        if midi < 40: midi += 12; moved += 1                                           # engine range starts at E2; this tab's low course is tuned to C2
        for off, m, g in ((0., midi, 1.), (.008, midi + 12 if low else midi, .7)):
            y = one(m, e['velocity'], e['duration']) * g; a = int((e['start'] + off) * SR); n = min(len(y), total - a); mix[a:a + n] += y[:n]
    print('steel notes', len(ev), 'distinct renders', len(cache), 'notes below E2 moved up an octave', moved, flush=True); save('steel', guitar._room(mix) if hasattr(guitar, '_room') else mix, SR)


def guitars():
    import torch
    import electric_clean_flow as ecf, guitar_flow as gf, piano_decay as pd
    sys.path[:0] = [str(V6), str(V7 / 'tones')]; import render_audio as ra, extract, render_rat_song3 as rs3
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    tg, dsp, load_guitar, Gesture, preset = ecf.engine(); tg.PRESETS['bridge_di'] = preset; flow = gf.load(ecf.NAME); se = float(np.load(gf.PAIRS)['engine_scale']); ra.HERE = V6; amp = ra.load(); SR = ra.SR
    man = [r for r in json.loads((extract.PAIRS / 'manifest.json').read_text())['pairs'] if r['split'] == 'train'][:6]; fr = lambda v: (lambda r: r[r > .1 * r.max()])(np.sqrt((v[:len(v) // 4096 * 4096].reshape(-1, 4096) ** 2).mean(1)))
    tx = np.concatenate([np.load(extract.PAIRS / r['file'])['x'].astype(np.float64) for r in man]); ty = np.concatenate([np.load(extract.PAIRS / r['file'])['y'].astype(np.float64) for r in man]); ref = float(np.percentile(fr(tx), 75)); top = float(np.abs(ty).max()); info = dict(amp_training_input_level_p75=round(ref, 4), amp_training_output_peak=round(top, 3))
    for name, track in (('guitar_lead', 0), ('guitar_rhythm', 3)):
        events, report = load_guitar(GP, track, SECONDS); g = tg.CleanGuitar(report['tuning']); original = g.note
        def with_flow(event, *a, **k):
            x, phase = original(event, *a, **k); y = pd.process(flow, resample_poly(x, 147, 160) * se, int(round(event.midi)), ecf.VEL) / se; y = resample_poly(y, 160, 147)[:len(x)]; return np.pad(y, (0, len(x) - len(y))), phase
        g.note = with_flow; di = g.render_di(events, 'bridge_di', SECONDS + TAIL); g.note = original; x = resample_poly(di, 147, 160); x = x * ref / float(np.percentile(fr(x), 75))        # amp input at its training level
        y, used = rs3.process(amp, x); pk = float(np.abs(y).max()); env = uniform_filter1d(maximum_filter1d(np.abs(y), 257), 129); y = y * np.minimum(1., top / np.maximum(env, 1e-9))
        info[name] = dict(notes=len(events), tuning=report['tuning'], frames_by_input_step_0_to_minus24dB=used.tolist(), amp_peak_before_limit=round(pk, 2), amp_peak_after=round(float(np.abs(y).max()), 2))
        print(name, info[name], flush=True); sf.write(STEMS / f'{name}_DI_before_amp.flac', x * (.5 / np.abs(x).max()), SR, subtype='PCM_16'); save(name, dsp.room(dsp.stereo(resample_poly(y, 160, 147)), wet=.10, t60=.8), OUT_SR)
    (HERE / 'guitars_notes.json').write_text(json.dumps(info, indent=1))


def drums():
    import drum_flow as df
    sys.path.insert(0, str(ENG / 'drum_set_fea')); import render_gp_drums as rd
    dm, model = df.engine(); flow = df.load(); s = float(np.load(df.PAIRS)['scale']); name, tempo, length, hits, mapping = rd.parse_drum_track(GP, 5); hits = [h for h in hits if h.start < SECONDS]; original = dm.synth_hit
    import drum_snare_v2 as sv; snare = sv.load(); ref = resample_poly(original(dm.DrumHit(0., 'snare', 108, 0.), model, np.random.default_rng(5)), 147, 160) * s      # snare: retrained model (drum_snare_v2.py)
    snare_gain = float(np.abs(df.process(flow, ref, 'snare')).max() / np.abs(df.process(snare, ref, 'snare')).max()); print('snare v2 gain (hand rule: same peak as the first snare model)', round(snare_gain, 2), flush=True); extra = json.loads((ENG / 'drum_set_fea/tab3_drums_fea_modes.json').read_text(encoding='utf-8'))            # has the toms
    import crash_flow as cf; tm, ckit = cf.kit(); crash = cf.load(); cs = float(np.load(cf.PAIRS)['scale']); old = original(dm.DrumHit(0., 'crash', 108, 0.), model, np.random.default_rng(1)); import crash_ring as cr; new = resample_poly(cr.hit(tm, ckit, crash, cs), 160, 147)[:len(old)]; print('crash seconds available', len(old) / dm.SR, flush=True)
    crash_gain = .7 * float(np.sqrt((old ** 2).mean() / (new ** 2).mean())); print('crash gain (hand rule: 3 dB under the old crash by RMS)', round(crash_gain, 3), flush=True)
    def with_flow(hit, mdl, rng, maximum_tail=None):
        x = original(hit, mdl if hit.piece in mdl['pieces'] else extra, rng, maximum_tail)
        if hit.piece == 'crash':                                             # new source (tone_lab crash) + crash flow, see crash_flow.py
            c = resample_poly(cr.hit(tm, ckit, crash, cs, hit.velocity, seed=int(rng.integers(1 << 30))), 160, 147) * crash_gain;     # + measured ring partials, see crash_ring.py y = np.zeros(len(x)); n = min(len(x), len(c)); y[:n] = c[:n]; fade = min(int(.05 * dm.SR), n); y[n - fade:n] *= np.linspace(1, 0, fade); return y
        if hit.piece not in df.ORDER: return x
        y = resample_poly(df.process(snare if hit.piece == 'snare' else flow, resample_poly(x, 147, 160) * s, hit.piece) / s, 160, 147)[:len(x)] * (snare_gain if hit.piece == 'snare' else 1.); y = np.pad(y, (0, len(x) - len(y))); fade = min(int(.01 * dm.SR), len(y)); y[-fade:] *= np.linspace(1, 0, fade); return y
    dm.synth_hit = with_flow; y = dm.render_drums(model, hits, SECONDS + TAIL); dm.synth_hit = original; print('drum hits', len(hits), {k[2]: v for k, v in mapping.items()}, flush=True); save('drums', y, dm.SR)


def mix():
    """Hand mix: each stem set to a level while it plays, panned, summed, 3 ms look-ahead limiter."""
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    plan = dict(piano=(-19., 0.), steel=(-21., 0.), guitar_lead=(-19., .25), guitar_rhythm=(-22., -.35), drums=(-18., 0.)); out = np.zeros((TOTAL, 2)); info = {}
    for name, (level, pan) in plan.items():
        y = np.load(STEMS / f'{name}.npy').astype(np.float64); r = np.sqrt((y[:TOTAL // 4800 * 4800].reshape(-1, 4800, 2) ** 2).mean((1, 2))); act = float(np.sqrt((r[r > .1 * r.max()] ** 2).mean())); y = y * 10 ** (level / 20) / act
        y = y * np.array([min(1., 1 - pan), min(1., 1 + pan)]); out += y; info[name] = dict(level_db=level, pan=pan, plays_seconds=[round(float(np.argmax(r > .1 * r.max()) * .1), 1), round(float((len(r) - np.argmax(r[::-1] > .1 * r.max())) * .1), 1)])
        sf.write(STEMS / f'{name}.flac', y * min(1., .98 / np.abs(y).max()), OUT_SR, subtype='PCM_16')
    env = uniform_filter1d(maximum_filter1d(np.abs(out).max(1), 289), 145); pk = float(np.abs(out).max()); out = out * np.minimum(1., .89 / np.maximum(env, 1e-9))[:, None]; out = np.clip(out, -.99, .99)
    f = np.fft.rfftfreq(TOTAL, 1 / OUT_SR); S = np.abs(np.fft.rfft(out.mean(1))) ** 2; edges = [30, 60, 120, 250, 500, 1000, 2000, 4000, 8000, 16000]; e = np.array([S[(f >= a) & (f < b)].sum() for a, b in zip(edges[:-1], edges[1:])])
    r = np.sqrt((out[:TOTAL // 48000 * 48000].reshape(-1, 48000, 2) ** 2).mean((1, 2)))
    info['mix'] = dict(peak_before_limiter=round(pk, 2), limited_fraction=round(float((env > .89).mean()), 4), band_db=dict(zip(map(str, edges[:-1]), np.round(10 * np.log10(e / e.sum()), 1).tolist())), rms_per_10s=np.round(r[:120].reshape(-1, 10).mean(1), 3).tolist(), note='hand mix; not listened to by the assistant')
    sf.write(HERE / 'ascend_first_120s_full_mix.flac', out, OUT_SR, subtype='PCM_16'); print(json.dumps(info, indent=1)); (HERE / 'mix_notes.json').write_text(json.dumps(info, indent=1))


if __name__ == '__main__':
    dict(piano=piano, steel=steel, guitars=guitars, drums=drums, mix=mix)[sys.argv[1]]()
