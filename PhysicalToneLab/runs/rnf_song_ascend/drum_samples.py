"""One file per drum piece, exactly as used in the first Ascend render (drums_v1): kick, snare, hi-hat through the per-hit
flow (first snare model), crash and tom straight from the engine.  Four hits each.   python drum_samples.py"""
import sys, json; sys.dont_write_bytecode = True
import numpy as np, soundfile as sf
from scipy.signal import resample_poly
import render_song as rs
sys.path[:0] = [str(rs.V7), str(rs.ENG / 'drum_set_fea')]
import drum_flow as df
dm, model = df.engine(); flow = df.load(); s = float(np.load(df.PAIRS)['scale']); original = dm.synth_hit; extra = json.loads((rs.ENG / 'drum_set_fea/tab3_drums_fea_modes.json').read_text(encoding='utf-8'))
def with_flow(hit, mdl, rng, maximum_tail=None):
    x = original(hit, mdl if hit.piece in mdl['pieces'] else extra, rng, maximum_tail)
    if hit.piece not in df.ORDER: return x
    y = resample_poly(df.process(flow, resample_poly(x, 147, 160) * s, hit.piece) / s, 160, 147)[:len(x)]; y = np.pad(y, (0, len(x) - len(y))); fade = min(int(.01 * dm.SR), len(y)); y[-fade:] *= np.linspace(1, 0, fade); return y
out = rs.HERE / 'drum_samples'; out.mkdir(exist_ok=True); dm.synth_hit = with_flow
for i, (piece, gap) in enumerate((('kick', .8), ('snare', .8), ('hihat', .5), ('crash', 2.5), ('tom_high', .8)), 1):
    y = dm.render_drums(model, [dm.DrumHit(.2 + k * gap, piece, 108, 0.) for k in range(4)], .2 + 4 * gap + 1.5); sf.write(out / f'{i}_{piece}.flac', y * min(1., .9 / np.abs(y).max()), dm.SR, subtype='PCM_16'); print(piece, 'peak', round(float(np.abs(y).max()), 3), 'rms', round(float(np.sqrt((y ** 2).mean())), 4))
dm.synth_hit = original
