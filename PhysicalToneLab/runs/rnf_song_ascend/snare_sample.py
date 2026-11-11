import sys, json; sys.dont_write_bytecode = True
import numpy as np, soundfile as sf
from scipy.signal import resample_poly
import render_song as rs
sys.path[:0] = [str(rs.V7), str(rs.ENG / 'drum_set_fea')]
import drum_flow as df, drum_snare_v2 as sv
dm, model = df.engine(); s = float(np.load(df.PAIRS)['scale']); original = dm.synth_hit; hits = [dm.DrumHit(.2 + k * .8, 'snare', 108, 0.) for k in range(4)]
def make(m):
    def f(hit, mdl, rng, maximum_tail=None):
        x = original(hit, mdl, rng, maximum_tail); y = resample_poly(df.process(m, resample_poly(x, 147, 160) * s, hit.piece) / s, 160, 147)[:len(x)]; return np.pad(y, (0, len(x) - len(y)))
    return f
for name, fn in (('2b_snare_physical_model_only', original), ('2c_snare_retrained', make(sv.load()))):
    dm.synth_hit = fn; y = dm.render_drums(model, hits, 4.9); dm.synth_hit = original; sf.write(rs.HERE / 'drum_samples' / f'{name}.flac', y * min(1., .9 / np.abs(y).max()), dm.SR, subtype='PCM_16')
    S = np.abs(np.fft.rfft(y.mean(1))) ** 2; f = np.fft.rfftfreq(len(y), 1 / dm.SR); print(name, 'share of energy above 3 kHz: %.0f %%' % (100 * S[f > 3000].sum() / S.sum()))
