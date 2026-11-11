"""RAT song, second render.  The first one (song_RAT_resonant_flow.flac) was rejected by ear: pauses, heavy break-up.
Cause measured: the song's guitar signal is ~8x hotter than the training notes and has far weaker harmonics than the
real Stratocaster the RAT model was trained on (640-2560 Hz: 10-12 dB lower); outside its training range the model turns
loud passages DOWN.  Fix (hand steps on the INPUT, not learned): a static EQ that brings the song's average spectrum to
the training guitar's, and input level set to the training notes' median.  Same trained weights.   python render_rat_song.py
"""
import sys
sys.dont_write_bytecode = True
import json
import numpy as np
import soundfile as sf
import render_tones as R
from render_tones import ra, OUT, SR, norm

CEN = np.array([113, 226, 453, 905, 1810, 3620, 7100.]); GDB = np.array([0, 0, -3.6, 10.3, 11.6, 3.5, -10.2])     # training Strat minus song DI, octave bands, capped at +-12


def matched(x, level):
    F = np.fft.rfftfreq(len(x), 1 / SR); x = np.fft.irfft(np.fft.rfft(x) * 10 ** (np.interp(np.log(np.maximum(F, 1)), np.log(CEN), GDB) / 20), len(x))
    fr = np.sqrt((x[:len(x) // 4096 * 4096].reshape(-1, 4096) ** 2).mean(1)); return x * level / np.median(fr[fr > .1 * fr.max()])


if __name__ == '__main__':
    song, sr = sf.read(ra.OLD / 'artifact_study/clean_release_di.wav', dtype='float64', always_2d=True); song = song.mean(1); m, seed = R.model('RAT'); top = R.ceiling('RAT'); info = {}
    blk = 8820; prof = lambda v: np.round(np.sqrt((v[:len(v) // blk * blk].reshape(-1, blk) ** 2).mean(1)), 2)
    for level in (.013, .008):
        ra.LIMITED[:] = [0, 0]; y = ra.process(m, matched(song, level)); pk = float(np.abs(y).max()); y = np.clip(y, -top, top); name = f'song_RAT_v2_input_matched_level{level}.flac'
        sf.write(OUT / name, norm(y), SR, subtype='PCM_16'); p = prof(y); act = p[prof(song) > .02 * prof(song).max()]
        info[name] = dict(frames_limited=list(ra.LIMITED), peak_before_ceiling=round(pk, 2), level_while_playing_min_median_max=[float(act.min()), float(np.median(act)), float(act.max())], crest_db=ra.describe(y)['crest_db']); print(name, info[name], flush=True)
    (OUT / 'song_RAT_v2_notes.json').write_text(json.dumps(dict(info, note='not listened to by the assistant'), indent=1))
