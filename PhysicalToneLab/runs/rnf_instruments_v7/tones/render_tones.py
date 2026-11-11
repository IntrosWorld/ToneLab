"""Listening files for the pedal models: the trained resonant flow of each pedal run on full audio (v6 render_audio.process,
same weights, no retraining).   python render_tones.py
  heldout_<tone>_{1_clean,2_real_pedal,3_resonant_flow}.flac : 12 held-out notes (frets 3, 8, 13, 18), never trained on
  song_<tone>_resonant_flow.flac                              : the clean song lead through each pedal model
Writes only to tones/audition/.  Not listened to by the assistant.
"""
import sys
sys.dont_write_bytecode = True
import json
import numpy as np
import soundfile as sf
import torch
import tones as T
from tones import HERE, TONES, DATA, models, tr, ex
import render_audio as ra

OUT = HERE / 'audition'; OUT.mkdir(exist_ok=True); SR = ex.SR
NOTES = [f'{s}-{f}.wav' for s, f in ((6, 3), (5, 3), (4, 8), (3, 8), (2, 13), (1, 13), (6, 8), (5, 13), (4, 3), (3, 18), (2, 8), (1, 3))]
norm = lambda v: v * (.89 / max(np.abs(v).max(), 1e-9))
IN_GAIN = dict(RAT=.5)        # hand setting, not learned: the RAT model is run 6 dB below training level (pick attacks overload it at full level)


def ceiling(tone):
    """Largest sample of the real pedal on the training notes; model output is clipped there (safety step, not learned)."""
    return max(np.abs(T.read(f)).max() for f in sorted((DATA / tone).glob('*.wav')) if T.SPLIT(int(f.stem.split('-')[1])) == 0)


def model(tone):
    ra.HERE = tr.HERE = HERE / tone; best = min((torch.load(c) for c in (HERE / tone / 'checkpoints').glob('resonant_seed*.pt')), key=lambda c: c['val'])   # chosen on validation
    m = models.build('resonant', best['seed']); m.load_state_dict(best['state']); return m.eval(), best['seed']


def main():
    info = {}; song, sr = sf.read(ra.OLD / 'artifact_study/clean_release_di.wav', dtype='float64', always_2d=True); song = song.mean(1); song = song * (.25 / np.quantile(np.abs(song), .9995))
    for tone in TONES:
        m, seed = model(tone); X, Y, P = [], [], []; ra.LIMITED[:] = [0, 0]; g = IN_GAIN.get(tone, 1.); top = ceiling(tone); raw_peak = 0.
        for n in NOTES:
            x, y = T.read(DATA / 'Clean' / n), T.read(DATA / tone / n); L = min(len(x), len(y), int(2.5 * SR)); x, y = x[:L] * (.25 / np.abs(x).max()), y[:L]
            o = ra.process(m, x * g); raw_peak = max(raw_peak, np.abs(o).max()); X.append(x); Y.append(y); P.append(np.clip(o, -top, top))
        x, y, p = map(np.concatenate, (X, Y, P)); lim_notes = list(ra.LIMITED); ra.LIMITED[:] = [0, 0]
        for name, v in (('1_clean', x), ('2_real_pedal', y), ('3_resonant_flow', p)): sf.write(OUT / f'heldout_{tone}_{name}.flac', norm(v), SR, subtype='PCM_16')
        s = ra.process(m, song * g); song_raw_peak = float(np.abs(s).max()); s = np.clip(s, -top, top); sf.write(OUT / f'song_{tone}_resonant_flow.flac', norm(s), SR, subtype='PCM_16')
        info[tone] = dict(seed=seed, input_gain=g, ceiling=round(float(top), 3), heldout_peak_before_ceiling=round(float(raw_peak), 2), song_peak_before_ceiling=round(song_raw_peak, 2), heldout_waveform_error=round(ra.esr(p, y), 3), heldout_clean_as_output=round(ra.esr(x * (x @ y) / (x @ x), y), 3), real_pedal=ra.describe(y), resonant_flow=ra.describe(p),
                          heldout_frames_limited=lim_notes, song=ra.describe(s), song_frames_limited=list(ra.LIMITED), nan=bool(np.isnan(s).any() or np.isnan(p).any())); print(tone, json.dumps(info[tone]), flush=True)
    sf.write(OUT / 'song_0_clean_input.flac', norm(song), SR, subtype='PCM_16')
    info['note'] = 'each file peak-normalised for listening; held-out notes are 2.5 s each; not listened to by the assistant'; (OUT / 'notes.json').write_text(json.dumps(info, indent=1))


if __name__ == '__main__':
    main()
