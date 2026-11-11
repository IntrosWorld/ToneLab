"""Render the Sun Raha Hai lead (clean_release DI) through the trained resonant flow.  Writes only to audition/.

The model works on one pitch per frame.  Frames where no pitch is found (pick attacks, transitions) borrow the
pitch of the nearest pitched frame; whatever is not a harmonic of the tracked pitch is not reproduced.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
import numpy as np
import soundfile as sf
import torch
import models, extract

HERE = Path(__file__).resolve().parent
OLD = HERE.parents[2] / 'NeuralInstrumentStudio_Experiments/energy_v3'
SR, WIN, HOP, H = extract.SR, extract.WIN, extract.HOP, extract.H


def main(seed=2):
    x, sr = sf.read(OLD / 'artifact_study/clean_release_di.wav', dtype='float64', always_2d=True); assert sr == SR; x = x.mean(1)
    ref, _ = sf.read(OLD / 'artifact_study/clean_release_dry.wav', dtype='float64', always_2d=True); ref = ref.mean(1)
    x = np.concatenate([np.zeros(WIN), x, np.zeros(WIN)]); starts = np.arange(0, len(x) - WIN, HOP); f0 = np.full(len(starts), np.nan); rms = np.zeros(len(starts))
    for n, s in enumerate(starts):
        fr = x[s:s + WIN]; rms[n] = np.sqrt(np.mean(fr ** 2))
        if rms[n] < .002: continue
        f = extract.yin(fr * extract.W)
        if f is not None and 60 < f < 1400: f0[n] = extract.refine(fr, f)
    voiced = np.isfinite(f0); idx = np.where(voiced)[0]; borrowed = 0
    for n in np.where(~voiced & (rms >= .002))[0]:                       # attacks: take the next pitched frame, else the previous
        later = idx[idx > n]; earlier = idx[idx < n]
        if len(later) and later[0] - n <= 4: f0[n] = f0[later[0]]; borrowed += 1
        elif len(earlier) and n - earlier[-1] <= 4: f0[n] = f0[earlier[-1]]; borrowed += 1
    use = np.where(np.isfinite(f0))[0]; F = f0[use]; X = np.stack([x[starts[n]:starts[n] + WIN] for n in use])
    B = extract.harmonics(X, F, H); K = np.minimum(H, np.floor(extract.TOP / F)).astype(int); mask = (np.arange(1, H + 1)[None] <= K[:, None]); B = B * mask
    resid = ((X - extract.resynth(B, F)) ** 2 * extract.W ** 2).sum(1) / np.maximum((X ** 2 * extract.W ** 2).sum(1), 1e-12)
    d = np.load(HERE / 'envelopes.npz'); tr = d['split'] == 0; sb = np.sqrt(np.mean(np.abs(d['B'][tr]) ** 2) * H); sa = np.sqrt(np.mean(np.abs(d['A'][tr]) ** 2) * H)
    c = torch.load(HERE / f'checkpoints/resonant_seed{seed}.pt'); m = models.build('resonant', seed); m.load_state_dict(c['state'])
    with torch.no_grad(): P = m(torch.from_numpy(B / sb).to(torch.cfloat), torch.from_numpy(F).float(), torch.from_numpy(mask).float()).numpy() * sa
    y = np.zeros(len(x)); frames = extract.resynth(P, F) * extract.W
    for n, fr in zip(use, frames): y[starts[n]:starts[n] + WIN] += fr
    y = y[WIN:-WIN] / 2.; x = x[WIN:-WIN]; n = min(len(y), len(ref)); out = HERE / 'audition'
    sf.write(out / 'song_1_clean_guitar_input.wav', x, SR, subtype='PCM_24')
    sf.write(out / 'song_2_previous_amp_same_notes.wav', ref[:n], SR, subtype='PCM_24')
    sf.write(out / 'song_3_resonant_flow.wav', y, SR, subtype='PCM_24')

    def describe(a):
        S = np.abs(np.fft.rfft(a)) ** 2; f = np.fft.rfftfreq(len(a), 1 / SR)
        return dict(seconds=round(len(a) / SR, 1), rms=round(float(np.sqrt(np.mean(a ** 2))), 4), peak=round(float(np.abs(a).max()), 3), centroid_hz=round(float((S * f).sum() / S.sum())),
                    crest_db=round(float(20 * np.log10(np.abs(a).max() / np.sqrt(np.mean(a ** 2)))), 1))
    env = lambda a: np.array([np.sqrt(np.mean(a[i:i + 2048] ** 2)) for i in range(0, n - 2048, 1024)])
    info = dict(frames=len(starts), pitched=int(voiced.sum()), pitch_borrowed_from_neighbour=borrowed, silent_or_unpitched=int(len(starts) - len(use)),
                input_not_captured_by_one_pitch_median=float(np.median(resid)), frames_over_10_percent_uncaptured=int((resid > .1).sum()), input_peak=float(np.abs(x).max()),
                clean=describe(x), previous_amp=describe(ref[:n]), resonant_flow=describe(y), level_envelope_correlation_with_previous_amp=round(float(np.corrcoef(env(y[:n]), env(ref[:n]))[0, 1]), 3),
                waveform_error_vs_previous_amp=round(float(((y[:n] - ref[:n]) ** 2).sum() / (ref[:n] ** 2).sum()), 3),
                note='song_2 is the earlier neural amp (dry, mono) on the same notes. Not listened to by the assistant.')
    (out / 'song_notes.json').write_text(json.dumps(info, indent=1)); print(json.dumps(info, indent=1))


if __name__ == '__main__':
    main()
