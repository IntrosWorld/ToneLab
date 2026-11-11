"""After `train.py test`: results chart + short listening examples from the TEST set.  Writes only here."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
import numpy as np
import soundfile as sf
import torch
import models, train, extract

HERE = Path(__file__).resolve().parent
NAMES = dict(resonant='Resonant flow (proposed)', shuffled='Same network, rule broken (control)', wiener_hammerstein='Classical EQ-waveshaper-EQ',
             blackbox_large='Black-box MLP, 18.8k', blackbox_canonical='Black-box MLP + phase symmetry', blackbox_small='Black-box MLP, 4.6k')


def chart():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    r = json.loads((HERE / 'results.json').read_text()); arms = sorted(NAMES, key=lambda a: r[a]['test_mean'])
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=160); y = np.arange(len(arms))[::-1]
    for yi, a in zip(y, arms):
        ax.barh(yi, r[a]['test_mean'], height=.55, color='#2a78d6' if a == 'resonant' else '#b9bec6')
        ax.plot(r[a]['test'], [yi] * 3, 'o', ms=4, color='#1f2430', zorder=3)
        ax.text(max(r[a]['test']) + .012, yi, f"{r[a]['test_mean']:.3f}", va='center', fontsize=9, color='#1f2430')
    ax.set_yticks(y); ax.set_yticklabels([f"{NAMES[a]}  ({r[a]['parameters']:,} params)" for a in arms], fontsize=9)
    ax.set_xlabel('Error on held-out test performances (lower is better); dots = 3 seeds', fontsize=9); ax.set_xlim(0, max(r[a]['test_mean'] for a in arms) * 1.18)
    for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
    ax.tick_params(axis='y', length=0); ax.grid(axis='x', color='#e6e8ec', lw=.8); ax.set_axisbelow(True)
    fig.suptitle('Learning a real amp tone from clean guitar: test error by model', fontsize=10.5, x=.02, ha='left'); fig.tight_layout(); fig.savefig(HERE / 'results_chart.png'); print('chart written')


def audition():
    d = np.load(HERE / 'envelopes.npz'); s = d['split'] == 2; fid, t, f0 = d['fid'][s], d['t'][s], d['f0'][s]; B, A, F0, mask = train.data(2)
    tr = d['split'] == 0; sa = np.sqrt(np.mean(np.abs(d['A'][tr]) ** 2) * models.H)
    c = torch.load(HERE / 'checkpoints/resonant_seed1.pt'); m = models.build('resonant', 1); m.load_state_dict(c['state'])
    with torch.no_grad(): P = m(B, F0, mask).numpy() * sa
    start = np.round(t * extract.SR).astype(int); brk = np.where((np.diff(start) != extract.HOP) | (np.diff(fid) != 0))[0] + 1
    runs = sorted(np.split(np.arange(len(t)), brk), key=len, reverse=True)[:10]; runs = sorted(runs, key=lambda r: (fid[r[0]], t[r[0]]))
    manifest = json.loads((extract.PAIRS / 'manifest.json').read_text())['pairs']; out = dict(clean=[], target=[], predicted=[]); gap = np.zeros(int(.25 * extract.SR)); info = []
    for r in runs:
        row = manifest[fid[r[0]]]; z = np.load(extract.PAIRS / row['file']); a, b = start[r[0]], start[r[-1]] + extract.WIN
        frames = extract.resynth(P[r], f0[r]) * extract.W; y = np.zeros(b - a)
        for i, fr in zip(start[r] - a, frames): y[i:i + extract.WIN] += fr
        y /= 2.                                                      # Hann windows at hop WIN/4 sum to 2
        edge = slice(extract.WIN // 2, len(y) - extract.WIN // 2)    # full-overlap region only
        out['clean'] += [z['x'][a:b][edge], gap]; out['target'] += [z['y'][a:b][edge], gap]; out['predicted'] += [y[edge], gap]
        info.append(dict(file=row['id'], start_s=float(t[r[0]]), seconds=float((b - a) / extract.SR), f0_hz=float(np.median(f0[r]))))
    (HERE / 'audition').mkdir(exist_ok=True)
    for name, label in (('clean', 'A_clean_guitar_input'), ('target', 'B_real_amp_target'), ('predicted', 'C_resonant_flow_prediction')):
        sf.write(HERE / f'audition/{label}.wav', np.concatenate(out[name]), extract.SR, subtype='PCM_24')
    (HERE / 'audition/notes.json').write_text(json.dumps(dict(note='10 longest steady single-note stretches of the TEST performances, 0.25 s gaps. C is rebuilt from predicted harmonics only (no pick noise, nothing above 8 kHz). Not listened to by the assistant.', excerpts=info), indent=1))
    print('audition written', len(runs), 'excerpts', round(sum(i['seconds'] for i in info), 1), 's')


if __name__ == '__main__':
    chart(); audition()
