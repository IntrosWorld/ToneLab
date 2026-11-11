"""RAT song, third render.  v2 was reported by ear to clip / sound odd at 10-11 s and 18-20 s; measured there: output
spikes up to 2x the ceiling and level dips on new pick attacks (frames where the safety limit acted).
Change: no clamping inside the flow.  Every frame is run at several input levels (0, -6, -12, -18, -24 dB) and the
loudest input whose state stays inside the training range at every step is used (the RAT saturates, so its output
level barely depends on input level).  Same trained weights; this is a hand-made safety rule, not learned.
"""
import sys
sys.dont_write_bytecode = True
import json
import numpy as np
import soundfile as sf
import torch
from torch import nn
from scipy.signal import resample_poly
import render_tones as R, render_rat_song as RS
from render_tones import ra, OUT, SR, norm, models

N, HOP = ra.N, ra.HOP; GAINS = (1., .5, .25, .125, .0625)


@torch.no_grad()
def process(m, x):
    sb, sa = ra.scales(); x2 = resample_poly(x, 2, 1) / sb; pad = N; x2 = np.concatenate([np.zeros(pad), x2, np.zeros(pad + N)])
    f = torch.fft.rfftfreq(N, 1 / (2 * SR)); band = ((f >= ra.LO) & (f <= ra.TOP)).float(); u = torch.log(f.clamp(ra.LO, ra.TOP) / 1000.)[:, None]
    pre, post = m.pre(u[:, 0]) * m.drive.exp(), m.post(u[:, 0]); a = m.fin(u); a = torch.complex(a[:, :m.nf], a[:, m.nf:]).T; q = m.q(u); Rk = m.r2 + m.r3; q = torch.complex(q[:, :Rk], q[:, Rk:]).T
    lin = m.lin(u); lin = torch.complex(lin[:, 0], lin[:, 1]); ka, kb, kp = nn.functional.softplus(m.ka(u)), nn.functional.softplus(m.kb(u)), m.kp(u); caps = torch.tensor(ra.step_caps(m)); r2, r3 = m.r2, m.r3
    win = torch.from_numpy(np.hanning(N + 1)[:-1]).float(); starts = np.arange(0, len(x2) - N, HOP); y = np.zeros(len(x2)); used = []
    for b in range(0, len(starts), 128):
        st = starts[b:b + 128]; fr = torch.from_numpy(np.stack([x2[s:s + N] for s in st])).float(); out = torch.zeros(len(st), N); done = torch.zeros(len(st), dtype=torch.bool); lvl = torch.full((len(st),), len(GAINS) - 1)
        for gi, g in enumerate(GAINS):
            todo = ~done
            if not todo.any(): break
            Z = torch.fft.rfft(fr[todo] * g) * pre * band; worst = torch.zeros(int(todo.sum()))
            for s in range(m.steps):
                h = m.log_h[s].exp(); p = (2 * Z.abs() / N).square() * band; w = torch.fft.irfft(Z[:, None] * a, N)
                prod = torch.cat([w[:, :r2] * w[:, r2:2 * r2], w[:, 2 * r2:2 * r2 + r3] * w[:, 2 * r2 + r3:2 * r2 + 2 * r3] * w[:, 2 * r2 + 2 * r3:]], 1); t = (q * torch.fft.rfft(prod)).sum(1)
                e = torch.einsum('kr,jr,fj->fk', ka, kb, p); ph = torch.einsum('kr,jr,fj->fk', kp, kb, p)
                Z = (Z * torch.exp(h * lin) * torch.polar(torch.ones_like(ph), h * torch.tanh(ph)) + h * t) / (1 + h * e) * band
                en = (2 * Z.abs() / N).square().sum(-1); worst = torch.maximum(worst, en / caps[s])
                if gi == len(GAINS) - 1: Z = Z * torch.sqrt((caps[s] / en.clamp_min(1e-12)).clamp(max=1.))[:, None]            # last resort only
            ok = (worst <= 1.) | (gi == len(GAINS) - 1); idx = torch.nonzero(todo)[:, 0][ok]; out[idx] = (torch.fft.irfft(Z * post, N) * win)[ok]; done[idx] = True; lvl[idx] = gi
        used += lvl.tolist()
        for s, o in zip(st, out.numpy()): y[s:s + N] += o
    y = y[pad:pad + 2 * len(x)] / 2. * sa; return resample_poly(y, 1, 2)[:len(x)], np.bincount(used, minlength=len(GAINS))


if __name__ == '__main__':
    song, sr = sf.read(ra.OLD / 'artifact_study/clean_release_di.wav', dtype='float64', always_2d=True); song = song.mean(1); m, seed = R.model('RAT'); top = R.ceiling('RAT')
    x = RS.matched(song, .008); y, used = process(m, x); pk = float(np.abs(y).max()); clipped = int((np.abs(y) > top).sum()); blk = 2205; play = np.sqrt((x[:len(x) // blk * blk].reshape(-1, blk) ** 2).mean(1)) > .003
    lv = np.sqrt((y[:len(y) // blk * blk].reshape(-1, blk) ** 2).mean(1))[play]; pkb = np.abs(y[:len(y) // blk * blk]).reshape(-1, blk).max(1)[play]
    info = dict(frames_by_input_step_0_to_minus24dB=used.tolist(), peak_before_ceiling=round(pk, 2), samples_over_ceiling=clipped, blocks_50ms_while_playing=int(play.sum()), blocks_with_level_under_0p3=int((lv < .3).sum()), blocks_with_peak_over_ceiling=int((pkb > top).sum()),
                level_percentiles_5_50_95=np.percentile(lv, [5, 50, 95]).round(2).tolist(), note='not listened to by the assistant')
    for a, b in ((10.3, 11.3), (19.0, 20.0)): info[f'level_{a}_{b}s'] = np.round(np.sqrt((y[int(a * SR):int(b * SR)].reshape(-1, 4410) ** 2).mean(1)), 2).tolist(); info[f'peak_{a}_{b}s'] = np.round(np.abs(y[int(a * SR):int(b * SR)]).reshape(-1, 4410).max(1), 2).tolist()
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    env = uniform_filter1d(maximum_filter1d(np.abs(y), 257), 129); y = y * np.minimum(1., .95 * top / np.maximum(env, 1e-9)); info['peak_after_smooth_limiter'] = round(float(np.abs(y).max()), 2)      # 3 ms look-ahead gain instead of hard clipping
    y = np.clip(y, -top, top); sf.write(OUT / 'song_RAT_v3_frame_level_retry.flac', norm(y), SR, subtype='PCM_16'); print(json.dumps(info, indent=1)); (OUT / 'song_RAT_v3_notes.json').write_text(json.dumps(info, indent=1))
