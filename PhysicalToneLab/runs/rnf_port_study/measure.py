"""Port study: what the Python implementation costs now, and what shorter frames (lower latency) do to accuracy.
Reads rnf_resonant_amp_v6 and rnf_instruments_v7, writes only here.   python measure.py
"""
import sys, time, json
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent; V6 = HERE.parent / 'rnf_resonant_amp_v6'; V7 = HERE.parent / 'rnf_instruments_v7'
sys.path[:0] = [str(V6), str(V7), str(V7 / 'tones')]
import render_audio as ra, extract
res = {}; m = ra.load(); SR = ra.SR
pairs = [(r['id'], np.load(extract.PAIRS / r['file'])) for r in json.loads((extract.PAIRS / 'manifest.json').read_text())['pairs'] if r['split'] == 'test']
caps = ra.step_caps(m); ra.step_caps = lambda m_: caps                       # computed once; not part of the per-audio cost

# 1. accuracy and speed vs frame length (same weights)
for N in (4096, 2048, 1024, 512, 256):
    ra.N, ra.HOP = N, N // 4; errs = []; sec = 0.; t0 = time.perf_counter(); ra.LIMITED[:] = [0, 0]
    for pid, d in pairs:
        x, t = d['x'].astype(np.float64), d['y'].astype(np.float64); y = ra.process(m, x); errs.append(round(ra.esr(y, t), 3)); sec += len(x) / SR
    el = time.perf_counter() - t0; res[f'amp_N{N}'] = dict(frame_ms=round(N / 2 / SR * 1e3, 1), hop_ms=round(N / 8 / SR * 1e3, 2), waveform_error=errs, audio_seconds=round(sec, 1), compute_seconds=round(el, 1), times_faster_than_real_time=round(sec / el, 2), frames_limited=list(ra.LIMITED)); print(f'amp_N{N}', res[f'amp_N{N}'], flush=True)

# 2. one thread (what a real-time audio callback gets)
torch.set_num_threads(1); ra.N, ra.HOP = 4096, 1024; x = pairs[0][1]['x'].astype(np.float64)[:10 * SR]; t0 = time.perf_counter(); ra.process(m, x); el = time.perf_counter() - t0
res['amp_N4096_one_thread'] = dict(times_faster_than_real_time=round(len(x) / SR / el, 2)); print(res['amp_N4096_one_thread'], flush=True)

# 3. where the time goes (N=4096, one thread, 64 frames)
N = 4096; Z = torch.randn(64, N // 2 + 1, dtype=torch.cfloat); a = torch.randn(20, N // 2 + 1, dtype=torch.cfloat); tm = {}
def clock(name, fn, reps=5):
    t0 = time.perf_counter()
    for _ in range(reps): out = fn()
    tm[name] = (time.perf_counter() - t0) / reps / 64 * 1e3; return out
w = clock('20 inverse FFTs', lambda: torch.fft.irfft(Z[:, None] * a, N)); prod = clock('8 products', lambda: torch.cat([w[:, :4] * w[:, 4:8], w[:, 8:12] * w[:, 12:16] * w[:, 16:]], 1)); clock('8 forward FFTs', lambda: torch.fft.rfft(prod))
ka = torch.rand(N // 2 + 1, 3); p = torch.rand(64, N // 2 + 1); clock('damping + phase', lambda: (torch.einsum('kr,jr,fj->fk', ka, ka, p), torch.einsum('kr,jr,fj->fk', ka, ka, p)))
res['ms_per_frame_per_step_N4096_one_thread'] = {k: round(v, 3) for k, v in tm.items()}; print(res['ms_per_frame_per_step_N4096_one_thread'], flush=True)

# 4. other models
torch.set_num_threads(14)
import chorus as C
cm = C.load(); x = np.random.default_rng(0).standard_normal(30 * SR) * .1; t0 = time.perf_counter(); C.process(cm, x); res['chorus'] = dict(times_faster_than_real_time=round(30 / (time.perf_counter() - t0), 1)); print('chorus', res['chorus'], flush=True)
import electric_clean_flow as ecf, guitar_flow as gf, piano_decay as pd
fl = gf.load(ecf.NAME); x = np.random.default_rng(0).standard_normal(int(2.5 * SR)) * .1; t0 = time.perf_counter()
for _ in range(10): pd.process(fl, x, 60, 100)
res['per_note_flow'] = dict(ms_per_2p5s_note=round((time.perf_counter() - t0) / 10 * 1e3, 1), times_faster_than_real_time=round(25 / (time.perf_counter() - t0), 1)); print('per-note flow', res['per_note_flow'], flush=True)
(HERE / 'python_measurements.json').write_text(json.dumps(res, indent=1))
