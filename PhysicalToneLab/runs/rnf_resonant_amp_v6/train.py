"""Train / evaluate the v6 arms.   python train.py fit <arm> <seed> [steps] [lr]    |    python train.py test

fit  : uses train frames, selects the checkpoint by validation error, never reads test frames.
test : loads the saved checkpoints and scores the test frames (run once, at the end).
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json, time
import numpy as np
import torch
import models

HERE = Path(__file__).resolve().parent
ARMS = ('resonant', 'shuffled', 'blackbox_small', 'blackbox_large', 'blackbox_canonical', 'wiener_hammerstein')


def data(split):
    d = np.load(HERE / 'envelopes.npz'); s = d['split'] == split; tr = d['split'] == 0
    sb = np.sqrt(np.mean(np.abs(d['B'][tr]) ** 2) * models.H); sa = np.sqrt(np.mean(np.abs(d['A'][tr]) ** 2) * models.H)
    B = torch.from_numpy(d['B'][s] / sb).to(torch.cfloat); A = torch.from_numpy(d['A'][s] / sa).to(torch.cfloat); f0 = torch.from_numpy(d['f0'][s]).float()
    mask = (torch.arange(1, models.H + 1)[None] <= torch.from_numpy(d['K'][s])[:, None]).float()
    return B, A, f0, mask


def esr(model, B, A, f0, mask):
    with torch.no_grad():
        num = den = 0.
        for i in range(0, len(B), 2048):
            p = model(B[i:i + 2048], f0[i:i + 2048], mask[i:i + 2048]); num += (p - A[i:i + 2048]).abs().square().sum().item(); den += A[i:i + 2048].abs().square().sum().item()
    return num / den


def fit(arm, seed, steps=4000, lr=3e-3, tag=''):
    torch.manual_seed(seed); torch.set_num_threads(2); model = models.build(arm, seed); B, A, f0, mask = data(0); val = data(1)
    opt = torch.optim.Adam(model.parameters(), lr); sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=.1)
    g = torch.Generator().manual_seed(seed); best = (float('inf'), None, 0); clock = time.time(); log = []
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(B), (512,), generator=g); p = model(B[idx], f0[idx], mask[idx]); e = (p - A[idx]).abs().square().sum(-1); a = A[idx].abs().square().sum(-1)
        loss = e.sum() / a.sum() + .25 * (e / (a + 1e-3)).clamp(max=4.).mean()            # overall error + per-frame error so quiet frames count
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step(); sched.step()
        if step % 200 == 0 or step == steps:
            v = esr(model, *val); log.append((step, float(loss), v))
            if v < best[0]: best = (v, {k: t.clone() for k, t in model.state_dict().items()}, step)
            print(f'{arm} seed {seed} step {step} loss {loss.item():.4f} val {v:.4f} best {best[0]:.4f} {time.time() - clock:.0f}s', flush=True)
    out = HERE / 'checkpoints'; out.mkdir(exist_ok=True); name = f'{arm}{tag}_seed{seed}'
    torch.save(dict(arm=arm, seed=seed, state=best[1], val=best[0], step=best[2], steps=steps, lr=lr, log=log, parameters=sum(p.numel() for p in model.parameters())), out / f'{name}.pt')
    return best[0]


def test():
    B, A, f0, mask = data(2); rows = {}
    with torch.no_grad(): rows['input_as_output'] = dict(test=[float(((B * (B.conj() * A).sum() / B.abs().square().sum() - A).abs().square().sum() / A.abs().square().sum()))], note='clean signal with one best complex gain')
    for arm in ARMS:
        r = dict(test=[], val=[], parameters=None)
        for seed in (1, 2, 3):
            c = torch.load(HERE / f'checkpoints/{arm}_seed{seed}.pt'); m = models.build(arm, seed); m.load_state_dict(c['state']); r['test'].append(esr(m, B, A, f0, mask)); r['val'].append(c['val']); r['parameters'] = c['parameters']
        r['test_mean'] = float(np.mean(r['test'])); rows[arm] = r; print(arm, r, flush=True)
    res = rows['resonant']['test']; rows['pass'] = bool(all(max(res) < min(rows[a]['test']) for a in ('shuffled', 'blackbox_small', 'blackbox_large')))
    rows['test_frames'] = len(B); (HERE / 'results.json').write_text(json.dumps(rows, indent=1)); print('PASS' if rows['pass'] else 'FAIL')


if __name__ == '__main__':
    if sys.argv[1] == 'test': test()
    else: fit(sys.argv[2], int(sys.argv[3]), *(int(sys.argv[4]),) if len(sys.argv) > 4 else (), **(dict(lr=float(sys.argv[5])) if len(sys.argv) > 5 else {}), **(dict(tag=sys.argv[6]) if len(sys.argv) > 6 else {}))
