"""One resonant flow for several tones: every coupling network also receives a learned tone vector.

Same equations as models.SpectralFlow (all three- and four-wave resonant products, level-dependent damping,
weight-tied steps); the only addition is a D-dimensional embedding per tone concatenated to the frequency feature.
Blending two embeddings gives a tone between two pedals.     python cond.py fit <seed> [steps]   |   python cond.py test
"""
import sys
sys.dont_write_bytecode = True
import json, math, time
import numpy as np
import torch
from torch import nn
import tones as T
from tones import HERE, TONES, models, tr

H, D = models.H, 4


class CondFlow(nn.Module):
    def __init__(self, n_tones=len(TONES), steps=8, r2=4, r3=4, rank=3, hidden=32, period=256):
        super().__init__(); self.steps, self.r2, self.r3, self.n = steps, r2, r3, period; self.nf = 2 * r2 + 3 * r3; i = 1 + D; m = models.mlp
        self.emb = nn.Embedding(n_tones, D); nn.init.normal_(self.emb.weight, std=.5)
        self.pre, self.post, self.fin, self.q = m(i, 16, 2), m(i, 16, 2), m(i, hidden, 2 * self.nf), m(i, hidden, 2 * (r2 + r3))
        self.lin, self.ka, self.kb, self.kp = m(i, 16, 2, 1), m(i, 16, rank, 1), m(i, 16, rank, 1), m(i, 16, rank, 1)
        self.log_h = nn.Parameter(torch.full((steps,), math.log(.25))); self.drive = m(D, 8, 1, 1)
        for net in (self.pre, self.post): nn.init.zeros_(net[-1].weight); nn.init.zeros_(net[-1].bias)
        nn.init.normal_(self.q[-1].weight, std=.02); nn.init.zeros_(self.q[-1].bias)

    def forward(self, B, f0, mask, tone=None, vec=None):
        c = self.emb(tone) if vec is None else vec; x = torch.cat([models.feat(f0)[..., None], c[:, None].expand(-1, H, -1)], -1)
        cg = lambda o: torch.polar(o[..., 0].clamp(-6, 6).exp(), o[..., 1]); z = B * cg(self.pre(x)) * self.drive(c).exp() * mask
        a = self.fin(x); a = torch.complex(a[..., :self.nf], a[..., self.nf:]); R = self.r2 + self.r3; q = self.q(x); q = torch.complex(q[..., :R], q[..., R:])
        lin = self.lin(x); lin = torch.complex(lin[..., 0], lin[..., 1]); sp = nn.functional.softplus; ka, kb, kp = sp(self.ka(x)), sp(self.kb(x)), self.kp(x); r2, r3 = self.r2, self.r3
        for s in range(self.steps):
            h = self.log_h[s].exp(); p = z.abs().square() * mask
            spec = torch.zeros(len(z), self.nf, self.n // 2 + 1, dtype=z.dtype); spec[:, :, 1:H + 1] = (a * z[..., None]).transpose(1, 2) * self.n / 2; w = torch.fft.irfft(spec, self.n)
            prod = torch.cat([w[:, :r2] * w[:, r2:2 * r2], w[:, 2 * r2:2 * r2 + r3] * w[:, 2 * r2 + r3:2 * r2 + 2 * r3] * w[:, 2 * r2 + 2 * r3:]], 1)
            t = (q * (torch.fft.rfft(prod)[:, :, 1:H + 1] * 2 / self.n).transpose(1, 2)).sum(-1)
            e = torch.einsum('fkr,fjr,fj->fk', ka, kb, p); ph = torch.einsum('fkr,fjr,fj->fk', kp, kb, p)
            z = (z * torch.exp(h * lin) * torch.polar(torch.ones_like(ph), h * torch.tanh(ph)) + h * t) / (1 + h * e) * mask
        return z * cg(self.post(x)) * mask


def data(split):
    """All tones stacked.  Inputs share one scale; each tone's target keeps its own scale (stored for rendering)."""
    out = [[] for _ in range(5)]
    for ti, tone in enumerate(TONES):
        tr.HERE = HERE / tone; B, A, f0, mask = tr.data(split)
        for o, v in zip(out, (B, A, f0, mask, torch.full((len(B),), ti))): o.append(v)
    return [torch.cat(o) for o in out]


def esr(model, B, A, f0, mask, tone):
    with torch.no_grad():
        num = den = 0.
        for i in range(0, len(B), 2048):
            p = model(B[i:i + 2048], f0[i:i + 2048], mask[i:i + 2048], tone[i:i + 2048]); num += (p - A[i:i + 2048]).abs().square().sum().item(); den += A[i:i + 2048].abs().square().sum().item()
    return num / den


def fit(seed=1, steps=3000, lr=3e-3):
    torch.manual_seed(seed); torch.set_num_threads(4); model = CondFlow(); B, A, f0, mask, tone = data(0); val = data(1); g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr); sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=.1); best = (9., None, 0); clock = time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(B), (512,), generator=g); p = model(B[idx], f0[idx], mask[idx], tone[idx]); e = (p - A[idx]).abs().square().sum(-1); a = A[idx].abs().square().sum(-1)
        loss = e.sum() / a.sum() + .25 * (e / (a + 1e-3)).clamp(max=4.).mean(); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step(); sched.step()
        if step % 250 == 0 or step == steps:
            v = esr(model, *val)
            if v < best[0]: best = (v, {k: t.clone() for k, t in model.state_dict().items()}, step)
            print(f'cond seed {seed} step {step} loss {loss.item():.4f} val {v:.4f} best {best[0]:.4f} {time.time() - clock:.0f}s', flush=True)
    (HERE / 'checkpoints').mkdir(exist_ok=True); torch.save(dict(state=best[1], val=best[0], step=best[2], seed=seed, parameters=sum(p.numel() for p in model.parameters())), HERE / f'checkpoints/cond_seed{seed}.pt')


def load(seed=1):
    c = torch.load(HERE / f'checkpoints/cond_seed{seed}.pt'); m = CondFlow(); m.load_state_dict(c['state']); return m.eval(), c


def test():
    B, A, f0, mask, tone = data(2); out = {}
    for ck in sorted((HERE / 'checkpoints').glob('cond_seed*.pt')):
        m, c = load(int(ck.stem.split('seed')[1])); out[ck.stem] = dict(parameters=c['parameters'], **{t: esr(m, *[v[tone == i] for v in (B, A, f0, mask, tone)]) for i, t in enumerate(TONES)})
    print(json.dumps(out)); (HERE / 'results_cond.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    dict(test=test, fit=lambda: fit(int(sys.argv[2]), *(int(sys.argv[3]),) if len(sys.argv) > 3 else ()))[sys.argv[1]]()
