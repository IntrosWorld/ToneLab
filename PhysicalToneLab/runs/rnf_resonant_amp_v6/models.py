"""v6 models.  All map clean harmonic envelopes B [F,H] complex (+ f0) to amped envelopes A [F,H] complex.

ResonantFlow  : the triad cell used as the distortion model.  A weight-tied flow of S steps whose only
                nonlinear terms are frequency-conserving:  z_i z_j -> k=i+j,  z_k conj(z_j) -> i=k-j,
                and level-dependent damping/phase |z_j|^2 z_k.  Couplings come from one shared network
                of the absolute frequencies, so the same weights serve every pitch.
                shuffle=True wires each product to a harmonic that violates the rule (control).
BlackBox      : MLP on real/imag parts (+ log f0).  canonical=True gives it the phase symmetry by
                rotating the fundamental to zero phase.
WienerHammerstein : EQ -> tanh-sum waveshaper on one synthesised period -> EQ (classical grey box).
"""
import math
import torch
from torch import nn

H = 40


def mlp(i, h, o, depth=2):
    layers = []; d = i
    for _ in range(depth): layers += [nn.Linear(d, h), nn.SiLU()]; d = h
    return nn.Sequential(*layers, nn.Linear(d, o))


def feat(f0):
    """log absolute frequency of each harmonic, [F,H]"""
    return torch.log(f0[:, None] * torch.arange(1, H + 1, device=f0.device) / 1000.)


class Gain(nn.Module):
    """Complex gain as a smooth function of absolute frequency (an EQ with phase)."""
    def __init__(self, hidden=16):
        super().__init__(); self.net = mlp(1, hidden, 2); nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, u):
        o = self.net(u[..., None]); return torch.polar(o[..., 0].clamp(-6, 6).exp(), o[..., 1])


class ResonantFlow(nn.Module):
    def __init__(self, steps=8, hidden=24, rank=3, shuffle=False, seed=0):
        super().__init__()
        self.steps = steps; self.pre, self.post = Gain(), Gain()
        up = [(i, j, i + j) for i in range(1, H + 1) for j in range(i, H + 1) if i + j <= H]            # z_i z_j -> k
        dn = [(k, j, k - j) for k in range(2, H + 1) for j in range(1, k) if k - j != j]                # z_k conj(z_j) -> i
        up, dn = torch.tensor(up), torch.tensor(dn)
        if shuffle:                                         # same products, same count, destination breaks i+j=k
            g = torch.Generator().manual_seed(1000 + seed)
            up[:, 2] = (up[:, 2] - 1 + torch.randint(1, H, (len(up),), generator=g)) % H + 1
            dn[:, 2] = (dn[:, 2] - 1 + torch.randint(1, H, (len(dn),), generator=g)) % H + 1
        self.register_buffer('up', up - 1); self.register_buffer('dn', dn - 1)
        self.g_up, self.g_dn = mlp(3, hidden, 2), mlp(3, hidden, 2)
        self.lin = mlp(1, 16, 2, 1); self.ka, self.kb, self.kp = mlp(1, 16, rank, 1), mlp(1, 16, rank, 1), mlp(1, 16, rank, 1)
        self.log_h = nn.Parameter(torch.full((steps,), math.log(.25))); self.drive = nn.Parameter(torch.tensor(0.))
        for net in (self.g_up, self.g_dn): nn.init.normal_(net[-1].weight, std=.02); nn.init.zeros_(net[-1].bias)

    def forward(self, B, f0, mask):
        u = feat(f0); z = B * self.pre(u) * self.drive.exp() * mask
        i, j, k = self.up.unbind(-1); o = self.g_up(torch.stack([u[:, i], u[:, j], u[:, k]], -1)); gu = torch.complex(o[..., 0], o[..., 1])
        a, b, c = self.dn.unbind(-1); o = self.g_dn(torch.stack([u[:, a], u[:, b], u[:, c]], -1)); gd = torch.complex(o[..., 0], o[..., 1])
        lin = self.lin(u[..., None]); lin = torch.complex(lin[..., 0], lin[..., 1])
        ka, kb, kp = nn.functional.softplus(self.ka(u[..., None])), nn.functional.softplus(self.kb(u[..., None])), self.kp(u[..., None])
        for s in range(self.steps):
            h = self.log_h[s].exp(); p = z.abs().square() * mask
            t = torch.zeros_like(z).index_add(1, k, gu * z[:, i] * z[:, j]).index_add(1, c, gd * z[:, a] * z[:, b].conj())
            e = torch.einsum('fkr,fjr,fj->fk', ka, kb, p)                 # level-dependent damping  >= 0
            ph = torch.einsum('fkr,fjr,fj->fk', kp, kb, p)                # level-dependent phase shift
            z = (z * torch.exp(h * lin) * torch.polar(torch.ones_like(ph), h * torch.tanh(ph)) + h * t) / (1 + h * e) * mask
        return z * self.post(u) * mask


class SpectralFlow(nn.Module):
    """Resonant flow with ALL three-wave (i+-j=k) and four-wave (i+-j+-l=k) products.

    Couplings are low-rank in frequency: g(w_i,w_j,w_k) = sum_r a_r(w_i) b_r(w_j) q_r(w_k).  With that form the
    whole resonant sum is computed exactly by filtering, multiplying one synthesised period pointwise and
    transforming back (period 256 > 4*H, so no aliasing).  shuffle=True sends every product to a wrong
    harmonic (fixed permutation): same parameters and cost, frequency conservation broken.
    """
    def __init__(self, steps=8, r2=4, r3=4, rank=3, hidden=24, shuffle=False, seed=0, period=256):
        super().__init__()
        self.steps, self.r2, self.r3, self.n = steps, r2, r3, period; self.nf = 2 * r2 + 3 * r3
        self.pre, self.post = Gain(), Gain(); self.fin = mlp(1, hidden, 2 * self.nf); self.q = mlp(1, hidden, 2 * (r2 + r3))
        self.lin = mlp(1, 16, 2, 1); self.ka, self.kb, self.kp = mlp(1, 16, rank, 1), mlp(1, 16, rank, 1), mlp(1, 16, rank, 1)
        self.log_h = nn.Parameter(torch.full((steps,), math.log(.25))); self.drive = nn.Parameter(torch.tensor(0.))
        nn.init.normal_(self.q[-1].weight, std=.02); nn.init.zeros_(self.q[-1].bias)
        g = torch.Generator().manual_seed(1000 + seed)
        perm = torch.stack([(torch.arange(H) + int(torch.randint(1, H, (1,), generator=g))) % H if shuffle else torch.arange(H) for _ in range(r2 + r3)])
        self.register_buffer('perm', perm); self.shuffle = shuffle

    def forward(self, B, f0, mask):
        u = feat(f0)[..., None]; z = B * self.pre(u[..., 0]) * self.drive.exp() * mask
        a = self.fin(u); a = torch.complex(a[..., :self.nf], a[..., self.nf:])                       # F,H,nf input filters
        q = self.q(u); q = torch.complex(q[..., :self.r2 + self.r3], q[..., self.r2 + self.r3:])      # F,H,R output couplings
        lin = self.lin(u); lin = torch.complex(lin[..., 0], lin[..., 1])
        ka, kb, kp = nn.functional.softplus(self.ka(u)), nn.functional.softplus(self.kb(u)), self.kp(u)
        r2, r3 = self.r2, self.r3
        for s in range(self.steps):
            h = self.log_h[s].exp(); p = z.abs().square() * mask
            spec = torch.zeros(len(z), self.nf, self.n // 2 + 1, dtype=z.dtype, device=z.device); spec[:, :, 1:H + 1] = (a * z[..., None]).transpose(1, 2) * self.n / 2
            w = torch.fft.irfft(spec, self.n)                                                        # F,nf,N filtered periods
            prod = torch.cat([w[:, :r2] * w[:, r2:2 * r2], w[:, 2 * r2:2 * r2 + r3] * w[:, 2 * r2 + r3:2 * r2 + 2 * r3] * w[:, 2 * r2 + 2 * r3:]], 1)
            P = torch.fft.rfft(prod)[:, :, 1:H + 1] * 2 / self.n                                     # F,R,H all resonant products
            if self.shuffle: P = torch.gather(P, 2, self.perm[None].expand(len(P), -1, -1))
            t = (q * P.transpose(1, 2)).sum(-1)
            e = torch.einsum('fkr,fjr,fj->fk', ka, kb, p); ph = torch.einsum('fkr,fjr,fj->fk', kp, kb, p)
            z = (z * torch.exp(h * lin) * torch.polar(torch.ones_like(ph), h * torch.tanh(ph)) + h * t) / (1 + h * e) * mask
        return z * self.post(u[..., 0]) * mask


class BlackBox(nn.Module):
    def __init__(self, hidden=24, depth=2, canonical=False):
        super().__init__(); self.canonical = canonical; self.net = mlp(2 * H + 1, hidden, 2 * H, depth)

    def forward(self, B, f0, mask):
        if self.canonical:
            rot = torch.polar(torch.ones_like(f0), -torch.angle(B[:, 0]))[:, None] ** torch.arange(1, H + 1, device=B.device); B = B * rot
        o = self.net(torch.cat([B.real, B.imag, torch.log(f0 / 200.)[:, None]], -1)); out = torch.complex(o[:, :H], o[:, H:]) * mask
        return out * rot.conj() if self.canonical else out


class WienerHammerstein(nn.Module):
    def __init__(self, units=6, period=256):
        super().__init__(); self.pre, self.post = Gain(), Gain(); self.n = period
        self.d = nn.Parameter(torch.linspace(-1, 2.5, units)); self.b = nn.Parameter(torch.linspace(-.3, .3, units)); self.c = nn.Parameter(torch.full((units,), 1. / units))
        self.drive = nn.Parameter(torch.tensor(0.))

    def forward(self, B, f0, mask):
        u = feat(f0); z = B * self.pre(u) * self.drive.exp() * mask
        spec = torch.zeros(len(z), self.n // 2 + 1, dtype=z.dtype, device=z.device); spec[:, 1:H + 1] = z * self.n / 2
        s = torch.fft.irfft(spec, self.n)                                                    # one period, time domain
        y = (self.c * torch.tanh(self.d.exp() * s[..., None] + self.b)).sum(-1)
        return torch.fft.rfft(y)[:, 1:H + 1] * 2 / self.n * self.post(u) * mask


def build(arm, seed=0):
    return dict(resonant=lambda: SpectralFlow(seed=seed), shuffled=lambda: SpectralFlow(shuffle=True, seed=seed),
                triads_only=lambda: ResonantFlow(seed=seed), triads_only_shuffled=lambda: ResonantFlow(shuffle=True, seed=seed),
                blackbox_small=lambda: BlackBox(24, 2), blackbox_large=lambda: BlackBox(64, 3),
                blackbox_canonical=lambda: BlackBox(24, 2, canonical=True), wiener_hammerstein=WienerHammerstein)[arm]()


if __name__ == '__main__':                                                # checks: parameter counts and the phase symmetry
    torch.manual_seed(0); F = 5; B = torch.randn(F, H, dtype=torch.cfloat) * .1; f0 = torch.tensor([82., 110., 196., 330., 440.]); mask = (torch.arange(1, H + 1) * f0[:, None] <= 8000).float()
    B = B * mask; tau = torch.rand(F) * 6.28; rot = torch.polar(torch.ones(F), tau)[:, None] ** torch.arange(1, H + 1)
    for arm in ('resonant', 'shuffled', 'blackbox_small', 'blackbox_large', 'blackbox_canonical', 'wiener_hammerstein'):
        m = build(arm); n = sum(p.numel() for p in m.parameters())
        with torch.no_grad():
            for p in m.parameters(): p.add_(.05 * torch.randn_like(p))
            y = m(B, f0, mask); y2 = m(B * rot, f0, mask); err = ((y2 - y * rot).abs().square().sum() / y.abs().square().sum()).item()
        print(f'{arm:20s} parameters {n:6d}   time-shift symmetry error {err:.2e}')
