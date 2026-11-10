"""Modal FEA drum kit: circular membranes for shells and plates for cymbals."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla
from scipy.signal import butter, sosfilt

SR = 48_000


@dataclass(frozen=True)
class DrumHit:
    start: float
    piece: str
    velocity: int = 108
    open_amount: float = 0.0


SPECS = {
    "kick":  {"radius": .28, "target": 58.,  "q": 8.,  "kind": "membrane", "modes": 18},
    "snare": {"radius": .18, "target": 185., "q": 9.,  "kind": "membrane", "modes": 22},
    "tom_high": {"radius": .13, "target": 205., "q": 10., "kind": "membrane", "modes": 18},
    "tom_mid":  {"radius": .16, "target": 155., "q": 10., "kind": "membrane", "modes": 18},
    "tom_low":  {"radius": .19, "target": 112., "q": 9.,  "kind": "membrane", "modes": 18},
    "tom_floor":{"radius": .22, "target": 82.,  "q": 8.,  "kind": "membrane", "modes": 18},
    "hihat": {"radius": .18, "target": 620., "q": 9.,  "kind": "plate", "modes": 28},
    "crash": {"radius": .23, "target": 390., "q": 12., "kind": "plate", "modes": 32},
}


def circular_laplacian(n=27):
    axis = np.linspace(-1, 1, n)
    step = axis[1] - axis[0]
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    active = xx*xx + yy*yy < .94**2
    index = -np.ones((n, n), int)
    index[active] = np.arange(active.sum())
    rows, cols, data = [], [], []
    for i, j in zip(*np.nonzero(active)):
        row, diagonal = index[i, j], 0.
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            diagonal += 1 / step**2
            ni, nj = i + di, j + dj
            if 0 <= ni < n and 0 <= nj < n and active[ni, nj]:
                rows.append(row); cols.append(index[ni, nj]); data.append(-1 / step**2)
        rows.append(row); cols.append(row); data.append(diagonal)
    return sparse.csr_matrix((data, (rows, cols)), shape=(active.sum(), active.sum())), active, index


def solve_piece(spec, n=27):
    laplacian, active, index = circular_laplacian(n)
    operator = laplacian if spec["kind"] == "membrane" else laplacian @ laplacian
    values, vectors = spla.eigsh(operator.tocsc(), k=spec["modes"], sigma=0, which="LM")
    order = np.argsort(values)
    values, vectors = np.maximum(values[order], 1e-9), vectors[:, order]
    # K phi = omega^2 M phi.  The plate operator is already biharmonic
    # (Laplacian squared), so both operators convert eigenvalue -> omega with
    # sqrt().  A fourth root here incorrectly gave cymbals membrane-like ratios.
    raw = np.sqrt(values)
    frequencies = raw / raw[0] * spec["target"]
    axis = np.linspace(-1, 1, n)
    nodes = []
    for x, y in ((.18, -.13), (.48, .12), (.02, .02)):
        i, j = int(np.argmin(abs(axis-x))), int(np.argmin(abs(axis-y)))
        nodes.append(index[i, j])
    weights = (.72*vectors[nodes[0], :] + .28*vectors[nodes[1], :]) * vectors[nodes[2], :]
    weights /= max(np.max(np.abs(weights)), 1e-12)
    return frequencies, weights, int(active.sum())


def generate_model(path: Path):
    pieces = {}
    for name, spec in SPECS.items():
        frequencies, weights, nodes = solve_piece(spec)
        pieces[name] = {
            "kind": spec["kind"], "radius_m": spec["radius"], "mesh_nodes": nodes,
            "modes": [
                {"hz": round(float(f), 4), "weight": round(float(w), 7),
                 "q": round(spec["q"]*(1+.025*i), 3)}
                for i, (f, w) in enumerate(zip(frequencies, weights))
            ],
        }
    model = {"method": "circular membrane/plate modal FEA", "boundary": "clamped rim", "pieces": pieces}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=2), encoding="utf-8")
    return model


def synth_hit(hit: DrumHit, model, rng, maximum_tail=None):
    piece = model["pieces"][hit.piece]
    tail = {"kick": .75, "snare": .58, "tom_high": .48, "tom_mid": .58,
            "tom_low": .72, "tom_floor": .88, "hihat": .24, "crash": 2.6}[hit.piece]
    if hit.piece == "hihat":
        tail += 1.05 * hit.open_amount
    if maximum_tail is not None:
        tail = min(tail, max(.035, maximum_tail))
    n = int(tail * SR)
    time = np.arange(n) / SR
    velocity = hit.velocity / 127
    signal = np.zeros(n)
    for i, mode in enumerate(piece["modes"]):
        frequency = mode["hz"] * (1 + rng.normal(0, .0012))
        if frequency >= SR*.46:
            continue
        decay = max(.045, mode["q"]/(math.pi*frequency)) * (1 + .35*hit.open_amount)
        signal += mode["weight"]/(1+i*.035) * np.exp(-time/decay) * np.sin(
            2*np.pi*frequency*time + rng.uniform(-math.pi, math.pi)
        )
    noise = rng.standard_normal(n)
    if hit.piece == "kick":
        # Coupled shell-air mode: the head tension relaxes during the stroke,
        # producing the short downward pitch movement of a miked kick.
        phase = 2*np.pi*(51*time + 30*.030*(1-np.exp(-time/.030)))
        air = np.sin(phase) * np.exp(-time/.21)
        sub_head = sosfilt(butter(2, [36, 78], "bandpass", fs=SR, output="sos"), noise)
        # Broad, low-Q head/contact bands provide chest and beater punch
        # without adding fixed oscillator pitches that repeat like a click.
        chest = sosfilt(butter(2, [72, 138], "bandpass", fs=SR, output="sos"), noise)
        knock = sosfilt(butter(2, [105, 235], "bandpass", fs=SR, output="sos"), noise)
        chest *= np.exp(-time/.105)
        knock *= np.exp(-time/.048)
        sub_head *= np.exp(-time/.19)
        click = sosfilt(butter(2, [900, 3600], "bandpass", fs=SR, output="sos"), noise)
        signal = .16*signal*(1-np.exp(-time/.002)) + .20*air + 5.8*sub_head
        signal += 7.2*chest + 3.6*knock
        signal += .24*click*np.exp(-time/.010)
        # Head/preamp nonlinearity supplies harmonics so the impact remains
        # powerful on speakers that cannot reproduce the 50 Hz fundamental.
        signal = .72*signal + .34*np.tanh(2.4*signal)
        signal = sosfilt(butter(2, 5200, "lowpass", fs=SR, output="sos"), signal)
    elif hit.piece == "snare":
        # The lower head and wires respond a fraction after the batter head.
        wires = sosfilt(butter(2, [1200, 9000], "bandpass", fs=SR, output="sos"), noise)
        body_noise = sosfilt(butter(2, [140, 950], "bandpass", fs=SR, output="sos"), noise)
        signal = .10*signal + .42*body_noise*np.exp(-time/.060)
        signal += 1.02*wires*np.exp(-time/.125)*(1-np.exp(-time/.0012))
        signal = sosfilt(butter(2, 105, "highpass", fs=SR, output="sos"), signal)
    elif hit.piece.startswith("tom_"):
        impact = sosfilt(butter(2, [90, 2600], "bandpass", fs=SR, output="sos"), noise)
        signal = .78*signal + .24*impact*np.exp(-time/.026)
        signal = sosfilt(butter(2, 4800, "lowpass", fs=SR, output="sos"), signal)
    else:
        # Broad-band contact radiation complements the inharmonic plate modes.
        metal_hi = sosfilt(butter(2, [4200, 17000], "bandpass", fs=SR, output="sos"), noise)
        metal_lo = sosfilt(butter(2, [1100, 7000], "bandpass", fs=SR, output="sos"), noise)
        shimmer = (metal_hi*.62 + metal_lo*.30) * (1 + .10*np.sin(2*np.pi*7.3*time))
        # Real cymbals radiate many overlapping, rapidly decorrelating modes;
        # keep FEA modes as structure but prevent any one from reading as a bell.
        # Only a trace of explicit modes remains; the plate eigenstructure now
        # colours a broad impact spectrum instead of producing audible chimes.
        signal = .035*signal + (1.02+.18*hit.open_amount)*shimmer*np.exp(-time/(tail*.36))
        signal = sosfilt(butter(2, 650, "highpass", fs=SR, output="sos"), signal)
    # Louder strikes brighten as well as increase level.
    signal *= velocity**1.18
    signal /= max(np.max(np.abs(signal)), 1e-9)
    if maximum_tail is not None:
        fade = min(int(.014*SR), len(signal))
        signal[-fade:] *= np.linspace(1, 0, fade)
    performance = rng.uniform(.92, 1.06) if hit.piece in ("hihat", "snare") else rng.uniform(.97, 1.03)
    levels = {"kick": .88, "snare": .78, "tom_high": .70, "tom_mid": .72,
              "tom_low": .76, "tom_floor": .80, "hihat": .42, "crash": .55}
    return signal * levels[hit.piece] * velocity * performance


def render_drums(model, score, seconds, seed=1973):
    rng = np.random.default_rng(seed)
    stereo = np.zeros((int(seconds*SR), 2))
    pans = {"kick": 0., "snare": .05, "tom_high": -.42, "tom_mid": -.18,
            "tom_low": .20, "tom_floor": .42, "hihat": -.58, "crash": -.18}
    ordered = sorted(score, key=lambda item: item.start)
    for hit_index, hit in enumerate(ordered):
        maximum_tail = None
        if hit.piece == "hihat":
            # A new hi-hat stroke closes/chokes the preceding open pair.
            following = next((other for other in ordered[hit_index+1:] if other.piece == "hihat"), None)
            if following is not None:
                maximum_tail = following.start-hit.start+.018
        audio = synth_hit(hit, model, rng, maximum_tail)
        start = int(hit.start*SR)
        span = min(len(audio), len(stereo)-start)
        if span <= 0:
            continue
        pan = pans[hit.piece] + (rng.uniform(-.055, .055) if hit.piece == "hihat" else 0.)
        stereo[start:start+span, 0] += math.sqrt((1-pan)/2) * audio[:span]
        stereo[start:start+span, 1] += math.sqrt((1+pan)/2) * audio[:span]
    dry = stereo.copy()
    for delay, gain, cross in ((.021, .10, False), (.043, .075, True), (.079, .048, False), (.127, .030, True)):
        n = int(delay*SR)
        stereo[n:] += gain * (dry[:-n, ::-1] if cross else dry[:-n])
    stereo *= .90 / max(np.max(np.abs(stereo)), 1e-9)
    return stereo
