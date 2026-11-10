#!/usr/bin/env python3
"""Modal FEA-style solve of an X-braced acoustic-guitar top plate.

The active grid follows a guitar outline, removes the soundhole, and locally
changes thickness over an X brace, tone bars, and bridge patch. The stiffness
matrix is assembled from orthotropic plate strain energy, keeping it symmetric
when material thickness varies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import guitar_physics as phys

# Quarter-sawn Sitka spruce.
E_LONG = 11.5e9
E_CROSS = 0.80e9
G_LR = 0.68e9
NU_LR = 0.36
RHO = 410.0

PLATE_X = 1.02
PLATE_Y = 0.42
PLATE_H = 0.0028
SOUNDHOLE = (0.39, 0.50, 0.049)  # normalised x/y, physical radius in m
BRIDGE = (0.64, 0.50)
LEFT_PICKUP = (0.58, 0.31)
RIGHT_PICKUP = (0.70, 0.69)


def _outline_half_width(x: np.ndarray) -> np.ndarray:
    """Smooth dreadnought-like half-width in metres."""
    upper = 0.165 * np.exp(-((x - 0.27) / 0.20) ** 4)
    lower = 0.205 * np.exp(-((x - 0.72) / 0.27) ** 4)
    waist = 0.060 * np.exp(-((x - 0.49) / 0.10) ** 2)
    end_taper = np.sin(np.pi * np.clip(x, 0.0, 1.0)) ** 0.20
    return np.maximum(0.045, (np.maximum(upper, lower) - waist) * end_taper)


def geometry(nx: int, ny: int):
    x = (np.arange(nx) + 0.5) / nx
    y = (np.arange(ny) + 0.5) / ny
    xx, yy = np.meshgrid(x, y, indexing="ij")
    y_m = (yy - 0.5) * PLATE_Y
    active = np.abs(y_m) <= _outline_half_width(xx)
    hx, hy, radius = SOUNDHOLE
    hole = ((xx - hx) * PLATE_X) ** 2 + ((yy - hy) * PLATE_Y) ** 2 <= radius**2
    active &= ~hole

    # Equivalent bending thickness map. The cube sum preserves the correct
    # h^3 scaling of a brace's bending contribution.
    h3 = np.full((nx, ny), PLATE_H**3)

    def add_brace(distance_m: np.ndarray, width: float, height: float):
        nonlocal h3
        profile = np.clip(1.0 - distance_m / (0.5 * width), 0.0, 1.0)
        h3 += (height**3) * profile**2

    xm = xx * PLATE_X
    ym = (yy - 0.5) * PLATE_Y
    # Two legs of the X crossing above the bridge.
    for slope in (-0.52, 0.52):
        distance = np.abs(ym - slope * (xm - 0.50)) / np.sqrt(1.0 + slope**2)
        add_brace(distance, 0.012, 0.013)
    # Transverse upper-face brace and two asymmetric tone bars.
    add_brace(np.abs(xm - 0.315), 0.010, 0.010)
    add_brace(np.abs(ym - 0.36 * (xm - 0.69) - 0.055), 0.010, 0.009)
    add_brace(np.abs(ym + 0.31 * (xm - 0.73) + 0.062), 0.010, 0.008)
    # Hardwood bridge/bridge plate.
    bridge_patch = (
        (np.abs(xm - BRIDGE[0] * PLATE_X) < 0.035)
        & (np.abs(ym) < 0.085)
    )
    h3 += bridge_patch * 0.0065**3
    h = np.cbrt(h3)
    h[~active] = 0.0
    return active, h


def _operators(active: np.ndarray, dx: float, dy: float):
    nx, ny = active.shape
    ids = -np.ones((nx, ny), dtype=int)
    ids[active] = np.arange(int(active.sum()))
    n = int(active.sum())

    def assemble(stencil):
        rows, cols, data = [], [], []
        for i, j in np.argwhere(active):
            row = ids[i, j]
            for di, dj, value in stencil:
                ii, jj = i + di, j + dj
                if 0 <= ii < nx and 0 <= jj < ny and active[ii, jj]:
                    rows.append(row)
                    cols.append(ids[ii, jj])
                    data.append(value)
                # Missing neighbours are zero-valued boundary/ghost nodes.
        return sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    lxx = assemble(((0, 0, -2.0 / dx**2), (-1, 0, 1.0 / dx**2),
                    (1, 0, 1.0 / dx**2)))
    lyy = assemble(((0, 0, -2.0 / dy**2), (0, -1, 1.0 / dy**2),
                    (0, 1, 1.0 / dy**2)))
    lxy = assemble(((-1, -1, 1.0 / (4 * dx * dy)),
                    (-1, 1, -1.0 / (4 * dx * dy)),
                    (1, -1, -1.0 / (4 * dx * dy)),
                    (1, 1, 1.0 / (4 * dx * dy))))
    return ids, lxx, lyy, lxy


def solve_modes(nx: int = 54, ny: int = 34, count: int = 36):
    active, h = geometry(nx, ny)
    dx, dy = PLATE_X / nx, PLATE_Y / ny
    area = dx * dy
    ids, lxx, lyy, lxy = _operators(active, dx, dy)
    hv = h[active]

    nu_rl = NU_LR * E_CROSS / E_LONG
    factor = hv**3 / 12.0
    d_long = E_LONG * factor / (1.0 - NU_LR * nu_rl)
    d_cross = E_CROSS * factor / (1.0 - NU_LR * nu_rl)
    d_nu = nu_rl * d_long
    d_shear = G_LR * factor

    dl = sp.diags(d_long)
    dc = sp.diags(d_cross)
    dn = sp.diags(d_nu)
    ds = sp.diags(d_shear)
    coupling = lxx.T @ dn @ lyy
    stiffness = area * (
        lxx.T @ dl @ lxx
        + lyy.T @ dc @ lyy
        + coupling
        + coupling.T
        + 4.0 * lxy.T @ ds @ lxy
    )
    stiffness = stiffness.tocsc()
    mass = sp.diags(RHO * hv * area).tocsc()
    eigenvalues, eigenvectors = spla.eigsh(
        stiffness, k=count, M=mass, sigma=-1.0, which="LM"
    )
    freqs = np.sqrt(np.maximum(eigenvalues, 0.0)) / (2.0 * np.pi)
    order = np.argsort(freqs)
    shapes = np.zeros((count, nx, ny))
    for dst, src in enumerate(order):
        shapes[dst][active] = eigenvectors[:, src]
    return freqs[order], shapes, active, h, ids


def _sample_modes(shapes: np.ndarray, coordinate: tuple[float, float]) -> np.ndarray:
    nx, ny = shapes.shape[1:]
    ix = min(nx - 1, max(0, int(coordinate[0] * nx)))
    iy = min(ny - 1, max(0, int(coordinate[1] * ny)))
    if np.allclose(shapes[:, ix, iy], 0.0):
        # Find nearest active grid point if a requested sample lands in a hole.
        candidates = np.argwhere(np.any(np.abs(shapes) > 0.0, axis=0))
        target = np.array([ix, iy])
        ix, iy = candidates[np.argmin(np.sum((candidates - target) ** 2, axis=1))]
    rms = np.sqrt(np.sum(shapes**2, axis=(1, 2)))
    return shapes[:, ix, iy] / np.maximum(rms, 1e-18)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parents[1] / "model")
    parser.add_argument("--modes", type=int, default=36)
    parser.add_argument("--nx", type=int, default=54)
    parser.add_argument("--ny", type=int, default=34)
    args = parser.parse_args()

    freqs, shapes, active, h, _ = solve_modes(args.nx, args.ny, args.modes)
    rms = np.sqrt(np.sum(shapes**2, axis=(1, 2)))
    radiation = np.abs(np.sum(shapes, axis=(1, 2))) / np.maximum(rms, 1e-18)
    bridge = _sample_modes(shapes, BRIDGE)
    left = _sample_modes(shapes, LEFT_PICKUP)
    right = _sample_modes(shapes, RIGHT_PICKUP)

    keep = (freqs > 70.0) & (freqs < 1600.0)
    score = radiation * np.abs(bridge)
    candidate = np.flatnonzero(keep)
    chosen = candidate[np.argsort(score[candidate])[-18:]]
    chosen = chosen[np.argsort(freqs[chosen])]
    norm = max(float(score[chosen].max()), 1e-18)
    stereo_scale = max(float(np.abs(left[chosen]).max()),
                       float(np.abs(right[chosen]).max()), 1e-18)

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "orthotropic irregular X-braced plate energy solve",
        "material": "quarter-sawn Sitka spruce",
        "grid": [args.nx, args.ny],
        "active_nodes": int(active.sum()),
        "top_m": [PLATE_X, PLATE_Y, PLATE_H],
        "soundhole": list(SOUNDHOLE),
        "helmholtz_hz": round(phys.helmholtz_frequency(), 3),
        "top_mass_kg": round(float(np.sum(RHO * h) * PLATE_X / args.nx
                                   * PLATE_Y / args.ny), 4),
        "modes": [
            {
                "hz": round(float(freqs[i]), 3),
                "q": round(float(phys.body_q(freqs[i])), 2),
                "weight": round(float(score[i] / norm), 6),
                "bridge": round(float(bridge[i]), 7),
                "left": round(float(left[i] / stereo_scale), 6),
                "right": round(float(right[i] / stereo_scale), 6),
            }
            for i in chosen
        ],
    }
    path = args.out / "body_modes.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"solved {args.modes} modes on {args.nx}x{args.ny}; "
          f"{active.sum()} active top nodes")
    print(f"top mass {payload['top_mass_kg']:.3f} kg, "
          f"A0 {payload['helmholtz_hz']:.1f} Hz")
    print(f"kept {len(chosen)} radiating bridge-coupled modes -> {path}")
    for mode in payload["modes"]:
        print(f"  {mode['hz']:8.2f} Hz  Q {mode['q']:5.1f}  "
              f"weight {mode['weight']:.4f}")


if __name__ == "__main__":
    main()
