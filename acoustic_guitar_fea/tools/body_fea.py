#!/usr/bin/env python3
"""Finite-difference modes of a braced steel-string guitar body.

The top and back are orthotropic thin plates on a guitar-shaped domain.
Spatially varying rigidity from the braces is assembled through strain-energy
operators, keeping the generalized eigenproblem symmetric.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

BODY_LENGTH = 0.985
BODY_WIDTH = 0.405
TOP_H = 0.0028
BACK_H = 0.0026
RHO_SPRUCE = 430.0
RHO_ROSEWOOD = 820.0
E_SPRUCE_LONG = 11.5e9
E_SPRUCE_CROSS = 0.85e9
E_ROSE_LONG = 15.5e9
E_ROSE_CROSS = 1.65e9
NU = 0.32
BRIDGE = (0.50, 0.69)
MIC_LEFT = (0.34, 0.62)
MIC_RIGHT = (0.67, 0.58)


def outline_half_width(y: np.ndarray) -> np.ndarray:
    """Smooth dreadnought-like half-width, normalized to maximum width."""
    yp = np.array([0.0, 0.10, 0.27, 0.43, 0.55, 0.73, 0.90, 1.0])
    wp = np.array([0.30, 0.62, 0.86, 0.60, 0.53, 0.94, 1.00, 0.72])
    return np.interp(y, yp, wp)


def body_mask(nx: int, ny: int) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return np.abs(xx) <= outline_half_width(yy)


def brace_map(nx: int, ny: int, kind: str) -> np.ndarray:
    """Rigidity multiplier from top X-bracing or back ladder braces."""
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    mult = np.ones((nx, ny), dtype=float)
    if kind == "top":
        # X-brace crossing just above the bridge, plus lower tone bars.
        d1 = np.abs((yy - 0.56) - 0.83 * (xx - 0.50))
        d2 = np.abs((yy - 0.56) + 0.83 * (xx - 0.50))
        mult += 9.0 * ((d1 < 0.022) | (d2 < 0.022))
        mult += 4.0 * ((np.abs(yy - 0.73) < 0.016) & (xx < 0.54))
        mult += 4.0 * ((np.abs(yy - 0.80) < 0.016) & (xx > 0.46))
        # Bridge plate.
        mult += 3.5 * (((xx - BRIDGE[0]) / 0.18) ** 2
                       + ((yy - BRIDGE[1]) / 0.045) ** 2 < 1.0)
    else:
        for centre in (0.24, 0.43, 0.64, 0.82):
            mult += 7.0 * (np.abs(yy - centre) < 0.016)
    return mult


def _operators(mask: np.ndarray, dx: float, dy: float):
    """Second derivatives on active nodes with a fixed rim boundary."""
    nx, ny = mask.shape
    index = -np.ones_like(mask, dtype=int)
    active = np.argwhere(mask)
    index[mask] = np.arange(len(active))

    def build(axis: int):
        rows, cols, vals = [], [], []
        step2 = dx * dx if axis == 0 else dy * dy
        for row, (i, j) in enumerate(active):
            rows.append(row); cols.append(row); vals.append(-2.0 / step2)
            for sign in (-1, 1):
                ni, nj = (i + sign, j) if axis == 0 else (i, j + sign)
                if 0 <= ni < nx and 0 <= nj < ny and mask[ni, nj]:
                    rows.append(row); cols.append(index[ni, nj]); vals.append(1.0 / step2)
        return sp.csr_matrix((vals, (rows, cols)), shape=(len(active), len(active)))

    return active, build(0), build(1)


def solve_plate(kind: str, nx: int = 43, ny: int = 65, count: int = 24):
    mask = body_mask(nx, ny)
    dx = BODY_WIDTH / (nx - 1)
    dy = BODY_LENGTH / (ny - 1)
    active, lxx, lyy = _operators(mask, dx, dy)
    brace = brace_map(nx, ny, kind)[mask]

    if kind == "top":
        h, rho = TOP_H, RHO_SPRUCE
        ex, ey = E_SPRUCE_CROSS, E_SPRUCE_LONG
    else:
        h, rho = BACK_H, RHO_ROSEWOOD
        ex, ey = E_ROSE_CROSS, E_ROSE_LONG
    denom = 12.0 * (1.0 - NU * NU)
    dxx = ex * h**3 / denom * brace
    dyy = ey * h**3 / denom * brace
    dxy = np.sqrt(dxx * dyy) * NU
    area = dx * dy

    c = lxx.T @ sp.diags(dxy) @ lyy
    stiffness = area * (
        lxx.T @ sp.diags(dxx) @ lxx
        + lyy.T @ sp.diags(dyy) @ lyy + c + c.T
    )
    mass = sp.identity(len(active), format="csc") * (rho * h * area)
    values, vectors = spla.eigsh(
        stiffness.tocsc(), k=count, M=mass, sigma=-1.0, which="LM"
    )
    freq = np.sqrt(np.maximum(values, 0.0)) / (2.0 * np.pi)
    order = np.argsort(freq)
    shapes = vectors[:, order].T
    # Normalize each shape to unit RMS over the active plate.
    shapes /= np.sqrt(np.mean(shapes * shapes, axis=1, keepdims=True))
    return freq[order], shapes, active, mask.shape


def _sample(shapes, active, shape, point):
    target = np.array([point[0] * (shape[0] - 1), point[1] * (shape[1] - 1)])
    nearest = np.argmin(np.sum((active - target) ** 2, axis=1))
    return shapes[:, nearest]


def make_modes(nx: int = 43, ny: int = 65, plate_modes: int = 44) -> list[dict]:
    modes: list[dict] = []
    for kind in ("top", "back"):
        freq, shapes, active, grid_shape = solve_plate(kind, nx, ny, plate_modes)
        bridge = _sample(shapes, active, grid_shape, BRIDGE)
        left = _sample(shapes, active, grid_shape, MIC_LEFT)
        right = _sample(shapes, active, grid_shape, MIC_RIGHT)
        # Bridge mobility and net volume displacement determine audibility.
        volume = np.abs(np.mean(shapes, axis=1))
        raw = np.abs(bridge) * (0.18 + volume)
        keep = (freq > 70.0) & (freq < 1800.0)
        candidates = np.flatnonzero(keep & (raw > 1e-4 * raw.max()))
        # Preserve the low body fundamentals and the strongest bridge-coupled
        # upper modes. Without the explicit low-mode reservation, ranking only
        # by bridge mobility can discard the main top resonance.
        low = candidates[:min(5, len(candidates))]
        strong = candidates[np.argsort(raw[candidates])[-13:]]
        candidates = np.unique(np.concatenate((low, strong)))
        candidates = candidates[np.argsort(freq[candidates])]
        scale = max(float(raw[candidates].max()), 1e-12)
        for i in candidates:
            modes.append({
                "kind": kind,
                "hz": round(float(freq[i]), 3),
                "q": round(float(np.clip(44.0 * (180.0 / freq[i])**0.30,
                                                 16.0, 52.0)), 2),
                "weight": round(float(raw[i] / scale), 6),
                "bridge": round(float(bridge[i]), 6),
                "left": round(float(left[i]), 6),
                "right": round(float(right[i]), 6),
            })
    # Cavity modes couple top and back. These are acoustic modes, not claimed
    # as plate-FEA results; keeping the provenance explicit matters.
    modes.extend([
        {"kind": "air", "hz": 101.0, "q": 9.0, "weight": 0.92,
         "bridge": 1.0, "left": 0.95, "right": 1.0},
        {"kind": "air", "hz": 205.0, "q": 13.0, "weight": 0.44,
         "bridge": 0.8, "left": 1.0, "right": -0.72},
        {"kind": "air", "hz": 415.0, "q": 18.0, "weight": 0.24,
         "bridge": 0.55, "left": -0.62, "right": 1.0},
    ])
    return sorted(modes, key=lambda m: m["hz"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parents[1] / "model")
    parser.add_argument("--nx", type=int, default=43)
    parser.add_argument("--ny", type=int, default=65)
    parser.add_argument("--plate-modes", type=int, default=44)
    args = parser.parse_args()
    modes = make_modes(args.nx, args.ny, args.plate_modes)
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "finite-difference braced orthotropic top/back plates",
        "geometry_m": {"length": BODY_LENGTH, "width": BODY_WIDTH,
                       "top_thickness": TOP_H, "back_thickness": BACK_H},
        "grid": [args.nx, args.ny],
        "bridge": list(BRIDGE),
        "microphones": {"left": list(MIC_LEFT), "right": list(MIC_RIGHT)},
        "modes": modes,
    }
    path = args.out / "body_modes.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {len(modes)} modes to {path}")
    for m in modes:
        print(f"{m['kind']:>4} {m['hz']:8.2f} Hz  Q={m['q']:5.1f} "
              f"weight={m['weight']:.3f}")


if __name__ == "__main__":
    main()
