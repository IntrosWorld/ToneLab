#!/usr/bin/env python3
"""Finite-difference modal analysis of a ribbed grand-piano soundboard.

The soundboard is the piano's loudspeaker.  Its low modes put audible formants
into every note, and those formants are what make a rendered piano sound like
a wooden box with strings in it rather than a bank of sine oscillators.

A bare rectangular orthotropic plate has a closed-form modal solution, so
solving it numerically would gain nothing.  The reason for a numerical solve
here is the *ribs*: a real soundboard carries a fan of spruce ribs glued across
the grain, and that spatially varying stiffness has no closed form.  This
module discretises the orthotropic Kirchhoff plate equation

    D1 w_xxxx + 2 D3 w_xxyy + D2 w_yyyy = rho h omega^2 w

on a uniform grid with simply-supported edges, with D and h varying over the
rib pattern, and solves the resulting sparse generalised eigenproblem.

Material constants are Sitka spruce, the standard soundboard timber.  Nothing
here is measured from audio: it is a structural model of a wooden plate.

Output is a small table of modal frequencies and radiation weights that feeds
the body-resonator bank in the RTL.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ---------------------------------------------------------------------------
# Sitka spruce, quarter-sawn, as used for piano soundboards.
# ---------------------------------------------------------------------------
E_LONG = 11.5e9      # Pa, along the grain
E_CROSS = 0.90e9     # Pa, across the grain
G_LR = 0.72e9        # Pa, shear modulus
NU_LR = 0.37         # major Poisson ratio
RHO = 400.0          # kg/m^3

# Concert-grand soundboard geometry.
PLATE_X = 1.40       # m, along the grain
PLATE_Y = 1.52       # m, across the grain
PLATE_H = 0.009      # m, board thickness
RIB_COUNT = 12
RIB_HEIGHT = 0.020   # m, rib depth above the board
RIB_WIDTH = 0.025

# Microphone positions over the board, as fractions of the plate.
LEFT_PICKUP = (0.33, 0.30)
RIGHT_PICKUP = (0.64, 0.71)    # m


def _rib_thickness(nx: int, ny: int) -> np.ndarray:
    """Effective thickness map: bare board plus a fan of cross-grain ribs.

    Ribs run roughly perpendicular to the grain and are spaced across the
    board.  Their bending contribution is folded into a local effective
    thickness, which is the usual smeared-stiffness treatment for a ribbed
    plate and keeps the plate operator fourth-order.
    """
    h = np.full((nx, ny), PLATE_H)
    # Rib centres spread across the x axis, angled slightly like a real fan.
    centres = np.linspace(0.08, 0.92, RIB_COUNT)
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    half = 0.5 * RIB_WIDTH / PLATE_X
    for c in centres:
        # A small skew makes the fan non-parallel, which splits mode pairs
        # that would otherwise be degenerate on a perfect rectangle.
        centre = c + 0.035 * (yy - 0.5)
        mask = np.abs(xx - centre) < half
        # Parallel-axis: a rib of height RIB_HEIGHT raises local bending
        # stiffness as if the board were substantially thicker.
        h = np.where(mask, (PLATE_H ** 3 + RIB_HEIGHT ** 3) ** (1.0 / 3.0), h)
    return h


def _rigidities(h: np.ndarray):
    """Orthotropic bending rigidities for a thickness map."""
    nu_rl = NU_LR * E_CROSS / E_LONG
    denom = 1.0 - NU_LR * nu_rl
    factor = h ** 3 / 12.0
    d1 = E_LONG * factor / denom
    d2 = E_CROSS * factor / denom
    dk = G_LR * factor
    d3 = d1 * nu_rl + 2.0 * dk
    return d1, d2, d3


def solve_modes(nx: int = 56, ny: int = 60, count: int = 40):
    """Solve for the lowest `count` soundboard modes.

    The stiffness matrix is assembled from the plate strain energy

        U = 1/2 integral [ D1 w_xx^2 + 2 nu D1 w_xx w_yy
                           + D2 w_yy^2 + 4 Dk w_xy^2 ] dA

    rather than by discretising the fourth-order operator directly.  Writing it
    this way keeps K symmetric positive semi-definite even though the ribs make
    D vary over the plate, which a direct D(x,y)*stencil assembly does not --
    that produces a non-symmetric matrix and spurious near-zero eigenvalues.

    Returns (frequencies_hz, mode_shapes) on the interior grid.
    """
    h = _rib_thickness(nx, ny)
    d1, d2, _ = _rigidities(h)
    nu_rl = NU_LR * E_CROSS / E_LONG
    dk = G_LR * h ** 3 / 12.0

    dx = PLATE_X / (nx + 1)
    dy = PLATE_Y / (ny + 1)
    area = dx * dy

    # Simply-supported edges mean w = 0 on the boundary, so the interior
    # difference operators simply drop their out-of-range neighbours.
    def second_difference(n, step):
        return sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(n, n)) / step ** 2

    def first_difference(n, step):
        return sp.diags([-1.0, 1.0], [-1, 1], shape=(n, n)) / (2.0 * step)

    eye_x, eye_y = sp.identity(nx), sp.identity(ny)
    lxx = sp.kron(second_difference(nx, dx), eye_y).tocsr()
    lyy = sp.kron(eye_x, second_difference(ny, dy)).tocsr()
    lxy = sp.kron(first_difference(nx, dx), first_difference(ny, dy)).tocsr()

    diag_d1 = sp.diags(d1.reshape(-1))
    diag_d2 = sp.diags(d2.reshape(-1))
    diag_nu = sp.diags((nu_rl * d1).reshape(-1))
    diag_dk = sp.diags(dk.reshape(-1))

    coupling = lxx.T @ diag_nu @ lyy
    stiffness = area * (
        lxx.T @ diag_d1 @ lxx
        + lyy.T @ diag_d2 @ lyy
        + coupling
        + coupling.T
        + 4.0 * (lxy.T @ diag_dk @ lxy)
    )
    stiffness = stiffness.tocsc()
    mass_matrix = sp.diags((RHO * h * area).reshape(-1)).tocsc()

    # Shift just below zero: sigma exactly at 0 can hit a singular factorisation.
    eigenvalues, eigenvectors = spla.eigsh(
        stiffness, k=count, M=mass_matrix, sigma=-1.0, which="LM"
    )
    eigenvalues = np.maximum(eigenvalues, 0.0)
    freqs = np.sqrt(eigenvalues) / (2.0 * np.pi)
    order = np.argsort(freqs)
    return freqs[order], eigenvectors[:, order].T.reshape(count, nx, ny)


def radiation_weights(shapes: np.ndarray) -> np.ndarray:
    """How strongly each mode couples to the air.

    A mode with many sign changes pumps air in and out in adjacent cells and
    cancels itself acoustically, so net volume displacement is the right
    first-order measure of how loudly a mode radiates.
    """
    net = np.abs(shapes.sum(axis=(1, 2)))
    rms = np.sqrt((shapes ** 2).sum(axis=(1, 2)))
    weight = net / np.maximum(rms, 1e-18)
    return weight / max(weight.max(), 1e-18)


def modal_q(freqs: np.ndarray) -> np.ndarray:
    """Quality factor of each body mode.

    Wood is lossy and the board is loaded by the rim, the ribs and the air, so
    body modes are broad.  Losses rise with frequency, so Q falls.
    """
    return np.clip(48.0 * (200.0 / np.maximum(freqs, 1.0)) ** 0.25, 12.0, 70.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "rtl")
    parser.add_argument("--modes", type=int, default=32)
    parser.add_argument("--nx", type=int, default=56)
    parser.add_argument("--ny", type=int, default=60)
    args = parser.parse_args()

    freqs, shapes = solve_modes(args.nx, args.ny, args.modes)
    weights = radiation_weights(shapes)
    qs = modal_q(freqs)

    # Keep the audible, well-radiating body modes.
    keep = (freqs > 45.0) & (freqs < 1400.0)
    freqs, weights, qs = freqs[keep], weights[keep], qs[keep]
    order = np.argsort(-weights)[:16]
    order = order[np.argsort(freqs[order])]
    freqs, weights, qs = freqs[order], weights[order], qs[order]
    weights = weights / max(weights.max(), 1e-18)

    # Two microphone positions over the board.  Because each mode has a
    # different shape, two points on the plate see genuinely different mixtures
    # of the modes -- which is where a real instrument's stereo width comes
    # from.  Panning a mono signal cannot reproduce this.
    kept_shapes = shapes[keep][order]
    picks = []
    for fx, fy in (LEFT_PICKUP, RIGHT_PICKUP):
        ix = min(args.nx - 1, max(0, int(fx * args.nx)))
        iy = min(args.ny - 1, max(0, int(fy * args.ny)))
        amp = kept_shapes[:, ix, iy]
        rms = np.sqrt((kept_shapes ** 2).sum(axis=(1, 2)))
        picks.append(amp / np.maximum(rms, 1e-18))
    left_pick, right_pick = picks
    scale = max(np.abs(left_pick).max(), np.abs(right_pick).max(), 1e-18)
    left_pick, right_pick = left_pick / scale, right_pick / scale

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "finite-difference orthotropic ribbed plate, Sitka spruce",
        "plate_m": [PLATE_X, PLATE_Y, PLATE_H],
        "ribs": RIB_COUNT,
        "pickups": {"left": list(LEFT_PICKUP), "right": list(RIGHT_PICKUP)},
        "modes": [
            {
                "hz": round(float(f), 3),
                "weight": round(float(w), 5),
                "q": round(float(q), 2),
                "left": round(float(lv), 5),
                "right": round(float(rv), 5),
            }
            for f, w, q, lv, rv in zip(freqs, weights, qs, left_pick, right_pick)
        ],
    }
    path = args.out / "soundboard_modes.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"solved {args.modes} plate modes on a {args.nx}x{args.ny} grid")
    print(f"kept {len(freqs)} radiating body modes -> {path}")
    for f, w, q, lv, rv in zip(freqs, weights, qs, left_pick, right_pick):
        print(f"   {f:8.2f} Hz   weight {w:6.4f}   Q {q:5.1f}   "
              f"L {lv:+7.4f}  R {rv:+7.4f}")


if __name__ == "__main__":
    main()
