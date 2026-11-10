"""Small self-contained body-plate + neck-beam FEA modal solve."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def body_mask(nx: int, ny: int) -> np.ndarray:
    x = np.linspace(-1, 1, nx)[:, None]
    y = np.linspace(0, 1, ny)[None, :]
    yp = [0, .1, .25, .42, .58, .78, .92, 1]
    wp = [.34, .55, .90, .71, .79, 1, .85, .47]
    width = np.interp(np.broadcast_to(y, (nx, ny)), yp, wp)
    outline = np.abs(np.broadcast_to(x, (nx, ny))) <= width
    neck_route = (np.abs(x) < .17) & (y < .12)
    pickups = (np.abs(x) < .25) & ((np.abs(y - .30) < .04) | (np.abs(y - .64) < .04))
    control = ((x - .43) / .23) ** 2 + ((y - .55) / .16) ** 2 < 1
    return outline & ~neck_route & ~pickups & ~control


def body_modes(nx=35, ny=49, count=22):
    mask = body_mask(nx, ny)
    active = np.argwhere(mask)
    index = -np.ones_like(mask, dtype=int)
    index[mask] = np.arange(len(active))
    dx, dy = .335 / (nx - 1), .465 / (ny - 1)

    def derivative(axis, step):
        rows, cols, data = [], [], []
        for row, (i, j) in enumerate(active):
            neighbours = 0
            for sign in (-1, 1):
                ni, nj = (i + sign, j) if axis == 0 else (i, j + sign)
                if 0 <= ni < nx and 0 <= nj < ny and mask[ni, nj]:
                    rows.append(row); cols.append(index[ni, nj]); data.append(1 / step**2)
                    neighbours += 1
            rows.append(row); cols.append(row); data.append(-neighbours / step**2)
        return sp.csr_matrix((data, (rows, cols)), shape=(len(active), len(active)))

    lap = derivative(0, dx) + derivative(1, dy)
    rigidity = 10.5e9 * .044**3 / (12 * (1 - .32**2))
    area = dx * dy
    stiffness = rigidity * area * (lap.T @ lap) + sp.eye(len(active)) * .02
    mass = sp.eye(len(active), format="csc") * 455 * .044 * area
    values, vectors = spla.eigsh(stiffness.tocsc(), k=count + 3, M=mass, sigma=-1, which="LM")
    freq = np.sqrt(np.maximum(values, 0)) / (2 * np.pi)
    order = np.argsort(freq)
    freq, vectors = freq[order], vectors[:, order].T
    keep = freq > 35
    return freq[keep][:count], vectors[keep][:count], active, mask


def neck_modes(elements=18, count=8):
    length = .475 / elements
    dofs = 2 * (elements + 1)
    inertia = .056 * .023**3 / 12
    area = .056 * .023
    k0, m0 = 11.8e9 * inertia / length**3, 690 * area * length / 420
    ke = k0 * np.array([[12,6*length,-12,6*length],[6*length,4*length**2,-6*length,2*length**2],[-12,-6*length,12,-6*length],[6*length,2*length**2,-6*length,4*length**2]])
    me = m0 * np.array([[156,22*length,54,-13*length],[22*length,4*length**2,13*length,-3*length**2],[54,13*length,156,-22*length],[-13*length,-3*length**2,-22*length,4*length**2]])
    stiffness, mass = np.zeros((dofs, dofs)), np.zeros((dofs, dofs))
    for e in range(elements):
        idx = np.arange(2*e, 2*e+4)
        stiffness[np.ix_(idx, idx)] += ke; mass[np.ix_(idx, idx)] += me
    values, vectors = scipy.linalg.eigh(stiffness[2:, 2:], mass[2:, 2:])
    freq = np.sqrt(np.maximum(values, 0)) / (2*np.pi)
    return freq[:count], vectors[0::2, :count].T


def make_modes():
    freq, shapes, active, mask = body_modes()
    shapes /= np.sqrt(np.mean(shapes**2, axis=1, keepdims=True))
    def sample(point):
        target = np.array([point[0]*(mask.shape[0]-1), point[1]*(mask.shape[1]-1)])
        return shapes[:, np.argmin(np.sum((active-target)**2, axis=1))]
    bridge, joint = sample((.50,.73)), sample((.50,.08))
    coupling = np.abs(bridge-joint); coupling /= coupling.max()
    modes = [{"kind":"body", "hz":round(float(f),3), "q":round(float(np.clip(70*(180/f)**.18,24,84)),2), "string_coupling":round(float(c),6)} for f,c in zip(freq,coupling) if f < 6500]
    nf, ns = neck_modes(); raw = np.abs(ns[:,-1]-ns[:,1]); raw /= raw.max()
    modes += [{"kind":"neck", "hz":round(float(f),3), "q":round(float(np.clip(56*(160/f)**.14,22,70)),2), "string_coupling":round(float(.72*c),6)} for f,c in zip(nf,raw) if 35 < f < 6500]
    return sorted(modes, key=lambda m:m["hz"]), mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1]/"model"/"structure_modes.json")
    args = parser.parse_args()
    modes, mask = make_modes()
    payload = {"source":"irregular alder equivalent-plate FEA plus maple Euler-Bernoulli neck FEA", "role":"bridge/nut mechanical admittance controlling string losses", "grid":[35,49], "active_nodes":int(mask.sum()), "modes":modes}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {len(modes)} modes -> {args.out}")


if __name__ == "__main__":
    main()

