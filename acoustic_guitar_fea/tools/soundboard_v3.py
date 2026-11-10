#!/usr/bin/env python3
"""A dimensioned, braced steel-string top: Rayleigh--Ritz plate energy solve.

This is a conservative structural model, not a measured instrument. The outside
rim is clamped, the soundhole is free, and each brace contributes its actual mass
and its composite-section directional bending energy. No frequency multiplier is
applied. A finite smooth basis approximates the hole-edge motion; --convergence
reports this approximation's low-mode sensitivity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import eigh

LENGTH, WIDTH, THICKNESS = .510, .405, .0028
E_LONG, E_CROSS, G_LR, NU_LR, RHO = 11.5e9, .80e9, .68e9, .36, 410.
HOLE = (.39, .50, .049)
BRIDGE = (.67, .50)
# A smooth outline, with the longitudinal axis running from neck to tail.
_WIDTH = CubicSpline(
    [0., .07, .24, .42, .48, .62, .75, .88, 1.],
    [.065, .130, .153, .126, .123, .177, .2025, .180, .065],
    bc_type=((1, 1.2), (1, -1.6)),
)
# Endpoints are fractions of the plate bounding box; width and height are metres.
BRACES = [
    ((.24, .20), (.88, .87), .0070, .0120, "X bass"),
    ((.24, .80), (.88, .13), .0070, .0120, "X treble"),
    ((.20, .12), (.20, .88), .0090, .0120, "upper transverse"),
    ((.70, .22), (.83, .62), .0060, .0080, "tone bar 1"),
    ((.82, .21), (.95, .57), .0060, .0070, "tone bar 2"),
]


def _inside(x, y):
    return (np.abs(y) <= _WIDTH(x / LENGTH)) & (
        (x - HOLE[0] * LENGTH)**2 + y**2 > HOLE[2]**2)


def _basis(x, y, mx, my):
    u = x / LENGTH
    v = .5 + y / (2 * _WIDTH(u))
    ax = np.sin(np.pi * u[:, None]) * np.sin(
        np.pi * u[:, None] * np.arange(1, mx + 1)[None, :])
    ay = np.sin(np.pi * v[:, None]) * np.sin(
        np.pi * v[:, None] * np.arange(1, my + 1)[None, :])
    return (ax[:, :, None] * ay[:, None, :]).reshape(len(x), mx * my)


def _basis_curvatures(x, y, mx, my):
    # Small differences differentiate smooth trial functions, not unknown grid
    # displacements. The quadrature grid therefore does not set a ghost boundary.
    eps = 2e-5
    b = _basis(x, y, mx, my)
    xx = (_basis(x + eps, y, mx, my) - 2*b
          + _basis(x - eps, y, mx, my)) / eps**2
    yy = (_basis(x, y + eps, mx, my) - 2*b
          + _basis(x, y - eps, mx, my)) / eps**2
    xy = (_basis(x + eps, y + eps, mx, my)
          - _basis(x + eps, y - eps, mx, my)
          - _basis(x - eps, y + eps, mx, my)
          + _basis(x - eps, y - eps, mx, my)) / (4 * eps**2)
    return b, xx, yy, xy


def _composite_extra(eb, hb, et):
    """EI per unit width added by a perfectly bonded rectangular rib.

    Top and rib have their own Young moduli; the neutral axis is computed from
    their axial stiffness. Density is handled separately, never via an h**3 map.
    """
    z = .5 * (THICKNESS + hb)
    at, ab = et * THICKNESS, eb * hb
    return eb * hb**3 / 12 + at * ab / (at + ab) * z**2


def _brace_fields(x, y, dx, dy):
    mass = np.full(len(x), RHO * THICKNESS)
    fields = []
    for start, end, width, height, name in BRACES:
        p = np.array([start[0]*LENGTH, (start[1]-.5)*WIDTH])
        q = np.array([end[0]*LENGTH, (end[1]-.5)*WIDTH])
        vec = q-p
        length = np.linalg.norm(vec)
        c, s = vec / length
        d = np.zeros(len(x))
        m = np.zeros(len(x))
        # Subcell quadrature resolves braces narrower than the integration grid.
        for ox in (-.375, -.125, .125, .375):
            for oy in (-.375, -.125, .125, .375):
                px, py = x + ox*dx - p[0], y + oy*dy - p[1]
                u = (px*c + py*s) / length
                cross = np.abs(-px*s + py*c)
                on = (u > 0) & (u < 1) & (cross < .5*width)
                # Taper to 2 mm at brace tips. One continuous scallop reduces
                # the lower-bout part of each X brace without rescaling modes.
                hb = .002 + (height-.002)*np.sin(np.pi*np.clip(u, 0, 1))**.65
                if name.startswith("X"):
                    hb *= 1 - .32*np.exp(-((u-.73)/.12)**2)
                et = 1 / (c**4/E_LONG + s**4/E_CROSS
                          + c*c*s*s*(1/G_LR - 2*NU_LR/E_LONG))
                d += on*_composite_extra(E_LONG, hb, et) / 16
                m += on*RHO*hb / 16
        fields.append((d, c, s, name))
        mass += m

    # Rosewood bridge plus maple bridge plate. The bridge grain runs across the
    # top grain. The separate longitudinal/transverse composite sections omit
    # their small Poisson coupling but preserve section mass and each axis' EI.
    for length, width, height, rho, ex, ey, label in [
        (.029, .151, .0070, 760., 1.64e9, 15.8e9, "bridge"),
        (.045, .150, .0025, 650., 1.10e9, 10.0e9, "bridge plate"),
    ]:
        fraction = np.zeros(len(x))
        for ox in (-.375, -.125, .125, .375):
            for oy in (-.375, -.125, .125, .375):
                fraction += ((np.abs(x+ox*dx-BRIDGE[0]*LENGTH) < length/2)
                             & (np.abs(y+oy*dy) < width/2)) / 16
        mass += fraction*rho*height
        fields += [(fraction*_composite_extra(ex, height, E_LONG), 1., 0., label+" x"),
                   (fraction*_composite_extra(ey, height, E_CROSS), 0., 1., label+" y")]
    return mass, fields


def solve_modes(nx=180, ny=142, mx=20, my=16, count=36):
    dx, dy = LENGTH/nx, WIDTH/ny
    xx, yy = np.meshgrid((np.arange(nx)+.5)*dx,
                         (np.arange(ny)+.5)*dy-WIDTH/2, indexing="ij")
    mask = _inside(xx, yy)
    x, y = xx[mask], yy[mask]
    b, lxx, lyy, lxy = _basis_curvatures(x, y, mx, my)
    mass_per_area, ribs = _brace_fields(x, y, dx, dy)
    nu_rl = NU_LR * E_CROSS / E_LONG
    factor = THICKNESS**3 / (12*(1-NU_LR*nu_rl))
    dl, dc, dn = E_LONG*factor, E_CROSS*factor, nu_rl*E_LONG*factor
    ds = G_LR*THICKNESS**3/12
    stiffness = (dl*(lxx.T@lxx) + dc*(lyy.T@lyy)
                 + dn*(lxx.T@lyy + lyy.T@lxx) + 4*ds*(lxy.T@lxy))
    for d, c, s, _ in ribs:
        derivative = c*c*lxx + s*s*lyy + 2*c*s*lxy
        stiffness += derivative.T @ (d[:, None]*derivative)
    stiffness *= dx*dy
    mass = b.T @ (mass_per_area[:, None]*b) * dx*dy
    eigenvalues, vectors = eigh(stiffness, mass, subset_by_index=[0, count-1])
    freq = np.sqrt(np.maximum(eigenvalues, 0)) / (2*np.pi)
    shapes = b @ vectors
    bridge_b = _basis(np.array([BRIDGE[0]*LENGTH]), np.array([0.]), mx, my)
    bridge = (bridge_b @ vectors)[0]
    # Fix an arbitrary modal phase using positive bridge displacement, while
    # preserving signed integrated radiation (cancellation across a mode).
    phase = np.where(bridge >= 0, 1., -1.)
    bridge *= phase
    shapes *= phase[None, :]
    radiation = np.sum(shapes, axis=0)*dx*dy
    residue = bridge*radiation
    return {
        "hz": freq, "bridge": bridge, "radiation": radiation,
        "residue": residue, "x": x, "y": y, "shapes": shapes,
        "area": dx*dy, "mass_kg": float(np.sum(mass_per_area)*dx*dy),
        "spruce_mass_kg": float(len(x)*dx*dy*RHO*THICKNESS),
        "nodes": len(x), "matrix_mass_min": float(np.linalg.eigvalsh(mass)[0]),
    }


def export(result, grid, basis, convergence):
    f = result["hz"]
    norm = max(np.max(np.abs(result["residue"])), 1e-12)
    # Far/near microphone residues include spatial retardation: each element's
    # displacement accelerates air, hence acoustic pressure has an omega**2
    # factor supplied by the renderer, not included in these geometry residues.
    mic = []
    for mx, my, mz in [(.25, -.18, .50), (.35, .18, .50)]:
        dist = np.sqrt((result["x"]-mx)**2 + (result["y"]-my)**2 + mz*mz)
        propagation = np.exp(-2j*np.pi*dist[:, None]*f[None, :]/343.) / dist[:, None]
        mic.append(np.sum(result["shapes"]*propagation, axis=0)*result["area"])
    # Rigid-wall A0 only: wall coupling is not resolved by this structural solve.
    volume = .014
    a0 = 343/(2*np.pi)*np.sqrt(np.pi*HOLE[2]**2/(volume*(THICKNESS+1.7*HOLE[2])))
    # Couple the moving top to one air slug through cavity compressibility.
    # Both top displacement and outward hole flow increase cavity volume deficit,
    # so the cavity potential energy is .5*kappa*(a @ q)**2. The air coordinate
    # is mass normalized just like the structural coordinates. Back/ribs remain
    # rigid. This produces coupled low modes without a compliance multiplier.
    rho_air, air_q = 1.2, 14.
    area_hole = np.pi*HOLE[2]**2
    neck_length = THICKNESS + 1.7*HOLE[2]
    a_air = np.sqrt(area_hole/(rho_air*neck_length))
    volume_vector = np.r_[result["radiation"], a_air]
    coupled_k = (np.diag(np.r_[(2*np.pi*f)**2, 0.])
                 + rho_air*343.**2/volume*np.outer(volume_vector, volume_vector))
    coupled_ev, coupled_u = eigh(coupled_k)
    coupled_omega = np.sqrt(coupled_ev)
    coupled_f = coupled_omega/(2*np.pi)
    force = np.r_[result["bridge"], 0.] @ coupled_u
    radiation = volume_vector @ coupled_u
    residue = force*radiation
    coupled_norm = max(float(np.max(np.abs(residue))), 1e-12)
    dry_q = np.clip(30*(180/f)**.22, 10, 40)
    assumed_damping = np.r_[2*np.pi*f/dry_q, 2*np.pi*a0/air_q]
    coupled_q = coupled_omega / (assumed_damping @ (coupled_u**2))
    coupled_mic = []
    # Retarded spatial radiation kernels, generated from geometry rather than
    # a measured impulse response. Linear delay interpolation preserves each
    # signed area's DC residue and gives strictly positive acoustic travel.
    radiation_rate = 48000
    radiation_kernels = []
    top_shapes = result["shapes"] @ coupled_u[:-1, :]
    for j, (mx, my, mz) in enumerate([(.25, -.18, .50), (.35, .18, .50)]):
        dist = np.sqrt((result["x"]-mx)**2 + (result["y"]-my)**2 + mz*mz)
        top_shapes = result["shapes"] @ coupled_u[:-1, :]
        propagation = np.exp(-2j*np.pi*dist[:, None]*coupled_f[None, :]/343.) / dist[:, None]
        top_mic = np.sum(top_shapes*propagation, axis=0)*result["area"]
        hole_distance = np.sqrt((HOLE[0]*LENGTH-mx)**2 + my*my + mz*mz)
        air_mic = (a_air*coupled_u[-1, :]*np.exp(-2j*np.pi*hole_distance*coupled_f/343.)
                   / hole_distance)
        coupled_mic.append(force*(top_mic+air_mic)/coupled_norm)
        delays = dist / 343. * radiation_rate
        first = np.floor(delays).astype(int)
        fractions = delays - first
        hole_delay = hole_distance / 343. * radiation_rate
        hole_first = int(np.floor(hole_delay))
        length = max(int(first.max()) + 2, hole_first + 2)
        kernels = np.zeros((len(coupled_f), length))
        for i in range(len(coupled_f)):
            weights = force[i]*top_shapes[:, i]*result["area"]/dist
            np.add.at(kernels[i], first, weights*(1.-fractions))
            np.add.at(kernels[i], first+1, weights*fractions)
            hole_weight = force[i]*a_air*coupled_u[-1, i]/hole_distance
            kernels[i, hole_first] += hole_weight*(1.-(hole_delay-hole_first))
            kernels[i, hole_first+1] += hole_weight*(hole_delay-hole_first)
        radiation_kernels.append(kernels)
    return {
        "source": "orthotropic clamped-rim/free-hole Ritz top + directional composite braces",
        "measured": False, "frequency_scale": 1.0,
        "top_m": [LENGTH, WIDTH, THICKNESS], "grid": list(grid),
        "basis": list(basis), "active_quadrature_points": result["nodes"],
        "soundhole": list(HOLE), "bridge_coordinate": list(BRIDGE),
        "top_mass_kg": round(result["mass_kg"], 6),
        "unbraced_spruce_mass_kg": round(result["spruce_mass_kg"], 6),
        "rigid_wall_helmholtz_hz": round(float(a0), 3),
        "cavity_volume_m3": volume,
        "coupling": "mass-normalized dry top plus one air-slug coordinate; K_air=(rho*c^2/V)*a*a.T",
        "q_source": "assumed wood/radiation loss curve and air-slug Q=14; not identified from measurements",
        "radiation_kernel_rate": radiation_rate,
        "radiation_kernel_units": "bridge*surface_area/r; pressure uses rho_air/(2*pi) times modal acceleration",
        "residue_units": "bridge[kg^-1/2] times radiation[m^2 kg^-1/2]",
        "limitations": [
            "The dry modes omit air coupling; coupled_modes include one lumped cavity but no moving back/ribs or higher air modes.",
            "Smooth finite basis approximates the natural free soundhole edge.",
            "Bridge and patch use independent axis composite sections; torsion and their mutual laminate coupling are omitted.",
            "Brace crossing and bridge-over-brace composite interaction are approximated by summed energies.",
            "Q and microphone radiation are modeling assumptions, not measured guitar transfer functions.",
        ],
        "convergence": convergence,
        "method_references": [
            "https://dael.euracoustics.org/confs/fa2023/data/articles/000627.pdf",
            "https://www.sciencedirect.com/science/article/abs/pii/S0045794904002974",
            "https://digital.lib.washington.edu/researchworks/items/3d118604-f2a2-4a3d-8629-64887a19b83c",
        ],
        "modes": [{
            "hz": round(float(f[i]), 3),
            "q": round(float(np.clip(30*(180/f[i])**.22, 10, 40)), 2),
            "weight": round(float(abs(result["residue"][i])/norm), 7),
            "signed_weight": round(float(result["residue"][i]/norm), 7),
            "bridge": round(float(result["bridge"][i]), 7),
            "integrated_radiation": round(float(result["radiation"][i]), 8),
            "left": round(float((result["bridge"][i]*mic[0][i]).real/norm), 7),
            "right": round(float((result["bridge"][i]*mic[1][i]).real/norm), 7),
            "left_imag": round(float((result["bridge"][i]*mic[0][i]).imag/norm), 7),
            "right_imag": round(float((result["bridge"][i]*mic[1][i]).imag/norm), 7),
        } for i in range(len(f))],
        "coupled_modes": [{
            "hz": round(float(coupled_f[i]), 3),
            "q": round(float(coupled_q[i]), 2),
            "weight": round(float(abs(residue[i])/coupled_norm), 7),
            "signed_weight": round(float(residue[i]/coupled_norm), 7),
            "bridge": round(float(force[i]), 7),
            "integrated_radiation": round(float(radiation[i]), 8),
            "air_participation": round(float(coupled_u[-1, i]**2), 7),
            "left": round(float(coupled_mic[0][i].real), 7),
            "right": round(float(coupled_mic[1][i].real), 7),
            "left_imag": round(float(coupled_mic[0][i].imag), 7),
            "right_imag": round(float(coupled_mic[1][i].imag), 7),
            "radiation_left": np.round(radiation_kernels[0][i], 12).tolist(),
            "radiation_right": np.round(radiation_kernels[1][i], 12).tolist(),
        } for i in range(len(coupled_f))],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1]/"model"/"body_modes_v3.json")
    parser.add_argument("--nx", type=int, default=180)
    parser.add_argument("--ny", type=int, default=142)
    parser.add_argument("--mx", type=int, default=20)
    parser.add_argument("--my", type=int, default=16)
    parser.add_argument("--modes", type=int, default=36)
    parser.add_argument("--convergence", action="store_true")
    args = parser.parse_args()
    result = solve_modes(args.nx, args.ny, args.mx, args.my, args.modes)
    convergence = []
    if args.convergence:
        for nx, ny, mx, my in [(args.nx, args.ny, args.mx-4, args.my-2),
                               (args.nx+36, args.ny+28, args.mx, args.my)]:
            other = solve_modes(nx, ny, mx, my, args.modes)
            convergence.append({"grid": [nx, ny], "basis": [mx, my],
                                "first_10_hz": np.round(other["hz"][:10], 3).tolist(),
                                "first_10_relative_change": np.round(other["hz"][:10]/result["hz"][:10]-1, 5).tolist(),
                                "top_mass_kg": round(other["mass_kg"], 6)})
    payload = export(result, (args.nx, args.ny), (args.mx, args.my), convergence)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ["top_mass_kg", "unbraced_spruce_mass_kg", "rigid_wall_helmholtz_hz", "convergence"]}, indent=2))
    print("First frequencies:", np.round(result["hz"][:16], 2))
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
