"""Reduced FEA of a 12-inch guitar speaker cone and open-back cabinet panels.

The result is a modal coloration table, not a measured impulse response.  Cone
motion is a tensioned membrane with small bending rigidity; plywood panels are
Kirchhoff plates.  Both are sparse generalized eigenproblems.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _laplacian(mask: np.ndarray, dx: float, dy: float):
    active = np.argwhere(mask)
    index = -np.ones_like(mask, dtype=int)
    index[mask] = np.arange(len(active))
    rows, cols, data = [], [], []
    for row, (i, j) in enumerate(active):
        diagonal = 0.0
        for di, dj, weight in ((-1,0,1/dx**2),(1,0,1/dx**2),(0,-1,1/dy**2),(0,1,1/dy**2)):
            ni, nj = i+di, j+dj
            diagonal -= weight
            if 0 <= ni < mask.shape[0] and 0 <= nj < mask.shape[1] and mask[ni,nj]:
                rows.append(row); cols.append(index[ni,nj]); data.append(weight)
        rows.append(row); cols.append(row); data.append(diagonal)
    return sp.csr_matrix((data,(rows,cols)),shape=(len(active),len(active))), active


def _solve(stiffness, mass, count):
    values, vectors = spla.eigsh(stiffness.tocsc(), k=count, M=mass.tocsc(), sigma=0.0, which="LM")
    freq = np.sqrt(np.maximum(values,0))/(2*np.pi)
    order = np.argsort(freq)
    vectors = vectors[:,order].T
    vectors /= np.sqrt(np.mean(vectors*vectors,axis=1,keepdims=True))
    return freq[order], vectors


def cone_modes(n=33, count=13):
    radius=.139; spacing=2*radius/(n-1)
    axis=np.linspace(-radius,radius,n); xx,yy=np.meshgrid(axis,axis,indexing="ij")
    mask=xx*xx+yy*yy < (.965*radius)**2
    lap,active=_laplacian(mask,spacing,spacing); area=spacing**2
    surface_density=.36
    tension=282.0                         # N/m; fundamental near 78 Hz
    bending=3.2e-3                        # N m; paper-cone breakup stiffness
    stiffness=(-tension*lap + bending*(lap.T@lap))*area
    mass=sp.eye(len(active))*surface_density*area
    freq,shapes=_solve(stiffness,mass,count)
    r=np.sqrt(np.sum((active-(n-1)/2)**2,axis=1))/((n-1)/2)
    drive=shapes @ np.exp(-(r/.34)**2)     # voice-coil region
    center=shapes @ np.exp(-(r/.25)**2)    # on-axis close microphone
    edge=shapes @ np.exp(-((r-.68)/.20)**2)
    scale=max(np.max(np.abs(drive)),1e-12)
    return freq, drive/scale, center/max(np.max(np.abs(center)),1e-12), edge/max(np.max(np.abs(edge)),1e-12)


def panel_modes(nx=23,ny=29,count=10):
    width,height=.44,.52; dx,dy=width/(nx-1),height/(ny-1)
    mask=np.ones((nx,ny),dtype=bool); mask[[0,-1],:]=False; mask[:,[0,-1]]=False
    lap,active=_laplacian(mask,dx,dy); area=dx*dy
    thickness=.018; young=5.2e9; nu=.30; rho=640
    rigidity=young*thickness**3/(12*(1-nu**2))
    stiffness=rigidity*area*(lap.T@lap)
    mass=sp.eye(len(active))*rho*thickness*area
    freq,shapes=_solve(stiffness,mass,count)
    x=active[:,0]/(nx-1); y=active[:,1]/(ny-1)
    drive=shapes @ np.exp(-((x-.5)**2+(y-.52)**2)/.055)
    center=shapes @ np.exp(-((x-.5)**2+(y-.5)**2)/.09)
    edge=shapes @ np.exp(-((x-.18)**2+(y-.55)**2)/.07)
    scale=max(np.max(np.abs(drive)),1e-12)
    return freq,drive/scale,center/max(np.max(np.abs(center)),1e-12),edge/max(np.max(np.abs(edge)),1e-12)


def make_model():
    modes=[]
    for kind,solver in (("cone",cone_modes),("panel",panel_modes)):
        freq,drive,center,edge=solver()
        for f,d,c,e in zip(freq,drive,center,edge):
            if 55 < f < 7200:
                q=np.clip((8.5 if kind=="cone" else 18)*(420/max(f,80))**.13,4.5,24)
                modes.append({"kind":kind,"hz":round(float(f),3),"q":round(float(q),2),
                              "drive_coupling":round(float(d),6),"mic_center":round(float(c),6),
                              "mic_edge":round(float(e),6)})
    return sorted(modes,key=lambda m:m["hz"])


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--out",type=Path,default=Path(__file__).resolve().parents[1]/"model"/"amp_cabinet_modes.json")
    args=parser.parse_args(); modes=make_model()
    payload={"source":"tensioned paper-cone membrane/bending FEA plus plywood Kirchhoff-panel FEA",
             "role":"nonlinear 1x12 open-back guitar-speaker and cabinet modal coloration",
             "modes":modes}
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(f"wrote {len(modes)} amplifier-cabinet modes -> {args.out}")


if __name__=="__main__": main()

