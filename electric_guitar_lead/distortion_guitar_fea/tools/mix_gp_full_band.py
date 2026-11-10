"""Mix independently rendered GP guitar, bass and drum FEA stems."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

from mix_tab2_guitars import read_pcm24, write_pcm24

SR = 48_000


def pad(audio, length):
    return np.pad(audio, ((0, length-len(audio)), (0, 0)))


def main():
    parser = argparse.ArgumentParser()
    for name in ("lead", "rhythm", "bass", "drums", "drum_score", "out"):
        parser.add_argument(f"--{name.replace('_','-')}", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--tempo", type=float, required=True)
    args = parser.parse_args()
    paths = {name: getattr(args, name) for name in ("lead", "rhythm", "bass", "drums")}
    stems = {name: read_pcm24(path) for name, path in paths.items()}
    length = max(map(len, stems.values()))
    lead, rhythm, bass, drums = (pad(stems[name], length) for name in ("lead", "rhythm", "bass", "drums"))

    lead_mid=lead.mean(axis=1); lead_side=(lead[:,0]-lead[:,1])*.5
    lead=np.column_stack([lead_mid+.25*lead_side,lead_mid-.25*lead_side])
    rhythm_mid=rhythm.mean(axis=1); rhythm_side=(rhythm[:,0]-rhythm[:,1])*.5
    rhythm_mid=sosfilt(butter(2,4300,"lowpass",fs=SR,output="sos"),rhythm_mid)
    rhythm=np.column_stack([.64*rhythm_mid+1.15*rhythm_side,.64*rhythm_mid-1.15*rhythm_side])
    bass_mono=sosfilt(butter(2,3300,"lowpass",fs=SR,output="sos"),bass.mean(axis=1))
    bass=np.column_stack([bass_mono,bass_mono])
    drums=sosfilt(butter(2,30,"highpass",fs=SR,output="sos"),drums,axis=0)
    drum_sub=sosfilt(butter(2,145,"lowpass",fs=SR,output="sos"),drums,axis=0)
    drum_air=sosfilt(butter(2,5200,"highpass",fs=SR,output="sos"),drums,axis=0)
    drums=drums+.23*drum_sub+.24*drum_air

    score=json.loads(args.drum_score.read_text(encoding="utf-8"))
    bass_gain=np.ones(length); rhythm_gain=np.ones(length)
    for event in score["events"]:
        start=int(event["start"]*SR)
        if not 0 <= start < length: continue
        if event["piece"]=="kick":
            span=min(int(.105*SR),length-start)
            release=.72+.28*(1-np.exp(-np.arange(span)/(SR*.050)))
            bass_gain[start:start+span]=np.minimum(bass_gain[start:start+span],release)
        elif event["piece"]=="snare":
            span=min(int(.075*SR),length-start)
            release=.84+.16*(1-np.exp(-np.arange(span)/(SR*.032)))
            rhythm_gain[start:start+span]=np.minimum(rhythm_gain[start:start+span],release)
    bass*=bass_gain[:,None]; rhythm*=rhythm_gain[:,None]

    mix=.70*lead+.35*rhythm+.60*bass+.76*drums
    mix=np.tanh(1.12*mix)/np.tanh(1.12)
    mix-=mix.mean(axis=0); mix*=.94/max(np.max(np.abs(mix)),1e-9)
    write_pcm24(args.out,mix)
    manifest={
        "source":args.source,"title":args.title,"tempo_bpm":args.tempo,
        "included_tracks":[
            {"index":0,"role":"lead guitar","stem":str(args.lead)},
            {"index":1,"role":"rhythm guitar","stem":str(args.rhythm)},
            {"index":2,"role":"bass","stem":str(args.bass)},
            {"index":3,"role":"drums","stem":str(args.drums)},
        ],
        "duration_s":length/SR,"peak":float(np.max(np.abs(mix))),
        "rms":float(np.sqrt(np.mean(mix*mix))),"output":str(args.out),
    }
    args.out.with_name(args.out.stem+"_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(manifest,indent=2))


if __name__=="__main__": main()
