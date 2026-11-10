"""Mix only tab2's distortion lead and background guitar FEA stems."""
from __future__ import annotations

import json
from pathlib import Path
import struct
import wave

import numpy as np
from scipy.signal import butter, sosfilt

ROOT=Path(__file__).resolve().parents[1]
SR=48_000


def read_pcm24(path):
    with wave.open(str(path),"rb") as wav:
        if (wav.getframerate(),wav.getnchannels(),wav.getsampwidth()) != (SR,2,3):
            raise ValueError(f"unexpected WAV format: {path}")
        raw=np.frombuffer(wav.readframes(wav.getnframes()),np.uint8).reshape(-1,6)
    audio=np.empty((len(raw),2),float)
    for channel in range(2):
        b=raw[:,channel*3:(channel+1)*3].astype(np.int32)
        value=b[:,0]|(b[:,1]<<8)|(b[:,2]<<16)
        audio[:,channel]=np.where(value&0x800000,value-0x1000000,value)/(2**23-1)
    return audio


def write_pcm24(path,audio):
    pcm=(np.clip(audio,-1,1)*(2**23-1)).astype(np.int32); raw=bytearray()
    for frame in pcm:
        for value in frame: raw += struct.pack("<i",int(value))[:3]
    with wave.open(str(path),"wb") as wav:
        wav.setnchannels(2);wav.setsampwidth(3);wav.setframerate(SR);wav.writeframes(raw)


def main():
    lead_path=ROOT/"out"/"tab2_lead_guitar_fea.wav"
    background_path=ROOT/"out"/"tab2_background_guitar_fea.wav"
    lead=read_pcm24(lead_path); background=read_pcm24(background_path)
    length=max(len(lead),len(background))
    lead=np.pad(lead,((0,length-len(lead)),(0,0)))
    background=np.pad(background,((0,length-len(background)),(0,0)))
    # Keep the lead solid in the middle; spread and darken the rhythm guitar so
    # its high-gain chords support rather than mask the melodic articulation.
    lead_mid=lead.mean(axis=1); lead_side=(lead[:,0]-lead[:,1])*.5
    lead=np.column_stack([lead_mid+.28*lead_side,lead_mid-.28*lead_side])
    bg_mid=background.mean(axis=1); bg_side=(background[:,0]-background[:,1])*.5
    bg_mid=sosfilt(butter(2,4800,"lowpass",fs=SR,output="sos"),bg_mid)
    background=np.column_stack([.68*bg_mid+1.18*bg_side,.68*bg_mid-1.18*bg_side])
    mix=.82*lead+.46*background
    mix=np.tanh(1.08*mix)/np.tanh(1.08)
    mix-=mix.mean(axis=0);mix*=.94/max(np.max(np.abs(mix)),1e-9)
    output=ROOT/"out"/"tab2_lead_and_background_guitars_fea.wav"
    write_pcm24(output,mix)
    manifest={
        "source":"ref/tab2.gp","title":"Guitar solo","tempo_bpm":163,
        "included_tracks":[
            {"index":0,"name":"Lead Guitar","sound":"Distortion Guitar"},
            {"index":1,"name":"Rhythm Guitar","role":"background","sound":"Distortion Guitar"}
        ],
        "excluded_tracks":[
            {"index":2,"name":"Electric Bass (finger)"},
            {"index":3,"name":"Drums"}
        ],
        "tuning_open_midi_low_to_high":[36,43,48,53,57,62],
        "duration_s":length/SR,"output":str(output)
    }
    (ROOT/"out"/"tab2_guitar_only_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(manifest,indent=2))


if __name__=="__main__":main()
