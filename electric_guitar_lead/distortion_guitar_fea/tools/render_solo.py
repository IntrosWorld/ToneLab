"""Render an original expressive A-minor-pentatonic electric-guitar solo."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct
import wave

import numpy as np
from scipy.signal import butter, sosfilt

import guitar_physics as phys
import amp_effects

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Gesture:
    start: float
    duration: float
    midi: float
    target: float | None = None
    velocity: int = 100
    articulation: str = "pick"
    vibrato_cents: float = 0.0
    string_index: int | None = None


def solo_score() -> list[Gesture]:
    """Motif-driven A-minor-pentatonic solo at an implied 100 BPM."""
    g = []
    def add(t,d,n,art="pick",target=None,v=100,vib=0):
        g.append(Gesture(t,d,n,target,v,art,vib))
    # Phrase 1: singable A-C-D-E motif and an answered high bend.
    add(.18,.44,69,"accent",v=110,vib=7); add(.68,.30,69,"hammer_on",target=72,v=88)
    add(1.02,.28,72,"hammer_on",target=74,v=87); add(1.34,.74,76,"pick",v=107,vib=12)
    add(2.24,.62,69,"slide",target=76,v=104,vib=8); add(2.92,.27,79,"accent",v=108)
    add(3.22,.31,79,"pull_off",target=76,v=86); add(3.58,.96,79,"bend_up",target=81,v=117,vib=17)
    # Phrase 2: descending answer, rhythmic space, then bluesy double-stops.
    add(4.72,.28,81,"accent",v=109); add(5.02,.25,79,"pull_off",target=76,v=87)
    add(5.29,.24,76,"pull_off",target=74,v=84); add(5.57,.62,72,"legato_slide",target=69,v=91,vib=9)
    add(6.38,.46,72,"accent",v=103,vib=8); add(6.38,.44,76,"pick",v=88,vib=6)
    add(6.92,.43,74,"pick",v=100); add(6.92,.41,79,"pick",v=86)
    add(7.43,.74,76,"bend_release",target=78,v=111,vib=12)
    # Phrase 3: fluid pentatonic run with deliberate accents, not a scale dump.
    run=[(69,"accent",104),(72,"hammer_on",88),(74,"hammer_on",86),(76,"pick",101),
         (79,"slide",96),(81,"pick",108),(79,"pull_off",86),(76,"pull_off",84),
         (74,"accent",104),(72,"pull_off",84),(69,"legato",81),(72,"pick",98)]
    t=8.40
    for i,(n,art,v) in enumerate(run):
        target=81 if (n,art)==(79,"slide") else None
        add(t+i*.19,.235,n,art,target=target,v=v)
    add(10.78,.82,74,"slide",target=81,v=110,vib=13)
    # Phrase 4: lyrical upper-register motif, tap/legato color and a held bend.
    add(11.76,.34,81,"accent",v=111); add(12.12,.28,81,"pull_off",target=79,v=87)
    add(12.43,.30,76,"pick",v=99); add(12.78,.66,79,"prebend_release",target=77,v=111,vib=12)
    add(13.55,.25,76,"tap",target=81,v=91); add(13.83,.24,81,"pull_off",target=76,v=82)
    add(14.10,.26,74,"hammer_on",target=76,v=85); add(14.40,1.08,79,"bend_up",target=81,v=119,vib=18)
    # Climax: harmonic squeal, short call, then the opening motif resolves to A.
    add(15.68,.70,81,"pinch_harmonic",v=120,vib=20)
    add(16.48,.28,79,"accent",v=111); add(16.79,.26,76,"pull_off",target=74,v=86)
    add(17.08,.27,72,"pick",v=101); add(17.39,.56,69,"slide",target=76,v=108,vib=10)
    add(18.10,.25,79,"accent",v=111); add(18.39,.25,79,"pull_off",target=76,v=87)
    add(18.68,.30,74,"pull_off",target=72,v=85); add(19.04,1.62,69,"accent",v=114,vib=18)
    return g


class LeadGuitar:
    def __init__(self, sample_rate=48_000, seed=1978, tempo_bpm=100.0, delay_level=1.0):
        self.sr = int(sample_rate); self.rng = np.random.default_rng(seed)
        self.tempo_bpm=float(tempo_bpm); self.delay_level=float(delay_level)
        path = ROOT/"model"/"structure_modes.json"
        self.modes = json.loads(path.read_text(encoding="utf-8"))["modes"]

    def gesture(self, event: Gesture, tail=.75):
        target = event.midi if event.target is None else event.target
        total = int((event.duration+tail)*self.sr); t = np.arange(total)/self.sr
        active_t = np.minimum(t, event.duration)
        pitch = phys.pitch_curve(event.midi,target,event.articulation,active_t,event.duration,event.vibrato_cents)
        count = phys.partial_count(max(event.midi,target),self.sr,event.string_index)
        numbers = np.arange(1,count+1,dtype=float)
        amps = phys.partial_amplitudes(event.midi,event.velocity,count)
        if event.articulation == "pinch_harmonic":
            # Thumb contact suppresses the fundamental and selects the 4th/5th
            # partial region; the string physics still sets exact frequencies.
            amps *= .05 + 1.8*np.exp(-.5*((numbers-4.5)/1.15)**2)
            amps /= max(float(np.max(amps)),1e-9)
        pickup = phys.pickup_response(numbers)
        f0 = phys.fundamental(pitch); b = phys.inharmonicity(event.midi,event.string_index)
        nominal = numbers*float(phys.fundamental(event.midi))*np.sqrt(1+b*numbers*numbers)
        t60 = phys.coupled_t60(nominal,event.midi,self.modes,event.string_index)
        excitation = phys.ARTICULATION_EXCITATION[event.articulation]
        signal = np.zeros(total)
        for i,n in enumerate(numbers):
            inst_f = n*f0*np.sqrt(1+b*n*n)
            if np.max(inst_f) > .48*self.sr: continue
            phase = np.cumsum(2*np.pi*inst_f/self.sr) + self.rng.uniform(-.04,.04)
            decay = np.exp(-math.log(1000)*t/t60[i])
            voltage = excitation*amps[i]*pickup[i]*(nominal[i]/nominal[0])**.46
            if abs(voltage) > 1e-5:
                signal += voltage*decay*np.sin(phase)
                signal += .16*voltage*np.exp(-1.5*math.log(1000)*t/t60[i])*np.sin(phase*2**(.6/1200)+.31)
        # The measured reference blooms into the note body rather than leading
        # with a spike. Finite pick release plus magnetic/electrical bandwidth
        # are represented by a 2.2 ms rise.
        signal *= 1.0-np.exp(-t/.0090)
        # Articulation-specific, equation/noise-derived contact energy.
        vel = event.velocity/127
        click_len = min(total,int((.0028 if event.articulation in ("pick","accent","bend_up","bend_release","pinch_harmonic") else .005)*self.sr))
        if click_len:
            noise = np.diff(np.r_[0,self.rng.standard_normal(click_len)])
            noise = sosfilt(butter(1,[900,4800],"bandpass",fs=self.sr,output="sos"),noise)
            signal[:click_len] += noise*np.exp(-np.arange(click_len)/(click_len*.18))*(.0015+.0040*vel)*excitation
        if event.articulation in ("hammer_on","pull_off","legato","legato_slide","tap"):
            fret_len=int(.009*self.sr); fret=self.rng.standard_normal(fret_len)
            signal[:fret_len] += fret*np.exp(-np.arange(fret_len)/(fret_len*.14))*.006
        if "slide" in event.articulation:
            scrape_n=min(int(event.duration*self.sr),total); scrape=self.rng.standard_normal(scrape_n)
            scrape=sosfilt(butter(2,[850,4200],"bandpass",fs=self.sr,output="sos"),scrape)
            motion=np.abs(np.gradient(pitch[:scrape_n]))*self.sr
            if event.articulation == "dead_slide_in":
                motion = np.maximum(motion, .42)
            signal[:scrape_n] += scrape*np.clip(motion/18,0,.035)
        if event.articulation == "palm_mute":
            # The heel of the picking hand adds a strong termination loss at
            # the bridge while leaving the initial string/pickup transient.
            signal *= np.exp(-math.log(1000)*t/.32)
        elif event.articulation in ("dead_note", "dead_slide_in"):
            # A released fretting finger kills pitch rapidly but leaves the
            # broadband pick/string impact used for rhythmic ghost notes.
            signal *= np.exp(-math.log(1000)*t/.075)
        release=int(event.duration*self.sr)
        if release<total:
            signal[release:]*=np.exp(-math.log(1000)*np.arange(total-release)/(.10*self.sr))
        # Calibrate by the unexcited modal bound rather than peak-normalizing
        # each event.  This preserves the real level hierarchy: pick > hammer
        # > pull-off > legato, and keeps pickup-position cancellations intact.
        modal_bound=np.sum(np.abs(amps*pickup*(nominal/nominal[0])**.58))*1.16
        return signal/max(float(modal_bound),1e-9)

    def render(self, score):
        total=int((max(e.start+e.duration for e in score)+3.2)*self.sr)
        mono=np.zeros(total)
        ordered=sorted(score,key=lambda event:event.start)
        for event_index,e in enumerate(ordered):
            note=self.gesture(e); offset=int(e.start*self.sr); span=min(len(note),total-offset)
            # One guitar string cannot sustain two independently fretted notes.
            # Damp the prior tail when the next event arrives on that string;
            # this prevents high-gain intermodulation during fast monophonic runs.
            if e.string_index is not None:
                following=next((other for other in ordered[event_index+1:]
                                if other.string_index==e.string_index and other.start>e.start),None)
                if following is not None:
                    choke=int((following.start-e.start)*self.sr)
                    if 0 <= choke < len(note):
                        fade=np.exp(-math.log(1000)*np.arange(len(note)-choke)/(.006*self.sr))
                        note[choke:]*=fade
            mono[offset:offset+span]+=(.17+.83*e.velocity/127)**1.3*note[:span]
        return amp_effects.process_lead_chain(mono,self.sr,ROOT/"model"/"amp_cabinet_modes.json",
                                              self.tempo_bpm,self.delay_level)


def write_wav(path, audio, sr):
    path.parent.mkdir(parents=True,exist_ok=True)
    pcm=(np.clip(audio,-1,1)*((1<<23)-1)).astype(np.int32)
    raw=bytearray()
    for frame in pcm:
        for value in frame: raw += struct.pack("<i",int(value))[:3]
    with wave.open(str(path),"wb") as wav:
        wav.setnchannels(2); wav.setsampwidth(3); wav.setframerate(sr); wav.writeframes(raw)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=ROOT/"out"/"rich_a_minor_lead_solo.wav")
    parser.add_argument("--rate",type=int,default=48_000)
    args=parser.parse_args(); score=solo_score(); guitar=LeadGuitar(args.rate)
    audio=guitar.render(score); write_wav(args.out,audio,args.rate)
    (ROOT/"out"/"rich_solo_score.json").write_text(json.dumps([e.__dict__ for e in score],indent=2),encoding="utf-8")
    print(f"rendered {len(score)} gestures, {len(audio)/args.rate:.2f} s -> {args.out}")


if __name__ == "__main__": main()
