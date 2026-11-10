"""Fingerstyle bass: stiff-string continuum modes plus existing body/neck FEA.

Continuum ratios avoid the coarse finite-element mesh's upper-partial dispersion.
Physical inharmonicity remains; the first partial is calibrated to the score pitch.
"""
from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np

from tone_lab import dsp
from tone_lab.score import ROOT, physics, parse_gpif, merge_ties, DYNAMICS


@dataclass(frozen=True)
class BassNote:
    start: float
    duration: float
    midi: float
    velocity: int = 100
    string_index: int = 0
    articulation: str = "finger"


def load_bass(source, seconds=12.3, track=2):
    import zipfile
    import xml.etree.ElementTree as ET
    with zipfile.ZipFile(source) as archive:
        root = ET.fromstring(archive.read("Content/score.gpif"))
    tuning = [int(v) for v in root.findall("./Tracks/Track")[track].findtext(
        ".//Property[@name='Tuning']/Pitches").split()]
    name, tempo, raw = parse_gpif(source, track, seconds)
    events = merge_ties(raw)
    notes = []
    for e in events:
        vel = DYNAMICS.get(e["dynamic"], 96) + 10*e["accent"] - 24*e["anti_accent"]
        notes.append(BassNote(e["start"], e["duration"]*.97, e["midi"],
                              int(np.clip(vel, 1, 127)), e["string"],
                              "dead" if e["muted"] else "finger"))
    return notes, {"source": str(source), "track_index": track, "track_name": name,
                   "tempo_bpm": tempo, "tuning": tuning, "raw_notes": len(raw),
                   "sounding_events": len(notes), "events": [n.__dict__ for n in notes]}


class FingerBass:
    def __init__(self, tuning=(28, 33, 38, 43), seed=2401):
        self.tuning = tuple(tuning)
        self.seed = seed
        path = ROOT/"bass_fea/bass_fea_modes.json"
        self.body = json.loads(path.read_text())["body_modes"]

    def modal_parameters(self, note):
        fret = note.midi-self.tuning[note.string_index]
        length = .864/2**(fret/12)
        density = (.0146, .0096, .0060, .0031)[note.string_index]
        diameter = (.00052, .00046, .00040, .00034)[note.string_index]
        tension = density*(2*.864*float(physics.fundamental(self.tuning[note.string_index])))**2
        b = np.pi**2*2.e11*(np.pi*diameter**4/64)/(tension*length**2)
        fundamental = float(physics.fundamental(note.midi))
        modes = np.arange(1, min(96, int(11000/fundamental))+1, dtype=float)
        frequencies = fundamental*modes*np.sqrt((1+b*modes*modes)/(1+b))
        keep = frequencies < 13000
        return modes[keep], frequencies[keep], length

    def note(self, note, rng, alternate=0):
        modes, frequencies, length = self.modal_parameters(note)
        t = np.arange(round((note.duration+.24)*dsp.SR))/dsp.SR
        velocity = np.clip(note.velocity/127, 0., 1.)
        position = (.190 + (-.005 if alternate % 2 else .005)+rng.normal(0, .0015))/length
        pickup = .70*np.sin(np.pi*modes*.146/length)*np.sinc(modes*.021/length)
        pickup += .30*np.sin(np.pi*modes*.234/length)*np.sinc(modes*.018/length)
        # Triangular displacement differentiated into magnetic-pickup voltage.
        amplitudes = np.sin(np.pi*modes*position)*pickup/modes
        amplitudes *= np.sinc(modes*(.0075-.0025*velocity)/length)
        amplitudes *= np.exp(-frequencies/(2100+3200*velocity))
        amplitudes *= .93*velocity**1.55
        signal = np.zeros(len(t))
        mobility = np.zeros(len(modes))
        for mode in self.body:
            hz = mode["hz"]
            detune = frequencies/hz-hz/frequencies
            mobility += mode["coupling"]**2/(1+(mode["q"]*detune)**2)
        t60 = 9.5*(82/frequencies)**.30/(1+1.8*mobility)
        for amp, frequency, decay in zip(amplitudes, frequencies, t60):
            # A small, rapidly relaxing tension increase follows the finger release.
            phase = 2*np.pi*frequency*(t+.00012*velocity*(1-np.exp(-t/.035)))
            envelope = np.exp(-np.log(1000)*t/decay)
            signal += amp*envelope*np.sin(phase)
            signal += .095*amp*envelope**1.7*np.sin(phase*2**(.35/1200)+.26)
        signal *= 1-np.exp(-t/.0018)
        touch = dsp.filt(rng.standard_normal(len(t)), [380, 2400], "bandpass")
        signal += .018*velocity**2*touch*np.exp(-t/.012)
        # Fret buzz is a short contact event on strong strokes, never a constant fizz.
        buzz = dsp.filt(rng.standard_normal(len(t)), [1600, 5600], "bandpass")
        signal += .021*max(0., velocity-.72)*buzz*np.exp(-t/.022)
        if note.articulation == "dead":
            signal = .16*signal*np.exp(-t/.011)+.055*velocity*touch*np.exp(-t/.028)
        release = np.maximum(t-note.duration, 0.)
        signal *= np.exp(-np.log(1000)*release/.075)
        return signal

    def render(self, notes, seconds, process=True):
        out = np.zeros(round(seconds*dsp.SR))
        rng = np.random.default_rng(self.seed)
        ordered = sorted(notes, key=lambda n: n.start)
        for i, note in enumerate(ordered):
            audio = self.note(note, rng, i)
            following = next((n for n in ordered[i+1:]
                              if n.string_index == note.string_index and n.start > note.start), None)
            if following:
                cut = round((following.start-note.start)*dsp.SR)
                if cut < len(audio):
                    audio[cut:] *= np.exp(-np.arange(len(audio)-cut)/(.0025*dsp.SR))
            start = round(note.start*dsp.SR)
            span = min(len(audio), len(out)-start)
            if start >= 0 and span > 0:
                out[start:start+span] += audio[:span]
        out = dsp.filt(out, 25, "highpass")
        if not process:
            return dsp.stereo(out)
        # Parallel DI/cabinet without delay or polarity tricks in the low end.
        cab = dsp.saturate(out, 1.15)
        cab = dsp.eq(cab, 780, 1.8, .8)
        cab = dsp.filt(cab, 4300, order=3)
        out = .68*out+.32*cab
        out = dsp.eq(out, 240, -1.8, .8)
        out = dsp.compressor(out, -22., 2.7, attack=.022, release=.13)
        return dsp.stereo(out)
