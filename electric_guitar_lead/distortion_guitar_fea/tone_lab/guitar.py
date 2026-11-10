"""Fretted string/pickup geometry, expressive clean amplification and chorus.

The existing score's pitch trajectories and body/neck FEA damping are retained.
The new pickup voltage model and effect chain are independent of the high-gain path.
"""
from __future__ import annotations

import json
import math

import numpy as np

from tone_lab import dsp
from tone_lab.score import ROOT, physics

PRESETS = {
    "clean_neck": {"label": "Warm clean", "pickup": [0.08, 0.12, 0.80],
                   "pick_m": .102, "resonance": 3100, "presence": -.5,
                   "cutoff": 6300, "drive": .60, "room": .13, "chorus": None},
    "clean_glass": {"label": "Glass clean", "pickup": [.48, .52, 0.],
                    "pick_m": .087, "resonance": 3900, "presence": 1.4,
                    "cutoff": 7400, "drive": .52, "room": .11, "chorus": None},
    "chorus_classic": {"label": "Classic chorus", "pickup": [0.08, 0.12, 0.80],
                       "pick_m": .102, "resonance": 3100, "presence": -.5,
                       "cutoff": 6300, "drive": .60, "room": .13,
                       "chorus": {"mix": .36, "depth": .0018, "rate": .37, "voices": 2}},
    "chorus_wide": {"label": "Wide chorus", "pickup": [.48, .52, 0.],
                    "pick_m": .087, "resonance": 3900, "presence": 1.4,
                    "cutoff": 7400, "drive": .52, "room": .20,
                    "chorus": {"mix": .40, "depth": .0026, "rate": .28, "voices": 4}},
}

LEGATO = {"hammer_on", "pull_off", "legato", "legato_slide", "tap"}


class CleanGuitar:
    def __init__(self, tuning=(40, 45, 50, 55, 59, 64), seed=1954):
        self.tuning = tuple(tuning)
        self.seed = seed
        self.modes = json.loads((ROOT/"model/structure_modes.json").read_text())["modes"]

    def note(self, event, preset, rng, phase_origin=0., held_seconds=None):
        index = event.string_index
        if index is None:
            index = int(np.flatnonzero(np.asarray(self.tuning) <= event.midi)[-1])
        target = event.midi if event.target is None else event.target
        duration = max(.002, event.duration if held_seconds is None else held_seconds)
        t = np.arange(round((duration+.24)*dsp.SR))/dsp.SR
        # The fretting location is set by the base fret, not by added bend tension.
        fretted_midi = min(event.midi, target) if event.articulation == "prebend_release" else event.midi
        fret = max(0., fretted_midi-self.tuning[index])
        length = .648/2**(fret/12)
        spec = physics.STRINGS[index]
        tension = spec.linear_density_kg_m*(2*.648*float(physics.fundamental(self.tuning[index])))**2
        inertia = np.pi*spec.core_diameter_m**4/64
        stiffness = np.pi**2*2.e11*inertia/(tension*length**2)
        pitch = physics.pitch_curve(event.midi, target, event.articulation,
                                   np.minimum(t, event.duration), event.duration, event.vibrato_cents)
        velocity = np.clip(event.velocity/127, 0., 1.)
        pitch += (.028*velocity**2)*np.exp(-t/.035)
        f0 = physics.fundamental(pitch)
        phase = phase_origin + np.cumsum(2*np.pi*f0/dsp.SR)
        modes = np.arange(1, min(110, int(17000/max(f0)))+1, dtype=float)
        # Divide by sqrt(1+B): the FIRST mode is exactly the requested pitch.
        ratios = modes*np.sqrt((1+stiffness*modes*modes)/(1+stiffness))
        keep = ratios*np.max(f0) < 18000
        modes, ratios = modes[keep], ratios[keep]
        # Pick and pickups stay fixed on the body as the speaking length shortens.
        pick = np.clip((preset["pick_m"]+rng.normal(0, .0012))/length, .06, .84)
        pickup = sum(weight*np.sin(np.pi*modes*(position/length))*np.sinc(modes*.008/length)
                     for weight, position in zip(preset["pickup"], (.041, .094, .149)))
        contact = (.0013 + .0009*(1-velocity))/length
        amplitudes = np.sin(np.pi*modes*pick)*np.sinc(modes*contact)*pickup/modes
        amplitudes *= np.exp(-ratios*float(np.mean(f0))/(4600+6800*velocity))
        # Fixed force-to-voltage calibration; never normalize individual notes.
        amplitudes *= .92 * velocity**1.45
        legato_gain = {"hammer_on": .70, "pull_off": .73, "legato": .64,
                       "legato_slide": .88, "tap": .69}.get(event.articulation, 1.)
        amplitudes *= legato_gain
        if event.articulation == "pinch_harmonic":
            amplitudes *= .04+1.8*np.exp(-.5*((modes-4.5)/1.1)**2)
        nominal = ratios*float(physics.fundamental(fretted_midi))
        mobility = physics.structure_mobility(nominal, self.modes)
        t60 = 13.5*(110/nominal)**.33/(1+.007*fret)/(1+2.2*mobility)
        t60 /= 1+(nominal/8500)**1.6
        signal = np.zeros(len(t))
        for a, ratio, decay in zip(amplitudes, ratios, t60):
            if abs(a) < 1e-5:
                continue
            env = np.exp(-np.log(1000)*t/max(.10, decay))
            # Short-lived transverse energy creates a bright pick, then warm sustain.
            attack_colour = .79 + .21*np.exp(-t/.037)
            signal += a*env*attack_colour*np.sin(ratio*phase)
            signal += .13*a*env**1.5*np.sin(ratio*phase*2**(.75/1200)+.23)
        signal *= 1-np.exp(-t/(.0015 if event.articulation in LEGATO else .00065))
        contact_noise = dsp.filt(rng.standard_normal(len(t)), [900, 5800], "bandpass", 1)
        signal += .007*velocity*legato_gain*contact_noise*np.exp(-t/.0035)
        if event.articulation in LEGATO:
            knock = dsp.filt(rng.standard_normal(len(t)), [300, 1700], "bandpass", 1)
            signal += .009*velocity*knock*np.exp(-t/.007)
        if "slide" in event.articulation:
            speed = np.abs(np.gradient(pitch))*dsp.SR
            scrape = dsp.filt(rng.standard_normal(len(t)), [850, 4100], "bandpass")
            signal += .009*velocity*scrape*np.minimum(speed/30, 1)*np.exp(-t/2)
        if event.articulation in {"dead_note", "dead_slide_in"}:
            signal *= np.exp(-t/.009)*.15
            scratch = dsp.filt(rng.standard_normal(len(t)), [550, 3800], "bandpass")
            signal += .043*velocity*scratch*np.exp(-t/.021)*(1-np.exp(-t/.001))
        elif event.articulation == "palm_mute":
            signal *= np.exp(-t/.062)
        release = np.maximum(t-duration, 0.)
        signal *= np.exp(-np.log(1000)*release/.09)
        return signal, phase

    def render_di(self, events, preset="clean_neck", seconds=None):
        config = PRESETS[preset]
        if seconds is None:
            seconds = max((e.start+e.duration for e in events), default=0)+3.2
        out = np.zeros(round(seconds*dsp.SR))
        rng = np.random.default_rng(self.seed)
        ordered = sorted(events, key=lambda e: (e.start, e.string_index or 0))
        phases = {}
        for number, event in enumerate(ordered):
            following = next((n for n in ordered[number+1:]
                              if n.string_index == event.string_index and n.start > event.start), None)
            held = event.duration
            if following and following.start-event.start < event.duration/.94:
                held = following.start-event.start
            initial = 0.
            previous = phases.get(event.string_index)
            if event.articulation in LEGATO and previous is not None:
                start, phase = previous
                pos = round((event.start-start)*dsp.SR)
                if 0 <= pos < len(phase):
                    initial = float(phase[pos] % (2*np.pi))
            note, phase = self.note(event, config, rng, initial, held)
            phases[event.string_index] = (event.start, phase)
            if following:
                cut = max(0, round((following.start-event.start)*dsp.SR))
                if cut < len(note):
                    note[cut:] *= np.exp(-np.arange(len(note)-cut)/(dsp.SR*.0015))
            start = round(event.start*dsp.SR)
            span = min(len(note), len(out)-start)
            if start >= 0 and span > 0:
                out[start:start+span] += note[:span]
        return dsp.filt(out, 38, "highpass")

    def process(self, di, preset):
        config = PRESETS[preset]
        x = dsp.eq(di, config["resonance"], 2.1, .85)
        x = dsp.compressor(x, -20., 1.65, attack=.016, release=.15)
        x = dsp.saturate(x, config["drive"])
        x = dsp.filt(x, 65, "highpass")
        x = dsp.eq(x, 190, 1.0, .8)
        x = dsp.eq(x, 680, -1.3, .7)
        x = dsp.eq(x, 2300, config["presence"], .85)
        x = dsp.filt(x, config["cutoff"], order=3)
        x = dsp.chorus(x, **config["chorus"]) if config["chorus"] else dsp.stereo(x)
        return dsp.room(x, wet=config["room"], t60=.92)
