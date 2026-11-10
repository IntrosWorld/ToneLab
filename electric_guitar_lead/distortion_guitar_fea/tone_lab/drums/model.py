"""Acoustic kit with membrane modes, snare-wire noise and dense metal modes.

The low modes use the saved FEA kit. Extra cymbal modes are statistical sound
design, not newly solved FEA. Each piece retains its mode frequencies across hits;
stick position, contact noise and coupling vary with a deterministic performance seed.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np

from tone_lab import dsp
from tone_lab.score import ROOT


@dataclass(frozen=True)
class DrumHit:
    start: float
    piece: str
    velocity: int = 100
    open_amount: float = 0.


def load_drums(source, seconds=12.3, track=3):
    sys.path.insert(0, str(ROOT/"drum_set_fea"))
    from render_gp_drums import parse_drum_track
    name, tempo, end, score, mapping = parse_drum_track(Path(source), track)
    hits = [DrumHit(h.start, h.piece, h.velocity, h.open_amount) for h in score if h.start < seconds]
    return hits, {"source": str(source), "track_index": track, "track_name": name,
                  "tempo_bpm": tempo, "score_end_seconds": end, "hits": len(hits),
                  "events": [h.__dict__ for h in hits]}


PAN = {"kick": 0., "snare": -.06, "hihat": -.38, "crash": -.43,
       "ride": .42, "tom_high": -.37, "tom_mid": -.15, "tom_low": .18, "tom_floor": .42}
TAIL = {"kick": 1., "snare": .9, "hihat": .50, "crash": 3.5, "ride": 2.2,
        "tom_high": 1.1, "tom_mid": 1.2, "tom_low": 1.4, "tom_floor": 1.6}
LEVEL = {"kick": .76, "snare": .62, "hihat": .14, "crash": .31, "ride": .20,
         "tom_high": .48, "tom_mid": .50, "tom_low": .52, "tom_floor": .56}


class AcousticKit:
    def __init__(self, seed=1973):
        self.seed = seed
        self.model = json.loads((ROOT/"drum_set_fea/drum_set_fea_modes.json").read_text())["pieces"]
        # The older GP2 modal table has only four pieces. The later GP3 solve
        # includes the toms; use it to fill missing pieces without altering GP2.
        extended = json.loads((ROOT/"drum_set_fea/tab3_drums_fea_modes.json").read_text())["pieces"]
        for name, piece in extended.items():
            self.model.setdefault(name, piece)
        self.metal = {}
        rng = np.random.default_rng(910)
        for piece, count, lo, hi in (("hihat", 256, 1300, 17700),
                                      ("crash", 512, 440, 18400), ("ride", 384, 720, 17000)):
            # Irregular dense plate-mode distribution, fixed for this instrument.
            frequencies = np.linspace(lo, hi, count)+rng.uniform(-.46, .46, count)*(hi-lo)/count
            weights = rng.uniform(.45, 1.1, count)*(frequencies/3000)**.13
            self.metal[piece] = (frequencies, weights/np.linalg.norm(weights))

    def hit(self, hit, rng, max_tail=None):
        piece = hit.piece
        if piece not in TAIL:
            raise ValueError(f"unknown drum piece: {piece}")
        tail = TAIL[piece]+(1.25*hit.open_amount if piece == "hihat" else 0.)
        if max_tail is not None:
            tail = min(tail, max(.025, max_tail))
        t = np.arange(round(tail*dsp.SR))/dsp.SR
        velocity = np.clip(hit.velocity/127, 0., 1.)
        noise = rng.standard_normal(len(t))
        signal = np.zeros(len(t))
        stick = rng.uniform(.92, 1.08)
        if piece in {"kick", "snare"} or piece.startswith("tom"):
            modes = self.model[piece]["modes"]
            base_decay = {"kick": .17, "snare": .11, "tom_high": .21,
                          "tom_mid": .24, "tom_low": .28, "tom_floor": .32}[piece]
            for i, mode in enumerate(modes[:20]):
                freq = mode["hz"]
                if freq >= 17000:
                    continue
                decay = base_decay/(1+.22*i)
                pitchfall = (.024 if piece == "snare" else .070)*velocity
                phase = 2*np.pi*freq*(t+pitchfall*.018*(1-np.exp(-t/.018)))
                # Coherent zero-displacement attack, position-dependent excitation.
                weight = mode["weight"]*(1+.08*rng.standard_normal())/(1+.08*i)
                signal += weight*np.sin(phase)*np.exp(-t/decay)
            signal *= (1-np.exp(-t/.0007))
        if piece == "kick":
            phase = 2*np.pi*(51*t + (12+9*velocity)*.022*(1-np.exp(-t/.022)))
            body = np.sin(phase)*np.exp(-t/.155)*(1-np.exp(-t/.0015))
            chest = dsp.filt(noise, [75, 220], "bandpass")*np.exp(-t/.038)
            beater = dsp.filt(noise, [1250, 5100], "bandpass")*np.exp(-t/.007)
            signal = .28*signal+.83*body+1.3*chest+.11*velocity*beater
            signal = dsp.filt(signal, 6300)
        elif piece == "snare":
            body = dsp.filt(noise, [170, 850], "bandpass")*np.exp(-t/.048)
            wire_hi = dsp.filt(noise, [2000, 12500], "bandpass")
            wire_mid = dsp.filt(rng.standard_normal(len(t)), [650, 6500], "bandpass")
            # Wires bloom after head impact and then decay unevenly with collisions.
            wire_env = (1-np.exp(-t/.0018))*(.73*np.exp(-t/.061)+.27*np.exp(-t/.14))
            chatter = 1+.12*np.sin(2*np.pi*83*t+stick)+.07*np.sin(2*np.pi*137*t)
            wires = (.58*wire_hi+.42*wire_mid)*wire_env*chatter
            crack = dsp.filt(noise, [1700, 7600], "bandpass")*np.exp(-t/.004)
            signal = .55*signal+.75*body+.74*wires+.12*velocity*crack
            signal = dsp.filt(signal, 110, "highpass")
        elif piece.startswith("tom"):
            contact = dsp.filt(noise, [700, 5200], "bandpass")*np.exp(-t/.009)
            air = dsp.filt(noise, [80, 450], "bandpass")*np.exp(-t/.065)
            signal = .78*signal+.22*contact+.40*air
            signal = dsp.filt(signal, 6500)
        else:
            frequencies, weights = self.metal[piece]
            if piece == "hihat":
                tau = .034+.23*hit.open_amount
            elif piece == "crash":
                tau = .43
            else:
                tau = .26
            for frequency, weight in zip(frequencies, weights):
                decay = tau*(5000/frequency)**.32
                phase = rng.uniform(0, 2*np.pi)
                amp = weight*rng.uniform(.75, 1.15)
                signal += amp*np.sin(2*np.pi*frequency*t+phase)*np.exp(-t/decay)
            # Dense modes carry the metal texture; broad contact supplies the wash.
            collision = dsp.filt(noise, [2200, 17500], "bandpass")
            wash = dsp.filt(rng.standard_normal(len(t)), [650, 11500], "bandpass")
            envelope = np.exp(-t/(tau*.85))
            signal = .76*signal+.46*collision*envelope+.25*wash*np.exp(-t/(tau*1.2))
            signal *= 1-np.exp(-t/(.0012 if piece == "crash" else .0003))
            signal = dsp.filt(signal, 1000 if piece == "hihat" else 450, "highpass")
        # Velocity controls both spectrum and amplitude BEFORE a fixed piece gain.
        # No per-hit normalization: ghost strokes stay ghost strokes.
        cutoff = min(18500, 4500+13000*velocity)
        signal = dsp.filt(signal, cutoff)
        signal *= LEVEL[piece]*velocity**1.65*stick
        return dsp.fade_end(signal, .014)

    def render(self, hits, seconds, ambience=True):
        result = np.zeros((round(seconds*dsp.SR), 2))
        rng = np.random.default_rng(self.seed)
        ordered = sorted(hits, key=lambda h: h.start)
        for i, hit in enumerate(ordered):
            tail = None
            # Only a CLOSED/pedal hat chokes ringing hats; an open stroke can overlap.
            if hit.piece == "hihat":
                closer = next((h for h in ordered[i+1:] if h.piece == "hihat"
                               and h.open_amount <= .10 and h.start > hit.start), None)
                if closer:
                    tail = closer.start-hit.start+.012
            audio = self.hit(hit, rng, tail)
            start = round(hit.start*dsp.SR)
            span = min(len(audio), len(result)-start)
            if start >= 0 and span > 0:
                pan = PAN[hit.piece]
                result[start:start+span, 0] += np.sqrt((1-pan)/2)*audio[:span]
                result[start:start+span, 1] += np.sqrt((1+pan)/2)*audio[:span]
        result = dsp.filt(result, 27, "highpass")
        if ambience:
            result = dsp.room(result, wet=.19, t60=.61, seed=741)
        # Preserve the transient; a gentle linked bus compressor catches only peaks.
        return dsp.compressor(result, -17., 1.55, attack=.025, release=.11)
