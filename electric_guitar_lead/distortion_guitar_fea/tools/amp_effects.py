"""White-box pedal, tube-amplifier, FEA speaker/cabinet, and room chain."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter, resample_poly, sosfilt


def _lp(x, hz, sr, order=2):
    return sosfilt(butter(order,hz,"lowpass",fs=sr,output="sos"),x)


def _hp(x, hz, sr, order=2):
    return sosfilt(butter(order,hz,"highpass",fs=sr,output="sos"),x)


def _resonator(x, hz, q, sr):
    r=math.exp(-math.pi*hz/(q*sr)); theta=2*math.pi*hz/sr
    return lfilter([(1-r)*2*math.sin(theta)],[1,-2*r*math.cos(theta),r*r],x)


def _compress_band(x, threshold, ratio, sr):
    # Feed-forward RMS-like envelope; soft knee retains pick articulation.
    env=np.sqrt(lfilter([1-math.exp(-1/(.0035*sr))],[1,-math.exp(-1/(.0035*sr))],x*x)+1e-10)
    over=np.maximum(env/threshold,1.0)
    gain=over**(-(1-1/ratio))
    gain=lfilter([1-math.exp(-1/(.115*sr))],[1,-math.exp(-1/(.115*sr))],gain)
    return x*(.10+.90*gain)


def compressor(x, sr):
    low=_lp(x,330,sr); high=_hp(x,2100,sr); mid=x-low-high
    return 1.12*(_compress_band(low,.24,2.0,sr)+_compress_band(mid,.18,2.7,sr)+_compress_band(high,.13,3.1,sr))


def dual_stage_overdrive(x, sr):
    # Tight bass before clipping, then two differently biased soft stages.
    stage=_hp(x,105,sr)
    stage=np.tanh(1.16*stage+.045*stage*stage)
    stage=.84*stage+.16*_lp(stage,1350,sr)
    stage=np.tanh(1.42*stage-.035*stage*stage+.008*stage**3)
    return _lp(stage,10_000,sr)


def high_gain_distortion(x, sr):
    """Tight cascaded high-gain preamp for the GP Distortion Guitar preset.

    The GP archive names an RSE factory sound but does not embed that sound's
    proprietary amp/pedal definition. These explicit stages provide the
    missing high-gain mechanism while keeping the implementation white-box.
    """
    # Fast downward expansion suppresses physical-string noise between notes.
    envelope=np.sqrt(lfilter([1-math.exp(-1/(.0025*sr))],
                             [1,-math.exp(-1/(.0025*sr))],x*x)+1e-12)
    gate=np.clip((envelope-.0035)/.012,0,1)**.38
    stage=x*(.12+.88*gate)
    # Tighten before distortion so low strings do not turn into intermodulation mud.
    stage=_hp(stage,92,sr,2)
    stage += .22*_resonator(stage,720,1.0,sr)
    # Four differently biased triode/pedal-like soft-clipping stages. Filtering
    # between stages is essential: gain alone would produce broadband fizz.
    stage=np.tanh(4.8*stage+.14*stage*stage)
    stage=_hp(_lp(stage,7200,sr),105,sr)
    stage=np.tanh(3.9*stage-.10*stage*stage+.018*stage**3)
    stage=.82*stage+.18*_resonator(stage,1280,1.15,sr)
    stage=_lp(stage,6500,sr)
    stage=np.tanh(3.25*stage+.055*stage*stage)
    stage=_hp(stage,78,sr)
    stage=np.tanh(2.65*stage-.035*stage*stage)
    # A small dry component restores pick identity under the saturated sustain.
    return _lp(.94*stage+.06*x,9100,sr)


def tone_stack(x, sr):
    low=_lp(x,280,sr); high=_hp(x,2350,sr); mid=x-low-high
    presence=_hp(x,3900,sr,1)
    return .90*low+1.13*mid+.60*high+.035*presence


def power_amp(x, sr):
    # Slow supply droop (sag), transformer bandwidth, and push-pull rounding.
    envelope=lfilter([1-math.exp(-1/(.115*sr))],[1,-math.exp(-1/(.115*sr))],np.abs(x))
    sag=1/(1+.48*envelope)
    driven=1.22*x*sag
    even=.032*driven*driven*np.sign(driven+.03)
    power=np.tanh(driven+even)/math.tanh(1.22)
    return _lp(_hp(power,68,sr),7800,sr)


def cabinet_mics(x, sr, model_path: Path):
    modes=json.loads(model_path.read_text(encoding="utf-8"))["modes"]
    base=_lp(_hp(x,74,sr),5250,sr,3)
    body=_lp(base,900,sr,2)-_lp(base,240,sr,2)
    upper=_hp(base,1050,sr,1)
    base=base+.34*body-.20*upper
    # Restore the small, measured 2.8--5.2 kHz pick/bite band after the broad
    # upper-mid reduction; this is only ~1--2% of reference energy.
    bite=_hp(base,2800,sr,2)
    base += 2.30*bite
    center=.68*base; edge=.72*base
    for mode in modes:
        f=float(mode["hz"]); q=float(mode["q"]); drive=float(mode["drive_coupling"])
        response=_resonator(x,f,q,sr)
        weight=.075 if mode["kind"]=="cone" else .040
        center += weight*drive*float(mode["mic_center"])*response
        edge += weight*drive*float(mode["mic_edge"])*response
    # Level-dependent cone breakup: guitar speakers are musical nonlinear radiators.
    center += .040*_lp(np.tanh(1.55*base)-1.55*base,5000,sr)
    edge += .030*_lp(np.tanh(1.38*base)-1.38*base,4500,sr)
    center += .08*_resonator(base,112,1.0,sr)+.035*_resonator(base,1780,1.4,sr)
    edge += .11*_resonator(base,138,1.0,sr)+.10*_resonator(base,760,1.2,sr)
    return center,edge


def delay_and_room(left, right, sr, tempo_bpm=100.0, delay_level=1.0):
    dry=np.column_stack([left,right]); wet=np.zeros_like(dry)
    # Warm score-synchronised dotted-eighth repeats, ping-pong stereo.
    n=len(left); pos=np.arange(n,dtype=float); time=pos/sr
    dotted_eighth=.75*60/max(float(tempo_bpm),1.0)
    for repeat,gain in ((1,.18),(2,.095),(3,.045)):
        delay=(dotted_eighth*repeat+.0013*np.sin(2*np.pi*(.31+.04*repeat)*time))*sr
        source=pos-delay
        l=np.interp(source,pos,right,left=0,right=0); r=np.interp(source,pos,left,left=0,right=0)
        wet[:,0]+=delay_level*gain*_lp(l,4200,sr,1); wet[:,1]+=delay_level*gain*_lp(r,4000,sr,1)
    for delay_s,gain in ((.019,.075),(.037,-.052),(.061,.038),(.097,.025)):
        d=int(delay_s*sr)
        wet[d:,0]+=gain*right[:-d]; wet[d:,1]+=gain*left[:-d]
    return .94*dry+wet


def process_lead_chain(mono, sr, cabinet_model: Path, tempo_bpm=100.0, delay_level=1.0):
    x=_hp(mono,58,sr)
    # Loaded middle single coil: cable capacitance produces a broad presence
    # resonance, while open tone controls leave it audible without ice-pick HF.
    x=_lp(x+.13*_resonator(x,3650,1.05,sr),9000,sr)
    x=compressor(x,sr)
    # Nonlinear sections run at 2x to keep newly generated harmonics from
    # folding back as brittle digital fizz.
    x=resample_poly(x,4,1); nonlinear_sr=4*sr
    x=high_gain_distortion(x,nonlinear_sr)
    x=tone_stack(x,nonlinear_sr)
    x=power_amp(x,nonlinear_sr)
    x=resample_poly(x,1,4)[:len(mono)]
    center,edge=cabinet_mics(x,sr,cabinet_model)
    # Both channels retain a centered close mic; the delayed edge mic supplies
    # width without pulling the lead away from the middle of the mix.
    delayed=np.zeros_like(edge); delayed[43:]=edge[:-43]
    left=.57*center+.43*edge
    right=.57*center+.43*delayed
    stereo=delay_and_room(left,right,sr,tempo_bpm,delay_level)
    stereo-=np.mean(stereo,axis=0,keepdims=True)
    stereo*=.93/max(np.max(np.abs(stereo)),1e-9)
    return stereo
