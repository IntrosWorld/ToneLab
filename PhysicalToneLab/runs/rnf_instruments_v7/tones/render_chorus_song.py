"""GP1 lead with chorus.   python render_chorus_song.py
  electric_chorus_1_engine_only.flac      physical model + the engine's own hand-made chorus ('chorus_wide'); no ML
  electric_chorus_2_flow_learned.flac     physical model + learned clean stage (electric_clean_flow) + learned chorus (chorus.py)
Both use the engine's clean amp chain and room.  The learned chorus is mono like the real pedal (same signal left and right).
The engine is used in memory only; its files are not touched.  Not listened to by the assistant.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
import numpy as np
import soundfile as sf
V7 = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(V7))
import chorus as C
import electric_clean_flow as ecf

dsp, g, plain, flowed, n = ecf.render_di(); tg = sys.modules['tone_lab.guitar']; m = C.load(); out = V7 / 'audition'; info = {}
tg.PRESETS['learned_chorus'] = dict(tg.PRESETS['clean_glass'], chorus={}); original = dsp.chorus
for name, di, preset, fn in (('electric_chorus_1_engine_only', plain, 'chorus_wide', original), ('electric_chorus_2_flow_learned', flowed, 'learned_chorus', lambda mono, **k: dsp.stereo(C.process(m, mono, sr=dsp.SR)))):
    dsp.chorus = fn; y = g.process(di, preset); dsp.chorus = original; y = y * (.89 / np.abs(y).max()); sf.write(out / f'{name}.flac', y, dsp.SR, subtype='PCM_16')
    info[name] = dict(seconds=round(len(y) / dsp.SR, 1), nan=bool(np.isnan(y).any()), rms=round(float(np.sqrt((y ** 2).mean())), 3), left_right_correlation=round(float(np.corrcoef(y[:, 0], y[:, 1])[0, 1]), 3))
print(json.dumps(info)); (out / 'electric_chorus_notes.json').write_text(json.dumps(dict(info, notes=n, note='each file peak-normalised; not listened to by the assistant'), indent=1))
