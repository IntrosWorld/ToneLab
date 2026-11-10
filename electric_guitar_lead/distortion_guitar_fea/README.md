# Distortion Guitar Lead FEA

This is the self-contained physical-model distortion lead used to render the
`Lead Guitar` track from `tab1.gp`. It is intentionally separate from the
parent `electric_guitar_lead` RTL/Karplus–Strong instrument.

Signal path:

1. Stiff steel-string partial model with continuous bends, slides and vibrato.
2. Magnetic single-coil pickup aperture and position response.
3. Alder-equivalent body plate and maple-neck FEA modal mobility.
4. Compressor, nonlinear overdrive, tone stack and power-amplifier stages.
5. FEA-derived speaker-cone and open-back cabinet modes.
6. Stereo close microphones, delay and algorithmic room reflections.

The bundled render contains only GP track 0, `Lead Guitar`, whose score sound
is `Distortion Guitar`. The acoustic rhythm-guitar and bass tracks are excluded.
The GP file references Guitar Pro's proprietary RSE factory preset but does not
embed its amp or pedal definition. `model/distortion_preset.json` records the
available RSE metadata and the explicit four-stage high-gain replacement used
by this project.

## Render

From this folder:

```powershell
python tools\render_gp_score.py ref\tab1.gp --track 0 --seconds 20 `
  --out out\tab1_heavy_distortion_guitar_fea.wav
```

Dependencies: Python 3, NumPy and SciPy.
# Complete tab2 arrangement

`python tools/render_tab2_full_band.py` calls the independent lead-guitar,
rhythm-guitar, bass, and drum renderers, then creates
`out/tab2_full_band_fea.wav`. The bass and drum-set implementations and stems
remain isolated in `bass_fea/` and `drum_set_fea/`.
