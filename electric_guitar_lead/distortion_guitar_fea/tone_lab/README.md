# Tone lab — GP1 clean and chorus, finger bass, acoustic kit

Independent Python sound-design work inside the distortion project. The original
distortion engine, bass/drum engines, reference GP files and renders are unchanged.
No recorded instrument samples or pretrained models are used.

## Listen

Open `listen.html` in a browser, or play the 48 kHz / 24-bit stereo WAVs in `out/`.
Use the same listening volume when comparing files.

| File in `out/` | What it contains |
|---|---|
| `gp1_clean_neck.wav` | Warm clean, mostly neck pickup, short studio room |
| `gp1_clean_glass.wav` | Brighter middle/bridge combination |
| `gp1_chorus_classic.wav` | Warm clean with two modulated chorus voices |
| `gp1_chorus_wide.wav` | Glass clean with four chorus voices and more room |
| `gp1_clean_neck_DI.wav` | Pickup signal before the amplifier and effects |
| `gp1_legacy_string_clean_reference.wav` | Original distortion string generator through the new clean amp, for diagnosis |
| `gp1_bass_entrance.wav` | GP1's first four bass notes, starting just after the lead phrase |
| `gp2_bass_finger.wav` | Improved bass, exact GP2 opening notes |
| `gp2_drums_acoustic.wav` | Improved kit, exact GP2 opening hits |
| `gp2_clean_chorus_band.wav` | New chorus lead, clean rhythm, bass and drums together |
| `gp2_bass_before_then_after.wav` | Original bass model followed by new bass; 650 ms gap |
| `gp2_drums_before_then_after.wav` | Original drum model followed by new kit; 650 ms gap |
| `bass/dynamics_and_strings.wav` | Five bass pitches, each soft / medium / hard |
| `drums/kit_and_velocity_sweep.wav` | Kick, snare, hats, four toms, crash, ride; soft / medium / hard, then hat choking |

The lead comparison uses exactly the same 47 sounding events as the original GP1
diagnostic: 61 raw notes merged across ties, with bends, slides, muted scrapes,
pull-offs and vibrato. The phrase ends at 12.2606 s; the file retains a 3.2 s effects
tail. GP1 contains no drums. Its first bass note starts at 12.3727 s, after the lead
phrase. The separate GP1 bass entrance is moved to 0.2 s with the source offset in
the manifest. The rhythm-section and band comparisons use the first 12.3 s of GP2
plus a 2.4 s tail; they are not invented accompaniment for GP1.

## Changes that affect the sound

- **Guitar:** signed triangular-pluck excitation and magnetic velocity observation;
  pickup/pick positions measured in metres relative to the fretted string length;
  fundamental-corrected stiff-string modes; frequency-dependent decay informed by
  the existing body/neck FEA; a faster pick attack; a second string polarization;
  subtle contact sounds; string-specific choking; phase-referenced legato attacks.
- **Clean signal path:** gentle linked compression, oversampled mild preamp
  rounding, pickup-load presence and open-combo EQ. No high-gain stage is used.
  Classic chorus and warm clean share an identical DI. Wide chorus and glass clean
  share an identical DI. The chorus uses interpolated modulated delays; the studio
  room is a generated diffuse response, not a recorded impulse response.
- **Bass:** continuum stiff-string ratios replace coarse mesh ratios (the old
  36-element mesh alone adds about 105 cents at partial 14). Body/neck FEA still
  controls damping. Finger position varies slightly between strokes, hard strokes
  add contact/buzz, and notes retain velocity-dependent attack and brightness.
  Same-string overlap is damped. The DI/cabinet blend is centered with no delayed
  right-channel copy.
- **Drums:** membrane modes and damped shell/air impact for kick and toms; delayed,
  uneven wire response for snare; 256 hat, 512 crash and 384 ride modes plus
  collision noise. Cymbal mode frequencies remain fixed across hits while coupling
  and contact vary. Only a closed/pedal hat chokes an open hat. No individual hit
  is peak-normalized, so ghost notes remain quiet. Original GP timing is retained.

These are reduced physical and statistical synthesis models. In particular, the
extra cymbal modes are sound-design approximations, not a new cymbal FEA solve.
The amp and room are designed effects. They have not been fitted to recorded
reference instruments, and numerical checks do not establish perceptual realism.
The WAVs are the listening review; realism should be judged on headphones/speakers.

## Rebuild

From `F:\College\projects\VLSI\electric_guitar_lead\distortion_guitar_fea`:

```powershell
python -m tone_lab.render
python -m pytest tests tone_lab/tests -q
python -m tone_lab.audit
```

NumPy and SciPy are the render dependencies; pytest is only needed for tests.
Run render commands sequentially because each updates the shared manifest.

```powershell
python -m tone_lab.render --only guitar
python -m tone_lab.render --only band
python -m tone_lab.render --only diagnostics
```

Preset values live in `tone_lab/guitar.py`. Bass and kit implementations are in
`tone_lab/bass/model.py` and `tone_lab/drums/model.py`. The existing GPIF reader and
FEA tables are imported read-only from this project, so keep `tone_lab` here.

`out/manifest.json` records source SHA-256 hashes, note/hit events, source offsets,
presets, levels, spectral bands, stereo correlation and oversampled peak estimates.
Solo comparison files target the same gated active RMS of -20 dBFS. This is not
LUFS normalization. The band targets -19 dBFS; the wide-range kit sweep uses
-23 dBFS. A linked safety stage and 4x peak check protect the -1.5 dBTP ceiling.
PCM export adds deterministic TPDF dither. It refuses nonfinite or clipped input.

The tests check actual rendered pitch, playing-effect retention, velocity range,
repeat-hit variation, note-off decay, hat choking, centered bass, chorus mono
compatibility, determinism, PCM export and peak headroom. Existing project tests
also run to detect regressions. This work remains floating point; RTL is a later
step after the timbres are accepted.

Background for the DSP choices: Julius O. Smith's
[Physical Audio Signal Processing](https://dsprelated.com/freebooks/pasp/) and
Skare & Abel's [modal crash-cymbal synthesis paper](https://www.dafx.de/paper-archive/2019/DAFx2019_paper_48.pdf).
The code is a new reduced implementation, not a reproduction of the paper's GPU model.
