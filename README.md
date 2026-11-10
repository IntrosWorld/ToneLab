# PhysicalToneLab: resonant flow on top of physical instrument models

This file is the overview of everything in `runs/`: what was built, what the measured results are, how to run each
part, and what is still weak. Each run folder has its own README with the full log, including failed attempts.

## 1. What the project is

Physical-model synthesizers (piano, guitars, bass, drums) produce the notes. A small learned stage, trained on real
recordings, moves their tone toward the real instrument or the real amplifier / pedal.

The learned stage is the **resonant flow** (`rnf_resonant_amp_v6/models.py`, class `SpectralFlow`, 3,696 parameters):

- State: one complex envelope per harmonic (or per frequency bin for full audio).
- 8 weight-tied steps.
- The only nonlinear terms allowed are frequency-conserving ones: three-wave products (i ± j = k), four-wave products
  (i ± j ± l = k) and level-dependent damping.
- Couplings come from one small shared network that reads absolute frequency, so the same weights serve every pitch.

Two forms are used:

| Form | Used for | Can create new frequencies? |
| --- | --- | --- |
| Full resonant flow | Amplifier, distortion pedals | Yes (that is what distortion is) |
| Restricted per-note / per-hit flow: log-gain = g(f, …) − rate · age (+ a small time-frequency network for snare and crash) | Piano, steel guitar, clean electric, bass, drums | No, it only reshapes level and decay |

Novelty wording that can be defended: this specific weight-tied resonant flow on modal envelopes with shared
frequency-conditioned couplings, plus the measured ablations. Resonant interactions and Volterra-style branches are
known ideas. No literature search has been done.

## 2. Folder map

| Folder | Contents |
| --- | --- |
| `rnf_resonant_amp_v6/` | The architecture, amp training, baselines, full-audio renderer |
| `rnf_instruments_v7/` | All other instruments: piano, steel guitar, clean electric, bass, drums, snare v2, crash |
| `rnf_instruments_v7/tones/` | Real pedals: Blues Driver, Tube Screamer, RAT, multi-tone model, chorus |
| `rnf_song_ascend/` | Full song render (Guitar Pro tab "Ascend", first 120 s), stems, drum samples |
| `rnf_port_study/` | Speed and latency measurements for a possible C++ port |
| `rnf_real_v1`, `rnf_amp_port_v1`–`v4`, `rnf_string_amp_v5`, `guitar_real_v1`, `brightness_recovery_v1` | Earlier attempts, mostly negative results; kept as a record |

The original engines are never modified: `grand_piano_improved`, `acoustic_guitar_fea`,
`electric_guitar_lead/distortion_guitar_fea` (with `bass_fea`, `drum_set_fea`, `tone_lab`), and the earlier
PhysicalToneLab folders. Where an engine setting had to change, it is changed in memory at render time.

## 3. Results (all on held-out data)

### Amplifier (v6) — error-to-signal ratio on harmonics, 3 seeds, lower is better

| Model | Parameters | Mean test error |
| --- | ---: | ---: |
| Resonant flow | 3,696 | 0.090 |
| EQ – waveshaper – EQ | 695 | 0.198 |
| Same network, resonance rule broken (control) | 3,696 | 0.339 |
| Black-box MLP, large | 18,768 | 0.393 |

Full audio with the same weights (chords and attacks included): waveform error 0.154 / 0.157 / 0.100 on three test
recordings (doing nothing: 0.89–0.96).

### Real pedals (EGFxSet, real Stratocaster) — test error, pass rule written before training

| Pedal | No model | Resonant flow (2 seeds) | Control | Best baseline | Rule |
| --- | ---: | --- | ---: | ---: | --- |
| Blues Driver | 0.638 | 0.030, 0.030 | 0.196 | 0.111 | Pass |
| Tube Screamer | 0.380 | 0.052, 0.061 | 0.084 | 0.053 | Not met (tie) |
| RAT | 0.620 | 0.052, 0.059 | 0.214 | 0.103 | Pass |

One conditional network for all three pedals (5,788 parameters, 1 seed): 0.033 / 0.072 / 0.061.

### Chorus (real pedal)

A 75-parameter modulated-delay layer trained on the waveform. Held-out median error: 0.096 (fixed filter 0.473, clean
0.640). This is a time-varying linear layer, not the nonlinear resonant flow.

### Acoustic instruments, clean electric and drums (restricted flow)

| Instrument | Measure against real notes | Physical model | With learned stage |
| --- | --- | ---: | ---: |
| Piano | Band error, gain stage | 2.07 | 1.78 |
| Steel guitar | Attack error, 104 notes | 7.0 dB | 5.3 dB |
| Clean electric | Error at 1 s, 34 notes | 8.9 dB | 6.6 dB |
| Bass | Error at 1 s, 117 notes | 19.5 dB | 6.4 dB |
| Snare | Attack error | 15.4 dB | 6.1 dB |
| Kick | Body error, 60–200 ms | 16.2 dB | 10.5 dB |
| Crash | Band error, held-out 17-inch cymbal | 8.5 dB | 2.9 dB (3.3 dB with ring partials) |

### Listening verdicts by the project author

- Piano chain H: approved.
- Steel guitar, bass, kick, snare (first model): heard as better than the engine.
- Full resonant flow on piano: rejected (added distortion). This is why acoustic instruments use the restricted form.
- RAT song v1 and v2: rejected. v3: no verdict yet.
- Crash with learned stage only: rejected (noise-like). With measured ring partials: no verdict yet in the song.
- Clean electric, chorus, snare v2, piano bass fix, sustain pedal, Ascend mixes v4/v5: no verdict yet.

## 4. Setup

- Python: `C:\Users\Micro\miniconda3\python` (PyTorch CPU, numpy, scipy, soundfile).
- Run from Git Bash. Set these first in every shell:

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8; PY=/c/Users/Micro/miniconda3/python
```

- The machine has 15.8 GB RAM. Run one heavy job at a time; parallel renders ran out of memory.
- Training data is downloaded by the fetch scripts into each run's `data/` folder (not redistributed):
  Iowa MIS (piano, crash), GuitarSet, IDMT-SMT-Bass, IDMT-SMT-Drums, EGFxSet.

## 5. How to run

Commands are run inside the folder named in each heading. Most scripts take a stage name; the usual order is
`prep` (build engine/real pairs) → `fit` (train) → `check` (held-out numbers) → `render` (audio).
Trained checkpoints already exist, so `check` and `render` work without retraining.

### Check that the originals are untouched (`rnf_instruments_v7/`)

```bash
$PY integrity.py verify
```

It must print `"changed_or_deleted": []` and `"intact": true`.

### Amplifier (`rnf_resonant_amp_v6/`)

```bash
$PY extract.py
```
```bash
bash run_all.sh
```
```bash
$PY train.py test
```
```bash
$PY report.py
```
```bash
$PY render_audio.py test
```
```bash
$PY render_audio.py song
```

`extract.py` builds the paired harmonic envelopes, `run_all.sh` trains every model and seed,
`train.py test` scores the test set, `report.py` makes the chart, and `render_audio.py` runs the trained weights on
full audio. To train one model: `$PY train.py fit <arm> <seed> [steps] [lr]`, with arm = `resonant`, `shuffled`,
`wiener_hammerstein`, `blackbox_small`, `blackbox_large` or `blackbox_canonical`.

### Pedals (`rnf_instruments_v7/tones/`)

```bash
$PY ../fetch_egfx.py
```
```bash
$PY tones.py extract RAT
```
```bash
$PY tones.py fit RAT resonant 1
```
```bash
$PY tones.py test
```
```bash
$PY render_tones.py
```

Tones: `BluesDriver`, `TubeScreamer`, `RAT`. `run_all.sh` trains everything. Multi-tone model:
`$PY cond.py fit <seed>` then `$PY cond.py test`. RAT on the song with the safety steps: `$PY render_rat_song3.py`.

### Chorus (`rnf_instruments_v7/tones/`)

```bash
$PY fetch_chorus.py
```
```bash
$PY chorus.py fit
```
```bash
$PY chorus.py test
```
```bash
$PY chorus.py render
```
```bash
$PY render_chorus_song.py
```

### Piano (`rnf_instruments_v7/`)

```bash
bash get_piano.sh
```
```bash
$PY piano_flow.py prep
```
```bash
$PY piano_decay.py fit
```
```bash
$PY piano_low.py fit
```

A MIDI file through the approved chain H needs these settings, then `perform_midi2.py <midi file> <seconds> <name>`:

```bash
PIANO_LOW_CKPT=piano_low_attack PIANO_AIR_DB=12 PIANO_AIR_HZ=2000 PIANO_BASS_DB=15 PIANO_BASS_HZ=300 $PY perform_midi2.py song.mid 120 H
```

### Steel guitar, clean electric, bass (`rnf_instruments_v7/`)

Each takes `prep`, `fit`, `check`, `render`:

```bash
$PY guitar_flow.py check
```
```bash
$PY electric_clean_flow.py check
```
```bash
$PY bass_flow.py check
```

Data first: `$PY fetch_guitarset.py`, `$PY fetch_egfx.py`, `$PY fetch_bass.py`.

### Drums (`rnf_instruments_v7/`)

```bash
$PY drum_flow.py check
```
```bash
$PY drum_snare_v2.py check
```
```bash
$PY crash_flow.py check
```
```bash
$PY crash_ring.py check
```

- `drum_flow.py`: `fetch | prep | fit | check | render` (kick, snare, hi-hat).
- `drum_snare_v2.py`: `prep | fit | check` (retrained snare).
- Crash: `$PY fetch_crash.py`, then `crash_flow.py prep | fit | check | sample`, then `crash_ring.py fit | check | sample`.
  `crash_tonality.py` prints how ringing or noisy each crash is.

### Full song (`rnf_song_ascend/`)

Run the stems one at a time, then the mix:

```bash
$PY render_song.py piano
```
```bash
$PY render_song.py steel
```
```bash
$PY render_song.py guitars
```
```bash
$PY render_song.py drums
```
```bash
$PY render_song.py mix
```

- Output: `ascend_first_120s_full_mix.flac`; stems in `stems/`.
- The tab path, length (120 s) and mix levels are at the top of `render_song.py` and in `mix()`.
- `PIANO_PEDAL=0` turns the sustain pedal off; `PIANO_BASS_DETUNE=1` turns the bass-note calming off.
- `drum_samples.py` writes one file per drum piece into `drum_samples/`.

### Port study (`rnf_port_study/`)

```bash
$PY measure.py
```

Result so far: Python renders the amp 4.9× faster than real time; frames down to 12–23 ms work with the same
weights; a naive C++ loop was 8× slower than PyTorch, so a port needs a tuned FFT library. Nothing has been ported.

## 6. Where the audio is

| What | Path |
| --- | --- |
| Latest full song | `rnf_song_ascend/ascend_first_120s_full_mix_v5_piano_pedal.flac` |
| Earlier song versions | `rnf_song_ascend/ascend_first_120s_full_mix_v1…v4*.flac` |
| Drum pieces alone | `rnf_song_ascend/drum_samples/` |
| Piano, guitar, bass, drums: engine only vs learned stage | `rnf_instruments_v7/audition/` (`*_1_engine_only`, `*_2_flow`) |
| Pedals: clean / real pedal / our model | `rnf_instruments_v7/tones/audition/heldout_<tone>_{1_clean,2_real_pedal,3_resonant_flow}.flac` |
| Amp | `rnf_resonant_amp_v6/audition/` |

File-name rule: `engine_only` = physical model alone, `real_pedal` / `real` = a real recording,
`flow` / `resonant_flow` / `learned` = our model's output.

## 7. What is hand-set, not learned

- Piano chain H: bass and air levels, room, and the in-memory calming of bass-string detune.
- Piano sustain pedal: a rule (change at each left-hand chord); sympathetic resonance is not modelled.
- RAT on songs: input EQ and level matching, per-frame level retry, look-ahead limiter.
- Crash: ring partials are measured from one real cymbal (a modal fit, not a network); crash level in the mix.
- Snare v2 level, 12-string emulation, octave shift of two notes below the guitar's range.
- Mix levels, panning, rooms, final limiter.

## 8. Known limits

- The amp is trained on a neural capture of a real amp with a synthesised guitar input, not on microphone recordings.
- Tube Screamer does not beat the baselines (tie).
- RAT is unstable on pick attacks and on inputs unlike its training guitar; it needs the hand safety steps.
- Chorus is mono; flanger, phaser, delay and reverb are not done.
- Kick rings about twice as long as the real one; bass has almost nothing above 1 kHz.
- Snare hiss improved by only about 2 dB in the retrain.
- Crash: 4–8 kHz is 3–9 dB too loud and 500–1000 Hz weak on the held-out cymbal.
- Drums are trained on one real kit; crash on four cymbals.
- Chords through the amp are the least tested case.
- The tab has no bass track, so the song has no bass.
- Not real time yet: frames are 46 ms; under 10 ms would need retraining.
- Numbers come from one machine and mostly one to three seeds. Audio files were measured by the assistant, not listened to.

## 9. This standalone folder

This folder is a copy; it is not a git repository. Layout is the same as the original project, so the scripts find
each other by relative path:

    README.md                                   this file
    PhysicalToneLab/runs/rnf_resonant_amp_v6    architecture + amp
    PhysicalToneLab/runs/rnf_instruments_v7     instruments, pedals (tones/), drums, crash
    PhysicalToneLab/runs/rnf_song_ascend        full song; the tab is in tab/Ascend.gp
    PhysicalToneLab/runs/rnf_port_study         speed measurements
    grand_piano_improved, acoustic_guitar_fea, electric_guitar_lead/distortion_guitar_fea   physical-model engines (code and model files only)

In section 5, "run inside the folder named in each heading" means inside PhysicalToneLab/runs/<folder> here.

Left out to save space (about 9 GB -> 1.2 GB):
- Raw downloaded recordings (rnf_instruments_v7/data/piano, bass, egfx, guitarset, drums, crash). The prepared training
  pairs (data/*.npz) and all checkpoints ARE included, so `check` and `render` stages work. `prep` stages, and
  `tones.py extract`, `chorus.py fit/test`, need the raw data again: run the fetch scripts first.
- The amp's source recordings (NeuralInstrumentStudio_Experiments/energy_v3). `train.py test` works from the included
  envelopes.npz; `extract.py`, `render_audio.py` and the RAT song scripts need that folder next to PhysicalToneLab.
- Song stems as .npy (the .flac stems are included). `render_song.py mix` needs the four stem stages re-run first.
- Engine output folders, old WAV renders, earlier negative-result runs, and the integrity check (it only applies to
  the original project).

Checked from this folder: crash_ring, guitar_flow, electric_clean_flow, bass_flow, drum_flow, drum_snare_v2,
piano_low `check`; `tones.py test`; v6 `train.py test`; tab parsing; one piano note, one crash hit, electric and drum
engines loading. Not re-run here: full song render, any training.
