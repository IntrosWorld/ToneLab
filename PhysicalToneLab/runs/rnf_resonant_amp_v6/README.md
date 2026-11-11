# v6: Resonant flow — the triad cell extended and used as the distortion model itself

**Result: PASS by the rule written in CRITERION.md before training.** Test set scored once.

Task: clean guitar harmonics in, the same notes through a captured real amp out
(NAM capture "Bug1990 Lead" + cabinet, the teacher of the preferred tone). 6,885 training frames,
376 validation, 1,647 test frames from performances never used in training (incl. two Redbone arrangements).
Error = complex error-to-signal ratio on the harmonics (0 = perfect, 0.81 = doing nothing but one gain).

| Model | Parameters | Test error, 3 seeds | Mean |
| --- | ---: | --- | ---: |
| **Resonant flow (proposed)** | 3,696 | 0.091, 0.089, 0.090 | **0.090** |
| Classical EQ - waveshaper - EQ (reference) | 695 | 0.207, 0.193, 0.194 | 0.198 |
| Same network, resonance rule broken (control) | 3,696 | 0.346, 0.309, 0.362 | 0.339 |
| Black-box MLP, large | 18,768 | 0.391, 0.394, 0.393 | 0.393 |
| Black-box MLP + phase symmetry (reference) | 4,568 | 0.411, 0.413, 0.415 | 0.413 |
| Black-box MLP, small | 4,568 | 0.467, 0.474, 0.476 | 0.472 |

## What the architecture is
State: one complex envelope per harmonic. A weight-tied flow of 8 steps. The only nonlinear terms allowed are
frequency-conserving ones: all three-wave products (i +- j = k), all four-wave products (i +- j +- l = k) and
level-dependent damping. Couplings are low-rank functions of absolute frequency from one small shared network,
so the same weights serve every pitch. The full resonant sums are computed exactly by filter - multiply one
period - transform back. It runs at frame rate (43 frames/s), not audio rate.

## Honest limits (say these when presenting)
- Guitar input is synthesised; the amp is a neural capture of a real amp, not a microphone recording.
- Only single steady notes are modelled and scored (no chords, no pick attack), harmonics below 8 kHz.
- The first design tonight (three-wave triads only) did NOT separate from its control on validation
  (0.273 vs 0.286, pilot); four-wave terms were added before any test frame was scored. See CRITERION.md.
- Earlier negative results (v1-v5, cell on unpaired real recordings) still stand.
- Novelty: frequency-conserving (resonant) interactions and filter-multiply-filter Volterra branches are known
  ideas; the claim that can be defended is this specific weight-tied resonant flow on modal envelopes with
  shared frequency-conditioned couplings, plus the measured ablation. No literature search was redone tonight.
- Listening files in audition/ were measured, not listened to. C is rebuilt from predicted harmonics only.
  Measured on those excerpts: waveform error of C against B 0.118, correlation 0.94.

Files: models.py (SpectralFlow), train.py, extract.py, report.py, results.json, results_chart.png, logs/, checkpoints/.
All 2,927 pre-existing files verified unchanged (integrity.py verify).

## Full-audio rendering (added after listening feedback)
The harmonics-only song render (audition/song_3) was heard as "saxophone / wind, no distortion": its held-note
harmonics matched the amp, but attacks, noise and note overlaps were missing. render_audio.py runs the SAME
trained weights on all frequency bins of short frames of the full signal (no retraining, band 70 Hz - 8 kHz,
state energy limited to the largest value seen in training). Waveform error against the real-amp target on the
test recordings, chords and attacks included: see audio_rate_test.json (0.154 / 0.157 / 0.100; doing nothing = 0.89-0.96).
Song: audition/song_4_resonant_flow_full_audio.wav (waveform error 0.11 against the earlier amp's render).
Nothing above 8 kHz is produced. Not listened to by the assistant.
