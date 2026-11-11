# v6: the triad cell as the distortion model itself (written BEFORE any training)

Change from v5: the cell is no longer placed in front of a separate amp model, and the data is paired.
Input  = complex harmonic envelopes of clean guitar (DI) frames.
Target = complex harmonic envelopes of the same frames after the captured real amp
         (NAM capture "Bug1990 Lead" + cab, the teacher of the user's preferred tone;
          pairs already in NeuralInstrumentStudio_Experiments/energy_v3/data/captured_amp).
Limits stated up front: the guitar input is synthesised, the amp is a neural capture of a real amp,
only single-pitch (monophonic, steady) frames are used, and only harmonics below 8 kHz are scored.

Split: existing manifest split by performance (train / validation / test). Model choices are made on
validation only. Test is evaluated once at the end for the final configurations.

Metric: complex error-to-signal ratio on held-out frames, sum|A_pred - A|^2 / sum|A|^2 (lower is better).

Models (3 seeds each):
 1. resonant flow: triads i+j=k and i-j=k, level-dependent damping, couplings from a shared network of frequency
 2. shuffled control: identical, but triads wired between harmonics that do NOT satisfy the rule
 3. black box A: MLP on real/imag parts, parameter count >= the resonant flow
 4. black box B: same MLP family, about 10x the parameters
Reference baselines, reported whichever way they fall (not part of the pass rule):
 5. classical Wiener-Hammerstein (EQ -> waveshaper -> EQ)
 6. black-box MLP given the phase symmetry by rotating the fundamental to zero phase

PASS = resonant flow has lower test error than models 2, 3 and 4, each by more than the seed spread
(worst resonant seed better than best seed of the other). Anything else is reported as a fail.

## Amendment (written after validation pilots, BEFORE any test frame was scored)

Pilot, seed 1, 800 steps, validation error only:
 triads-only flow 0.273 | triads-only shuffled 0.286 | Wiener-Hammerstein 0.252 | black box large 0.411 |
 black box canonical 0.489 | black box small 0.680.
The triads-only flow beat the black boxes but was not clearly separated from its shuffled control.
Diagnosis: amp clipping is mainly odd-order; a cell with only three-wave terms cannot make it directly.
Change: model 1 now contains all three-wave AND four-wave resonant products (low-rank couplings, computed
by the filter - multiply - transform method).  Model 2 is the same network with every product sent to a
wrong harmonic.  Pilot of the new pair (seed 1, 800 steps, validation): resonant 0.114, shuffled 0.276.
The triads-only pair stays in the code (arms triads_only, triads_only_shuffled) and its pilot is reported.
Pass rule, split and metric are unchanged.  Final runs: 3 seeds, 3000 steps for the flows and the
Wiener-Hammerstein; black boxes get 6000 steps and the better of two learning rates chosen on validation.
