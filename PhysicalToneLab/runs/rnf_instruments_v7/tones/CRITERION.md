# Tones: rule written before training

Data: EGFxSet (real Stratocaster, bridge pickup, 138 notes), clean vs three real pedals. Test = frets 3, 8, 13, 18.
Metric: complex error-to-signal ratio on harmonic envelopes of test frames (same as rnf_resonant_amp_v6).
Per pedal: resonant flow (2 seeds, 1500 steps), shuffled control (1 seed), Wiener-Hammerstein (2 seeds, 3000 steps),
black-box MLP 18.8k (2 seeds, 6000 steps). Fewer seeds than v6 because of compute time.
PASS for a pedal = every resonant seed has lower test error than the shuffled control and than every black-box seed.
Wiener-Hammerstein is reported whichever way it falls.
Multi-tone model (cond.py): one network for all three pedals with a learned tone vector; reported per pedal next to
the separate models; no pass rule, descriptive.
An earlier extraction used a wrong clean-to-pedal delay (hundreds of samples, from a periodic cross-correlation);
it was discarded before any training. Delay is 0: attacks show the files are aligned to within ~3 samples.
