# v7: resonant flow for the other instruments (piano first)

Same architecture as v6 (rnf_resonant_amp_v6/models.py SpectralFlow), run on full-audio frames.
Task: note from the physical piano engine (grand_piano_improved, dry, no room) -> real recorded note.
Real data: University of Iowa Musical Instrument Samples, piano, ff and mf, 24 pitches (every 3rd semitone C1-C7),
"may be downloaded and used for any projects, without restrictions" (theremin.music.uiowa.edu/MIS.html). In data/piano.
Real and engine notes are different waveforms, so error = mean absolute difference of log band energies
(80 bands, 60 Hz - 10 kHz) over the first 3 s of each note. Held-out pitches: A1 A2 A3 A4 A5 A6 (never trained on).

| Version | Held-out error |
| --- | ---: |
| Engine, unprocessed | 2.069 |
| EQ only (control: same network, nonlinear terms off) | 1.821 |
| Resonant flow | 1.392 |

One seed each, 1,200 steps, ~10 minutes: a first result, not a finished study. No shuffled control yet.
The absolute error is still large; real recordings include room sound and noise the engine does not have.
Audio (audition/): piano_1_engine_only, piano_2_resonant_flow, piano_3_eq_only_control. Measured, not listened to.
Not done yet: acoustic guitar, bass, drums (sources identified: Iowa guitar is a nylon-string classical,
Fraunhofer IDMT-SMT-Bass 1.6 GB CC BY-NC-ND, IDMT-SMT-Drums 287 MB).

## Listening verdict and follow-up (same day)
The user heard piano_2_resonant_flow.wav as "really very bad". Measured cause: the flow puts ~10% of the energy
between the partials (engine: ~0-2%), i.e. it lowers the band-energy error by adding distortion products, which
also intermodulate on chords. The 1.392 score above is therefore NOT a real improvement.
Variant with frequency mixing switched off (level-dependent gain and tone only, cannot create new frequencies):
held-out error 1.784 (EQ only 1.821, engine 2.069), nothing added between partials on chords.
File: audition/piano_4_flow_without_frequency_mixing.wav. Honest reading: for piano, once distortion is ruled out,
the architecture adds little beyond an EQ; what is missing (room, noise, per-partial decay) is not something a
frame-by-frame tone map can supply.

## Per-note redesign (after "low end could be better")
Measured first: piano_4 had cut the bass (-58 dB share below 60 Hz vs -10 in the engine; band limit at 60 Hz plus
thin-bass Iowa recordings). Real notes differ from the engine by note: low notes have far more upper partials,
mid/high notes decay about twice as fast.
piano_clean.py / piano_decay.py: the flow now acts on each note before mixing, conditioned on pitch and velocity,
with phase-preserving terms only (no new frequencies; energy between partials on chords 0.0005 vs 0.10 for piano_2).
Below 150 Hz the training target is the engine itself.
Held-out band error (this loss, 25 Hz - 10 kHz, not comparable with the table above): engine 3.10, per-note EQ 2.41,
per-note flow with level terms 2.28, per-note flow with decay correction 2.11.
Decay correction moves decay toward the real notes but overshoots on some held-out pitches (A4: -30 dB vs real -20).
piano_9 = piano_8 plus a low shelf (+ printed gain) restoring the engine's share of energy below 250 Hz; that shelf
is a manual choice made for the user's low-end preference, not learned. None of these files was listened to.

## Low-note fix (after "low notes sound bad, not richer")
Measured on low notes in the piano_9 chain: 20-50 dB more energy below 120 Hz than the real notes (engine
fundamentals protected, then +11.7 dB shelf), and upper partials that grew over time (C2, 1-2 kHz: 63 -> 67 dB in
1.5 s; real 57 -> 52). piano_low.py: all bands trained toward the real notes, decay-rate correction limited to
>= -0.5 /s, no shelf. C1 below 60 Hz: 83 dB -> 36 dB (real 31).
demo10 renders: A_low_notes_fixed = piano_low below C3, the approved piano_decay + shelf above G3, crossfade between;
B_new_model_everywhere = piano_low for every note. Not listened to.

## Bass level (after "still unable to feel the bass")
A/B followed the real recordings and had ~23 dB less 60-120 Hz than the piano_9 chain. C and D keep the fixed low
notes (no growing partials) and add the engine's own content below 150 Hz back under notes below C3-G3:
C at the engine's level (0 dB), D at +8 dB. The bass level is a listening choice, not learned. Not listened to.
E (after "needs more bass"): engine content below 220 Hz at +14 dB under the fixed low notes; bass shares
-11.4 / -5.7 dB at 60-120 / 120-250 Hz vs -9.6 / -2.5 in the engine-only render. Listening choice, not learned.

## "Richer, premium bass, brighter overall" (perform_midi2.py)
G_rich_stereo: same trained per-note models, now rendered in stereo (each engine channel through the same gains),
with the engine's own room (mix 0.22), engine content below 300 Hz at +15 dB under low notes, and engine content
above 2 kHz at +12 dB on every note. These three are listening settings, not learned. Not listened to.

## Buzz on bass notes from 44 s (version G) -> version H
Heard as "something vibrating / leaking from the bass" from 44 s, where the C2-B2 notes (0.23 s long) begin.
Not frame ripple (measured 0.0%). Cause: the low-note model overshot the 1-4 kHz partials at note onset by 10-25 dB
versus the real notes, because the loss weighted all 3 s of each note equally. piano_low_attack = same model, loss
weighted 1/(0.08 + note age); onset 1-2 kHz now within ~3 dB of real on the checked notes. H also adds no engine
highs on low notes. Roughness share (1.5-6 kHz envelope at 70-200 Hz, 44-60 s): engine 6.3%, G 21.1%, H 14.8%.

## Acoustic guitar, steel-string (guitar_flow.py, fetch_guitarset.py, guitar_summary.py)
Engine: acoustic_guitar_fea (steel-string model), read-only. Real data: GuitarSet (Zenodo 3371780, CC BY 4.0),
microphone recordings; 48 of the 180 solo takes fetched (79 MB), 381 notes that sound alone, paired with the engine's
note on the same string and fret (median length 0.45 s). Iowa nylon set was dropped at the user's request.
Same per-note model as piano (phase-preserving, pitch/velocity conditioned, attack-weighted).
Three training attempts; the first two were WORSE than the engine on held-out attacks:
 1. raw levels: learned per-note loudness and boosted sub-fundamental rumble;
 2. level-matched + ignoring quiet bands: quiet bands unconstrained, highs +12-14 dB (10.9 dB error vs engine 7.0);
 3. band counts when either target or output is loud in it: 5.3 dB vs engine 7.0 dB mean absolute band error on the
    first 0.25 s of 104 held-out notes (guitar_summary.json); no band biased by more than 4 dB.
The engine was already close (decay over 0.3 s: engine -6.8 dB, real -6.6 dB), so the learned change is modest:
about 6 dB less 500 Hz - 2 kHz and about 4 dB more above 4 kHz.
Audio: audition/guitar_{demo,arpeggio}_{1_engine_only,2_flow}.flac, stereo with the engine's room. Not listened to.

## Electric bass, fingerstyle (bass_flow.py, fetch_bass.py)
Engine: distortion_guitar_fea/bass_fea (read-only; its existing bass_fea_modes.json is loaded, generate_model is never
called). Real data: IDMT-SMT-Bass (Zenodo 7188892, CC BY-NC-ND 4.0, research use, not redistributed): the 469 PS/FS
fingerstyle single notes (3 basses x 3 pickup settings), fetched selectively (~90 MB of 1.6 GB); 468 used.
Main gap found: engine notes decay ~5x too fast (level change 0.25 -> 1.7 s: engine -34.9 dB, real -6.4 dB).
First model (piano/guitar rule: a note may not be lengthened) could not fix it and cut the bands above 1 kHz
(attack error 17.6 dB vs engine 14.6). BassFlow may remove up to 90% of the engine's own decay rate per partial and
keeps bands down to -60 dB in the loss. Held-out pitches (117 notes), bass_summary.json:
 attack band error 11.7 dB (engine 14.6); at 1 s 6.4 dB (engine 19.5); sustain -7.3 dB (real -6.4).
Not fixed: above ~1 kHz the output is 20-35 dB below the real notes; the engine has only 14 partials per string, so
there is nothing there to bring up (no finger/string noise).
Audio: audition/bass_tab2_{1_engine_only,2_flow}.flac, both through the engine's own body/cabinet/compressor chain
(synth_note wrapped in memory only). Not listened to.

## Drums: kick, snare, hi-hat (drum_flow.py)
Engine: distortion_guitar_fea/drum_set_fea (read-only; existing drum_set_fea_modes.json loaded, generate_model never
called). Real data: IDMT-SMT-Drums V2 (Zenodo 7544164, CC BY-NC-ND 4.0, research use, not redistributed): the 42
RealDrum '#train' files (isolated kick/snare/hi-hat tracks of one real acoustic kit, 11 MB), split into 298 hits.
Per-hit model with short frames (46 ms, hop 5.8 ms): log-gain = g(f, piece) - rate(f, piece) * time since the hit.
Three trainings: (1) boosted hi-hat lows by 40-60 dB to imitate recording rumble; (2) per-piece floor + 20 dB boost
cap; (3) below the floor (snare 150 Hz, hi-hat 300 Hz) the model may only cut, tail weighted more.
Held-out files (drum_summary.json), band error vs real, engine -> model:
 kick   attack 7.8 -> 7.4 dB, body (60-200 ms) 16.2 -> 10.5 dB, level drop by 160 ms -7.4 -> -10.3 (real -19.9)
 snare  attack 15.4 -> 6.1 dB, body 18.7 -> 9.2 dB, level drop -9.2 -> -21.1 (real -28.0)
 hi-hat above 500 Hz within about +-5 dB of real (engine up to 12 dB off); level drop -13.9 -> -17.3 (real -11.9,
        so slightly too short); the summary's mean error is dominated by the protected bands below 300 Hz.
Kick is the weakest: it still rings about twice as long as the real kick. Crash has no real data and is unchanged.
One kit only, so this is that kit's sound. Audio: audition/drums_tab2_{1_engine_only,2_flow}.flac. Not listened to.

## Electric-guitar tones from real pedals (tones/)

Data: EGFxSet (Zenodo 7044411, CC BY 4.0), real Stratocaster, bridge pickup, 138 notes, clean vs Blues Driver, Tube Screamer, RAT.
Architecture: the full resonant flow of rnf_resonant_amp_v6, unchanged (3,696 parameters). Split by fret: test 3, 8, 13, 18.
Rule written before training: tones/CRITERION.md. Test error-to-signal ratio on harmonic envelopes (tones/results.json):

    pedal          no model  resonant (2 seeds)  shuffled  Wiener-Hammerstein  black box 18.8k   rule
    Blues Driver   0.638     0.030, 0.030        0.196     0.135               0.111, 0.114      PASS
    Tube Screamer  0.380     0.052, 0.061        0.084     0.053               0.055, 0.056      NOT MET (tie; one seed behind black box and W-H)
    RAT            0.620     0.052, 0.059        0.214     0.103               0.115, 0.116      PASS

One conditional network for all three pedals (tones/cond.py, 4-dim tone vector, 5,788 parameters, 1 seed, no pass rule):
Blues Driver 0.033, Tube Screamer 0.072, RAT 0.061. Blending tone vectors was not tested.

Full-audio rendering (tones/render_tones.py, v6 render_audio.process, same weights), 12 held-out notes, waveform error
vs the real pedal: Blues Driver 0.046, Tube Screamer 0.090, RAT 0.158 (clean as output: 0.71 / 0.40 / 0.72).
First render had spikes at pick attacks (peaks 5-10x the real pedal's): attacks are broadband and outside the training
frames. Declared hand steps, not learned: output clipped at the real pedal's largest training sample; RAT run 6 dB below
training input level. RAT remains the weak one on full audio: safety limit active on 18 % of held-out frames and 32 % of
song frames. Audio in tones/audition/ has NOT been listened to by the assistant.
An earlier extraction with a wrong clean-to-pedal delay was discarded before any training.
Not covered: chorus, flanger, phaser, delay, reverb (time-varying / long memory; the flow is frame-wise).

### RAT song: rejected by ear twice, re-rendered (same trained weights each time)

v1 song_RAT_resonant_flow.flac: user heard "pauses, heavily distorted". Measured: the song's guitar signal is ~8x hotter
than the training notes (median frame rms 0.106 vs 0.013) and has far weaker harmonics than the real Stratocaster
(640-2560 Hz 10-12 dB lower). Outside its training range the model turned LOUD passages DOWN (0.13 out for 0.12 in,
0.44 out for a decaying 0.02 in); safety limit active on 32 % of frames.
v2 song_RAT_v2_input_matched_level0.008.flac (tones/render_rat_song.py): static input EQ to the training guitar's average
spectrum + input level at the training median. Limit on 3 % of frames. User heard clipping / odd effect at 10-11 s and
18-20 s. Measured there: spikes up to 2x the ceiling and level dips to 0.12-0.38 (normal 0.5) on new pick attacks.
v3 song_RAT_v3_frame_level_retry.flac (tones/render_rat_song3.py): no clamping inside the flow; each frame is run at
0/-6/-12/-18/-24 dB input and the loudest one that stays inside the training range is used (2491/57/15/4/0 frames);
remaining spikes handled by a 3 ms look-ahead gain instead of hard clipping. Level while playing 0.47-0.56 (5th-95th
percentile), 1 block of 360 under 0.3. All of these input/output steps are hand-made safety rules, NOT learned.
Verdict on v3: pending.

## Clean electric guitar (electric_clean_flow.py)

Engine: tone_lab CleanGuitar (read only), bridge pickup only. Real: EGFxSet Clean, bridge pickup, 138 notes.
Same per-note model and loss as the steel-string guitar. 34 held-out notes (electric_clean_summary.json):
band error vs real at the attack 4.35 -> 3.16 dB, at 1 s 8.94 -> 6.63 dB; level change 0.25 -> 1.7 s: engine -13.3 dB,
model -17.3 dB, real -17.2 dB. Audio: audition/electric_clean_{1_engine_only,2_flow}.flac (GP1 lead, engine clean amp
chain and room after the flow). Not listened to by the assistant. Verdict: pending.

## Chorus from a real pedal (tones/chorus.py)

Data: EGFxSet Clean vs Chorus, bridge pickup. Measured before modelling: output = dry + one delayed copy of about equal
level, delay swinging ~+-1.2 ms at 2.17 Hz, at a different position in every recording (free-running pedal).
Attempt 1 (FAILED, code replaced): short-frame harmonic envelopes with one swing position shared by all recordings:
validation error 0.358 vs 0.431 for a fixed filter.
Attempt 2: modulated-delay layer trained on the waveform, 75 parameters (two frequency-reading gain networks, base delay,
rate, depth, wave shape) + one swing-position number per recording (searched, not learned, for held-out notes).
24 held-out notes (tones/chorus/results.json), waveform error-to-signal ratio, median: clean as output 0.640, fixed
filter 0.473, modulated delay 0.096 (mean 0.110, worst note 6-8 0.323); on the part after 1.5 s that the swing-position
search did not see: 0.082. Learned: base delay 5.45 ms, rate 2.162 Hz, swing 4.32-6.49 ms.
This is NOT the resonant flow's nonlinear part: it is a time-varying linear term using the same kind of coupling
network. Flanger, phaser, delay, reverb: not done.
Audio (not listened to by the assistant): tones/audition/heldout_Chorus_{1_clean,2_real_pedal,3_learned_model}.flac;
audition/electric_chorus_1_engine_only.flac (engine's hand-made chorus, no ML) and electric_chorus_2_flow_learned.flac
(learned clean stage + learned chorus, mono like the pedal). Verdict: pending.

## Ascend render (runs/rnf_song_ascend) and the fixes it led to

Full tab, first 120 s, every track through engine + learned stage (render_song.py). User verdict on v1: piano bass notes have
a vibration / sustain-type sound; snare "or something" in the drums sounds like white noise (user not sure it is the snare).
Piano bass: measured on a held 3 s G1, the ENGINE's partials 2-6 swing by 20-40 dB (two unison strings detuned ~1.9 cents in
the bass, plus the two polarisations). The learned stage (gain - rate*age) cannot remove that. Fix in memory, engine files
untouched: both detunings scaled to 0.2 below C3, back to the engine's value by G3 (render_song.calm_bass / one_calm).
Hand setting, NOT learned. After: G1 and C2 partials 1-6 decay monotonically.
Snare (drum_snare_v2.py): attempt 1 (body-band level matching) and attempt 2 (cut limit -69 dB) changed nothing (held error
2.2287 both); both of my explanations were wrong. Attempt 3 adds a network of (frequency, time since hit) to the log-gain:
held error 2.137; hiss vs real (relative to body) improved by only ~2 dB at 0-160 ms (now +7, +3, +5, +9 dB per 40 ms).
Kick and hi-hat keep drum_flow.pt. Per-piece samples in runs/rnf_song_ascend/drum_samples for the user to identify the
noisy piece (crash and tom are engine only). v2 mix: ascend_first_120s_full_mix_v2_piano_bass_snare.flac. Verdict: pending.

## Crash cymbal (crash_flow.py) - the user identified the crash, not the snare, as the white-noise sound

The crash had no learned stage (no real data). Measured: drum_set_fea crash has 73 % of its energy above 8 kHz and 2 % below
2 kHz, for the whole 1.5 s; real crashes are about flat 200 Hz-8 kHz and lose the top quickly.
Real data: Iowa MIS suspended crash cymbals 13/17/18/20 inch, stick, normal + shoulder, mf + ff (fetch_crash.py, 14 hits;
pp files dropped: peak 0.002). Source changed to the project's tone_lab crash (512 modes, read only) because a gain model
cannot create body from 32 modes. Model: log-gain = g(f) - rate(f)*age + h(f, age). 17 inch cymbal held out (4 hits):
band error vs real over 0-2.2 s, 200 Hz-16 kHz: source 8.5 dB -> model 2.9 dB mean abs; 8-16 kHz from +6..+20 dB to
within 1 dB (except -5 at 2.2 s); 200-500 Hz still 2-10 dB low. Energy above 8 kHz: 73 % (old crash) -> 14 %.
In the song the crash level is a hand rule (3 dB under the old crash by RMS).
Song: runs/rnf_song_ascend/ascend_first_120s_full_mix_v3_new_crash.flac. Verdict: pending.

## Crash: measured ring partials (crash_ring.py, crash_tonality.py)
User verdict on the crash-flow crash (v3 mix): still like white noise, "tin type high note missing".
Measured on the held-out 17 inch cymbal (0.5-1.5 s after the hit): the 10 strongest partials hold 67/51/43/61 % of the
energy in the 1.2-2.5 / 2.5-5 / 5-10 / 10-16 kHz bands of a real crash; flow-only crash 39/24/21/14 %.
Change: 166 partials (frequency, level, decay) measured from one real training hit (18 inch, shoulder, ff) are added as
decaying sines; the flow output is turned down by the share those partials hold. Not a network: a measured modal fit.
Result on the 17 inch: 47/55/58/24 %. Band-level error 2.9 -> 3.3 dB (4-8 kHz now 3-9 dB too loud; 500-1000 Hz up to
10 dB weak late). In the song each crash is cut at 2.6 s (length of the old crash) with a 50 ms fade; level is the same
hand rule (3 dB under the old crash by RMS). Mix: rnf_song_ascend/ascend_first_120s_full_mix_v4_crash_ring.flac.
Not listened to by the assistant. Originals verified unchanged.

## Ascend: piano sustain pedal (render_song.py piano, PIANO_PEDAL=1)
Asked for by the user. The tab has no pedal marks. Hand rule: pedal changed at each new left-hand chord (0, 3, ... 21 s),
every note rings to the next change, last pedal held to 25.5 s. 76 of 98 notes lengthened. Sympathetic string resonance
is not modelled. Mix: ascend_first_120s_full_mix_v5_piano_pedal.flac; piano alone: piano_alone_{1_no_pedal,2_with_pedal}.flac.
Not listened to by the assistant. Originals verified unchanged.
