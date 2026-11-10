# Acoustic guitar — FEA/physical design model

This is a self-contained steel-string acoustic-guitar sound model. It
uses no recordings, samples, wavetable audio, measured impulse response, or learned
model. It is deliberately **floating-point Python only** at this stage: there
is no Verilog/SystemVerilog yet.

`grand_piano` inspired the workflow—physics first, spatial body solve, fast
reference renderer, tests before hardware—but this folder does not import from
or modify that project.

## What is modeled

- Six string sets in standard E2–A2–D3–G3–B3–E4 tuning, including realistic
  scale length, linear density, tension, steel-core bending stiffness, frets,
  and stiff-string inharmonicity.
- A continuous per-string modal state initialized with the signed `sin(n*pi*p)/n`
  bridge-force spectrum. Pick and finger release develop smoothly without
  erasing upper harmonics. Hammer-ons and pull-offs inherit the vibrating
  string instead of starting a separate note; retriggers damp its old state.
  Positive mechanical bridge mobility adds frequency-specific string decay.
- An alternative traveling-wave string initialized with the signed `sin(n*pi*p)/n`
  bridge-force spectrum of a released triangular displacement, small contact variation,
  a pluck-position comb, tuned fractional-delay and dispersive allpass sections,
  frequency-dependent round-trip loss, velocity brightness and note-off damping.
  An additive modal pluck remains available as `AcousticGuitar(source="modal")`
  for engineering comparison, as does `source="waveguide"`. The default is
  `source="continuous"`.
- An irregular dreadnought-like Sitka-spruce top rather than a rectangle.
- A circular soundhole removed from the mesh.
- Spatially varying bending stiffness from an X brace, upper transverse brace,
  asymmetric tone bars, and hardwood bridge plate.
- Generalized modal energy solve with separate spruce/brace/bridge mass,
  directional composite brace stiffness and a free soundhole edge.
- Coupled top/air modes from a 0.510 × 0.405 m soundboard and 14 litre cavity,
  signed bridge/radiation residues, and causal spatial radiation. Each surface
  element and the soundhole contribute their actual positive travel time to
  the microphone. These short delay kernels are computed from geometry.
- Paired transverse string polarizations with slightly different tuning and
  losses, producing a moving, two-stage decay; a small initial tension glide.
- Preserved velocity dynamics, fixed body/radiation gains, and tab-aware
  string/fret selection with same-string retrigger damping.
- An upper-body radiation approximation above the solved modal range and a small close room
  made from causal early reflections, with a mono-compatible direct component.

The body solve is a reduced engineering model, not a manufacturing-grade shell
mesh. Braces use directional composite beam energies, the back
and sides are rigid. The soundboard and enclosed air are coupled in the modal
solve. String loading is a passive modal reduction of bridge admittance, not
a full simultaneous string/body feedback solve. Those are the main
areas to improve before claiming luthier-grade prediction.

## Build the body and render

From this folder:

```powershell
python tools\soundboard_v3.py --convergence --out model\body_modes_v4.json
python tools\render_model.py --out out\demo.wav
python analysis\validate_render.py out\demo.wav
python -m pytest tests -q
```

Slow emotional-uplifting fingerstyle arpeggio (68 BPM):

```powershell
python tools\render_model.py --piece emotional-arpeggio --technique finger --out out\emotional_uplifting_arpeggio.wav
```

Bright, energetic “instant mood lift” arpeggio (104 BPM):

```powershell
python tools\render_model.py --piece sunrise-arpeggio --tempo 104 --technique bright --out out\sunrise_uplifting_arpeggio.wav
```

Upbeat strumming with alternating bass notes (100 BPM):

```powershell
python tools\render_model.py --piece sunshine-strum --tempo 100 --technique pick --out out\sunshine_bass_strumming.wav
```

The same performance revoiced on open strings and frets 0–4:

```powershell
python tools\render_model.py --piece sunshine-strum-low --tempo 100 --technique strum --out out\real_bassy_low_fret_strumming.wav
```

Pluck styles:

```powershell
python tools\render_model.py --technique warm   --out out\warm.wav
python tools\render_model.py --technique finger --out out\finger.wav
python tools\render_model.py --technique pick   --out out\pick.wav
python tools\render_model.py --technique bright --out out\bright.wav
```

MIDI input is supported for notes E2 through E6:

```powershell
python tools\render_model.py --midi song.mid --out out\song.wav
```

## Guitar Pro acoustic duet

Render only selected lead/rhythm tracks from a modern zipped `.gp`/GPIF file.
Both are played through this steel-string acoustic model, regardless of their
original GM instrument programs:

```powershell
python tools\render_gp_duet.py "C:\Users\Micro\Downloads\assamj.gp" --lead-track 0 --rhythm-track 1 --seconds 30 --reference "C:\Users\Micro\Downloads\test-learn.mp3" --out out\assamj_first30s_acoustic_duet_v4_reference.wav
```

The result is exactly 30 seconds of 48 kHz, stereo, 24-bit PCM, with a short
fade at the excerpt boundary. `_lead.wav` and `_rhythm.wav` share the mix gain
and sum to the duet (within 24-bit rounding). A JSON manifest records original
and performed events, excluded tracks, synthesis settings and SHA-256 hashes.

For `assamj.gp`, the selected tracks are **Lead Guitar** (originally jazz
electric, program 26) and **Rhythm Guitar** (originally nylon acoustic, program
24). Bass, drums and vocals are excluded. The score's 107 BPM, opening rests,
tablature and tied durations are preserved: 46 lead attacks and 47 rhythm
attacks. The lead enters at 4.486 seconds. Five pull-offs and one hammer-on
retain the previous string's phase and energy with a small new finger contact;
no bends occur within the requested first 30 seconds. Repeatable, correlated
phrase touch varies by at most eight velocity units; curved chord sweeps span
at most 50 ms. Melodic onsets, chord endings and written rests are preserved.

The fourth revision addresses the weak body contribution and disconnected
articulations in version three. The geometry-derived body now supplies low/mid
radiation instead of being masked by a broadband raw-force path. Both parts use
finger contact, informed by the supplied fingerstyle reference. Previous WAVs
are retained; this version ends in `_v4_reference`. Install `requirements.txt`
for Numba's fast oscillator loop; a slower Python fallback is available.

The reference's isolated tonal ridges around 2.2–4 kHz persist longer than
version three's. The fourth revision conservatively reduces the intrinsic
high-partial loss slope from .24 to .14 while retaining fundamental damping
and positive mechanical bridge loading. It also raises the lead mix gain by
6 dB to keep the melody audible against the stronger acoustic bass response.
The shared radiation balance uses .35 spatial-body gain and .30 unresolved
upper-mode gain with a 1 kHz transition. Of three compared balances, this
restored the reference-like middle/upper harmonic proportions while retaining
low chord weight; it is a tonal calibration, not a change to modal geometry.

`test-learn.mp3` is an analysis reference only. Its attacks and harmonic decay
are measured independently of recording level; the file is not mixed into,
sampled by, or required to play the synthesized guitar. The optional CLI
argument records its provenance in the render manifest. This is parameter and
physical-model refinement, not neural training or an exact timbre clone.
See `analysis/guitar_feel_v4/reference/README.md` for the measurements and
their limits. Numerical checks establish signal integrity and model behavior;
they do not establish a perceptual match to the recording.

Recorded acoustic notes from the University of Iowa and Jim Woodhouse's
Euphonics examples are used only for analysis of attack, harmonic balance and
decay. They are nylon references, not a calibrated steel-string target, and no
recording is loaded by the synthesizer. See
`analysis/guitar_richness_v3/reference/README.md` for provenance and limitations.

The importer handles dotted/tuplet rhythm, voice-local ties and step tempos.
Repeat navigation and tempo ramps are explicitly rejected. This is GPIF input,
not legacy GP3/GP4/GP5 parsing. The renderer currently requires standard tuning
and physical fret positions that agree with the score's pitch.

## Folder layout

- `tools/guitar_physics.py` — strings, fretting, pluck, losses, and air mode.
- `tools/soundboard_fea.py` — braced-top modal solve.
- `tools/soundboard_v3.py` — dimensioned Ritz top with directional braces and air coupling.
- `tools/render_model.py` — floating-point sound renderer.
- `tools/acoustic_engine.py` — dynamic string excitation and fixed body transfer.
- `tools/plucked_string.py` — tuned traveling-wave string, coherent bridge-force wave and losses.
- `tools/string_performance.py` — persistent string state and finger/pick transitions.
- `tools/guitar_performance.py` — phrase touch and bounded chord hand motion.
- `tools/gp_score.py` — modern Guitar Pro score import.
- `tools/render_gp_duet.py` — selected-track acoustic mix, stems and manifest.
- `model/body_modes_v4.json` — active top/air modes and spatial radiation delay kernels.
- `model/body_modes_v3.json` — previous top/air modal coefficients.
- `model/body_modes.json` — earlier effective-plate coefficients, retained for comparison.
- `analysis/validate_render.py` — WAV sanity metrics.
- `tests/test_physics.py` — physical and numerical invariants.
- `out/` — generated audio.

## Before eventual RTL

The natural hardware shape is a pair of fractional-delay string loops per voice,
plus damped two-state resonators for the body modes and A0 air mode. Before translating it, the next
design pass should add two-way bridge impedance coupling, a calibrated body
geometry and radiation solve, fret/contact interactions, and coefficient quantization experiments.
Only after the floating-point sound and invariants are accepted should those
resonators be converted to fixed point.

The previous effective plate was 1.02 m long. The active solve uses a
0.510 × 0.405 × 0.0028 m top, with 0.237 kg including braces and bridge.
Its first coupled resonances are 85.765 and 176.234 Hz. Frequencies are not
scaled after solving. Damping, higher-frequency radiation and
polarization parameters are plausible synthesis approximations,
not measurements from a particular guitar. The continuous HOPO state transfer
is an energy-bounded modal approximation, not a full finger/string collision solve.

The pluck-force formulation follows the discussion in Jim Woodhouse's
[Synthesising plucked string sounds](https://euphonics.org/5-3-synthesising-plucked-strings/).
Paired polarizations and unequal decay follow the synthesis architecture in
Matti Karjalainen's [New techniques and effects in model-based sound synthesis](https://www.dafx.de/paper-archive/1999/karjalainen2.pdf), section 2.1.
The traveling-wave excitation follows Julius O. Smith's discussion of
[shortened guitar-body excitations](https://www.dsprelated.com/freebooks/pasp/Approximating_Shortened_Excitations_Noise.html)
and [acoustic-guitar string terminations](https://www.dsprelated.com/freebooks/pasp/Acoustic_Guitars.html).
