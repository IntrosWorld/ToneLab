# Grand Piano — improved software edition

This working copy is `F:\College\projects\VLSI\grand_piano_improved`. The original `grand_piano` folder is preserved: all **1,211 original files, including `.git` metadata,** matched their recorded SHA256 hashes, with no missing or extra files. See [the integrity audit](validation/original_integrity.json).

The instrument synthesizes piano sound from equations for strings, hammer contact, dampers, soundboard resonances and room response. This update improves MIDI correctness, velocity response, rendering efficiency, sustain behavior, WAV export and the desktop workflow.

## Open the piano

For the Unreal piano viewer, double-click **[PianoPlayer.cmd](PianoPlayer.cmd)**. It opens Blue Smoke directly in a game window, using the same launch approach as the guitar player. Use **[PianoStyles.bat](visual/PianoStyles.bat)** to choose another view. These viewers use the installed Unreal runtime; they do not open the editor interface and are not packaged standalone executables.

For MIDI audio rendering:

1. Double-click **[Piano.bat](Piano.bat)** in this folder.
2. If dependencies are missing, run **[Setup.bat](Setup.bat)** once, then open Piano again. Setup creates `.venv` inside this copy. It requires Python 3.10 or newer with tkinter.
3. Browse to a MIDI file, choose a preview or the whole file, adjust the release tail, room amount, sample rate and sustain, then select **Render to WAV…**.
4. Use **Play WAV** or **Open folder** after rendering. **Cancel** stops the render process and preserves any existing output file.

The app shows MIDI metadata and render progress. GUI sample rates are 44.1, 48 and 96 kHz. Error details are saved in `out/logs/piano_gui.log`.

## Listen to the verified examples

All seven requested performances were parsed in full and rendered as previews from their first 30 seconds, with four seconds of release tail:

| MIDI | Preview notes | Audio length | WAV |
| --- | ---: | ---: | --- |
| demo9 | 176 | 34.000 s | [Listen](output/improved/demo9_30s_improved.wav) |
| demo10 | 254 | 34.000 s | [Listen](output/improved/demo10_30s_improved.wav) |
| demo11 | 174 | 34.000 s | [Listen](output/improved/demo11_30s_improved.wav) |
| demo12 | 140 | 34.000 s | [Listen](output/improved/demo12_30s_improved.wav) |
| demo13 | 199 | 34.000 s | [Listen](output/improved/demo13_30s_improved.wav) |
| demo14 | 203 | 33.861 s | [Listen](output/improved/demo14_30s_improved.wav) |
| demo15 | 219 | 34.000 s | [Listen](output/improved/demo15_30s_improved.wav) |

These are 48 kHz, 24-bit stereo files. Every preview contained finite samples and zero clipping, with a normalized peak of approximately −1.01 dBFS. The tail begins after the last retained note release, so a preview can be slightly shorter than the requested window plus tail. [Detailed measurements](output/improved/suite_report.json).

The parser recovers **495 strikes previously lost across demo9–15**, including **454 in demo13**. A separate [sustain example](output/review/demo13_sustain_review.wav) exercises the pedal override. [Changes and evidence](docs/IMPROVEMENTS.md).

## Command line

Run commands from this folder. Use `.venv\Scripts\python.exe` instead of `python` if you installed through Setup.

```powershell
python tools/render_model.py --midi midi/demo13.mid --seconds 30 --tail 4 --out output/demo13.wav --stats output/demo13.json
python tools/render_model.py --midi midi/demo15.mid --out output/demo15_full.wav
python tools/render_model.py --midi midi/demo13.mid --seconds 15 --sustain-all --room-mix 0.20 --out output/demo13_sustain.wav
```

| Option | Behavior |
| --- | --- |
| `--midi FILE` | Input MIDI; omit it for the built-in demonstration phrase. |
| `--out FILE.wav` | Required 24-bit stereo output filename. |
| `--seconds N` / `--duration N` | Keep notes starting before N seconds and cap their releases at N. Omit for the whole performance. |
| `--tail N` | Release/room tail after the last release; default 4 seconds. |
| `--rate N` / `--sample-rate N` | Sample rate; default 48000. Use 44100 or higher for the full piano bandwidth. |
| `--sustain-all` | Hold the pedal for the performance, overriding MIDI pedal controls. |
| `--sustain-level N` | Scale sympathetic resonance; default 1.0. |
| `--room-mix N`, `--body-mix N` | Room/soundboard mix from 0 to 1; defaults 0.34 and 0.20. |
| `--no-normalize` | Use fixed master gain to retain dynamics between files, with overload attenuation. |
| `--cache-mb N` | Template-cache budget; default 256 MiB. Audio working buffers still scale with duration. |
| `--stats FILE.json` | Save settings, MIDI warnings, timing, cache use and audio measurements. |

## Verify the software

Install test dependencies only when needed:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests/test_physics.py tests/test_midi_events.py tests/test_renderer.py tests/test_renderer_review.py tests/test_gui.py -q
python tools/validate_render.py --seconds 30 --numbers 9 10 11 12 13 14 15
python tools/validate_render.py --baseline-seconds 4
```

The recorded software run passed **105 tests**. [Test log](validation/software_tests.log), [final audio report](validation/final_audio_report.json).

A cold comparison of the same four-second demo15 passage plus two-second tail took 54.90 seconds with the original renderer and 31.20 seconds with this copy, a measured **1.76×** ratio. Other work was running concurrently; this is one observation, not a general speed guarantee. [Original audio](output/improved/demo15_comparison_original.wav), [improved audio](output/improved/demo15_comparison_improved.wav), [timing report](output/improved/comparison_report.json).

## Visual project and background

Open [PianoStyles.bat](visual/PianoStyles.bat) for the five visual choices: **BlueSmoke** (4:3), **EmberGold** (16:9), the retained **NocturneBlue/NocturneEmber** hand scenes, and **StudioHands**, based on the Blender studio reference. See [the visual guide](visual/README.md) and [saved user references](visual/references/BRIEF.md). All five saved-scene checks, final game-view captures and repeated autoplay checks passed, including the invisible presentation pawn. See [the five final captures](visual/README.md#verified-game-view-captures) and [verification details](visual/README.md#verification-status). These are editable Unreal project scenes, not a packaged standalone app or a movie export.

The [original README](docs/ORIGINAL_README.md) is retained as historical design documentation, including the original physical model discussion and earlier measurements. Its old hardware and synthesis statements are not a certification of this software update. Current work and verification focus on the software instrument and its visual presentation.
