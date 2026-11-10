#!/usr/bin/env python3
"""Floating-point acoustic-guitar physical-model renderer."""
from __future__ import annotations

import argparse
from pathlib import Path
import wave

import numpy as np
from acoustic_engine import AcousticGuitar, GuitarBody, _resonator

ROOT = Path(__file__).resolve().parents[1]


def load_events(path: Path):
    import mido

    midi = mido.MidiFile(str(path))
    now = 0.0
    active = {}
    events = []
    for msg in midi:
        now += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            active[msg.note] = (msg.velocity, now)
        elif msg.type == "note_off" or (
            msg.type == "note_on" and msg.velocity == 0
        ):
            item = active.pop(msg.note, None)
            if item is not None and 40 <= msg.note <= 88:
                velocity, start = item
                events.append((msg.note, velocity, start, max(0.04, now - start)))
    for note, (velocity, start) in active.items():
        if 40 <= note <= 88:
            events.append((note, velocity, start, 1.0))
    return events


def demo_events():
    """Open-string arpeggio, picked chord, and short melodic cadence."""
    events = []
    t = 0.0
    for note in (40, 45, 50, 55, 59, 64):
        events.append((note, 86, t, 2.4))
        t += 0.34
    t += 0.55
    # G - Cadd9 - Em7 - Dsus4, voiced as guitar chords.
    for chord in ((43, 50, 55, 59, 62, 67), (48, 52, 55, 62, 64),
                  (40, 47, 52, 55, 59, 64), (50, 57, 62, 67)):
        for j, note in enumerate(chord):
            events.append((note, 92 - 3 * j, t + 0.024 * j, 1.55))
        t += 1.18
    for note in (64, 67, 71, 69, 67, 64, 62, 64):
        events.append((note, 78, t, 0.46))
        t += 0.36
    return events


def emotional_arpeggio_events(tempo: float = 68.0):
    """Slow, humanized fingerstyle study in G major.

    Progression: Gadd9 - D/F# - Em7 - Cmaj9 - G/B - Am7 -
    Dsus4/D - G6/9. Each chord occupies one 4/4 bar and is articulated as
    eighth notes. Shared open-string tones make the harmony flow naturally
    rather than sounding like eight unrelated block chords.
    """
    eighth = 30.0 / tempo
    bar = 8.0 * eighth
    # Low to high guitar voicings. MIDI pitches stay inside the physical
    # model's E2-E6 range and favor open-position sonorities.
    chords = (
        (43, 50, 55, 59, 62, 69),  # Gadd9
        (42, 50, 57, 62, 66, 69),  # D/F#
        (40, 47, 50, 55, 59, 64),  # Em7
        (48, 52, 55, 59, 62, 67),  # Cmaj9
        (47, 50, 55, 59, 62, 67),  # G/B
        (45, 52, 55, 60, 64, 67),  # Am7
        (50, 57, 62, 67, 69, 74),  # Dsus4; F# resolves below
        (43, 50, 55, 59, 64, 69),  # G6/9
    )
    patterns = (
        (0, 2, 3, 4, 5, 4, 3, 2),
        (0, 2, 4, 3, 5, 4, 3, 2),
        (0, 1, 3, 4, 5, 4, 3, 1),
        (0, 2, 3, 5, 4, 5, 3, 2),
    )
    # A gentle arc: intimate opening, lift through Cmaj9/Am7, relaxed ending.
    bar_levels = (78, 80, 76, 84, 82, 87, 89, 77)
    events = []
    for bar_index, chord in enumerate(chords):
        pattern = patterns[bar_index % len(patterns)]
        start = bar_index * bar
        for step, voice in enumerate(pattern):
            note = chord[voice]
            # Resolve the suspended G4 to F#4 near the end of the D bar.
            if bar_index == 6 and step >= 6 and note == 67:
                note = 66
            accent = 8 if step in (0, 4) else (3 if step in (2, 6) else 0)
            # Tiny deterministic offsets prevent a mechanically quantized feel.
            human_time = (0.0, 0.010, -0.006, 0.007, -0.004, 0.012,
                          -0.007, 0.004)[step]
            velocity = min(112, bar_levels[bar_index] + accent)
            duration = 2.55 if step < 6 else 2.15
            events.append((note, velocity,
                           max(0.0, start + step * eighth + human_time),
                           duration))

    # Let the final G6/9 bloom: softly re-articulate bass and singing top A.
    final = 8.0 * bar
    events.extend([
        (43, 72, final - 0.48 * bar, 4.2),
        (62, 66, final - 0.34 * bar, 3.7),
        (69, 70, final - 0.20 * bar, 3.5),
    ])
    return events


def sunrise_arpeggio_events(tempo: float = 104.0):
    """Bright, propulsive fingerpicked piece designed to lift immediately.

    The harmony lives in D major and keeps common open-string colors ringing.
    A bass-note + upper-string pinch opens every bar, the inner eighth-note
    pattern supplies motion, and the top voice climbs toward the final D6/9.
    """
    eighth = 30.0 / tempo
    bar = 8.0 * eighth
    chords = (
        (50, 57, 62, 66, 69, 76),  # Dadd9
        (49, 52, 57, 61, 64, 71),  # Aadd9/C#
        (47, 54, 57, 62, 66, 69),  # Bm7
        (43, 50, 55, 59, 62, 66),  # Gmaj9
        (42, 50, 57, 62, 66, 69),  # D/F#
        (43, 50, 55, 59, 62, 69),  # Gadd9
        (47, 54, 59, 62, 66, 71),  # Bm11
        (45, 52, 57, 62, 64, 69),  # Asus4; C# resolution below
        (40, 47, 52, 55, 59, 66),  # Em9
        (43, 50, 55, 59, 66, 71),  # Gmaj7
        (45, 52, 57, 61, 64, 71),  # Aadd9
        (50, 57, 62, 66, 71, 76),  # D6/9
    )
    patterns = (
        (0, 3, 2, 5, 1, 4, 2, 5),
        (0, 2, 4, 5, 1, 3, 4, 5),
        (0, 3, 1, 4, 2, 5, 3, 4),
        (0, 2, 3, 5, 1, 4, 3, 5),
    )
    # Two waves of energy, with the second carrying the strongest lift.
    levels = (91, 89, 88, 94, 94, 97, 95, 99, 96, 101, 104, 96)
    timing = (0.0, 0.006, -0.005, 0.008, -0.004, 0.005, -0.006, 0.004)
    events = []
    for bar_index, chord in enumerate(chords):
        start = bar_index * bar
        pattern = patterns[bar_index % len(patterns)]
        for step, voice in enumerate(pattern):
            note = chord[voice]
            # Asus4 becomes A major on the back half of its bar.
            if bar_index == 7 and step >= 4 and note == 62:
                note = 61
            accent = 7 if step == 0 else (5 if step == 4 else 0)
            events.append((note, min(118, levels[bar_index] + accent),
                           start + step * eighth + timing[step], 1.38))

        # Thumb + two-finger pinches make the groove read instantly. They are
        # rolled by milliseconds rather than landing as synthetic block chords.
        for beat_step in (0, 4):
            pinch_time = start + beat_step * eighth
            pinch_level = levels[bar_index] + (5 if beat_step == 0 else 1)
            for delay, voice in ((0.018, 3), (0.036, 4), (0.054, 5)):
                note = chord[voice]
                if bar_index == 7 and beat_step == 4 and note == 62:
                    note = 61
                events.append((note, min(116, pinch_level),
                               pinch_time + delay, 1.10))

    # Wide final voicing: the stagger preserves the physical guitar attack
    # while ending with an unmistakably sunny E-over-D color.
    final = len(chords) * bar
    for delay, note, velocity in (
        (0.00, 50, 104), (0.035, 57, 98), (0.070, 62, 96),
        (0.105, 66, 94), (0.140, 71, 96), (0.175, 76, 101),
    ):
        events.append((note, velocity, final + delay, 4.6))
    return events


def sunshine_strum_events(tempo: float = 100.0, low_frets: bool = False):
    """Upbeat acoustic strumming with alternating, clearly exposed bass.

    Downstrokes travel low-to-high, upstrokes high-to-low, and each half-bar
    begins with a thumb/pick bass note before the chord arrives. Shorter
    offbeat strokes keep the body model lively instead of producing a wash.
    """
    beat = 60.0 / tempo
    bar = 4.0 * beat
    bright_chords = (
        (50, 57, 62, 66, 69, 76),  # Dadd9
        (49, 52, 57, 61, 64, 71),  # A/C#
        (47, 54, 57, 62, 66, 69),  # Bm7
        (43, 50, 55, 59, 62, 69),  # Gadd9
        (42, 50, 57, 62, 66, 69),  # D/F#
        (43, 50, 55, 59, 62, 71),  # Gmaj7
        (47, 54, 59, 62, 66, 71),  # Bm11
        (45, 52, 57, 62, 64, 69),  # Asus4
        (40, 47, 52, 55, 59, 66),  # Em9
        (43, 50, 55, 59, 62, 69),  # Gadd9
        (45, 52, 57, 61, 64, 71),  # Aadd9
        (50, 57, 62, 66, 71, 76),  # D6/9
    )
    # Open-position/lower-fret alternatives. A literal octave shift would put
    # the bass below the modeled (and real standard-tuned) low E; these keep
    # the bass intact while folding the upper chord tones down by one octave.
    low_chords = (
        (45, 50, 54, 57, 62, 64),  # Dadd9, open-position color
        (49, 52, 57, 59, 61, 64),  # A/C# add9
        (47, 50, 54, 57, 59, 62),  # Bm7
        (43, 50, 55, 57, 59, 62),  # Gadd9
        (42, 50, 54, 57, 62, 64),  # D/F#
        (43, 50, 54, 55, 59, 62),  # Gmaj7
        (47, 50, 52, 54, 57, 59),  # Bm11
        (45, 50, 52, 57, 59, 62),  # Asus4
        (40, 47, 50, 52, 54, 55),  # Em9
        (43, 50, 55, 57, 59, 62),  # Gadd9
        (45, 47, 49, 52, 57, 61),  # Aadd9
        (45, 47, 50, 52, 54, 57),  # D6/9
    )
    chords = low_chords if low_frets else bright_chords
    # True root/alternate-bass pairs, independent of stored voicing order.
    # This matters for D: its useful low A2 lies below the D3 root.
    bass_pairs = (
        (50, 45), (49, 45), (47, 54), (43, 50),
        (42, 50), (43, 50), (47, 54), (45, 52),
        (40, 47), (43, 50), (45, 52), (50, 45),
    )
    energy = (90, 88, 87, 93, 94, 97, 95, 100, 96, 102, 106, 99)
    events = []

    def stroke(chord, when, velocity, direction="down", duration=0.72,
               upper_only=False):
        voices = list(range(2 if upper_only else 0, len(chord)))
        if direction == "up":
            voices.reverse()
        spacing = 0.010 if direction == "down" else 0.008
        for order, voice in enumerate(voices):
            # Energy is greatest at the leading edge. On a downstroke that is
            # the wound bass side, a major reason full real strums sound weighty.
            lead = round(7.0 * (len(voices) - 1 - order)
                         / max(1, len(voices) - 1))
            events.append((chord[voice], min(119, velocity + lead),
                           when + order * spacing, duration))

    for bar_index, chord in enumerate(chords):
        start = bar_index * bar
        level = energy[bar_index]

        # Alternating root/fifth bass. The 70 ms gap before the downstroke lets
        # the low note speak instead of being hidden inside the chord transient.
        root_bass, alternate_bass = bass_pairs[bar_index]
        events.append((root_bass, min(120, level + 14), start, 1.16))
        # The pick continues onto the chord above the isolated bass rather than
        # immediately creating a second independent copy of the same string.
        stroke(chord, start + 0.115 * beat, level + 2, "down", 0.90,
               upper_only=True)
        stroke(chord, start + 0.82 * beat, level - 10, "up", 0.40,
               upper_only=True)
        stroke(chord, start + 1.48 * beat, level - 3, "down", 0.56,
               upper_only=True)
        stroke(chord, start + 1.86 * beat, level - 12, "up", 0.34,
               upper_only=True)

        events.append((alternate_bass, min(119, level + 11),
                       start + 2.0 * beat, 1.12))
        stroke(chord, start + 2.115 * beat, level + 1, "down", 0.82,
               upper_only=True)
        stroke(chord, start + 2.82 * beat, level - 9, "up", 0.38,
               upper_only=True)
        stroke(chord, start + 3.46 * beat, level - 1, "down", 0.48,
               upper_only=True)
        stroke(chord, start + 3.82 * beat, level - 11, "up", 0.30,
               upper_only=True)

    # Final root, then a broad low-to-high D6/9 stroke allowed to ring.
    final = len(chords) * bar
    events.append((50, 112, final, 4.8))
    stroke(chords[-1], final + 0.13, 108, "down", 4.5)
    return events


def write_wav(path: Path, stereo: np.ndarray, sample_rate: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    if stereo.ndim != 2 or stereo.shape[1] != 2 or not np.all(np.isfinite(stereo)):
        raise ValueError("expected a finite stereo waveform")
    values = (np.clip(stereo, -1.0, 1.0) * ((1 << 23) - 1)).astype(np.int32)
    packed = np.empty((*values.shape, 3), dtype=np.uint8)
    for byte in range(3):
        packed[..., byte] = (values >> (8 * byte)) & 255
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(3)
        wav.setframerate(sample_rate)
        wav.writeframes(packed.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "demo.wav")
    parser.add_argument("--midi", type=Path)
    parser.add_argument("--rate", type=int, default=48_000)
    parser.add_argument(
        "--technique",
        choices=("warm", "finger", "strum", "pick", "bright"),
        default="finger",
    )
    parser.add_argument("--piece", choices=("demo", "emotional-arpeggio",
                                             "sunrise-arpeggio",
                                             "sunshine-strum",
                                             "sunshine-strum-low"),
                        default="demo",
                        help="built-in composition used when --midi is omitted")
    parser.add_argument("--tempo", type=float, default=68.0,
                        help="tempo for emotional-arpeggio (beats per minute)")
    parser.add_argument("--seconds", type=float,
                        help="render only notes beginning before this time")
    args = parser.parse_args()
    if args.midi:
        events = load_events(args.midi)
    elif args.piece == "emotional-arpeggio":
        events = emotional_arpeggio_events(args.tempo)
    elif args.piece == "sunrise-arpeggio":
        events = sunrise_arpeggio_events(args.tempo)
    elif args.piece == "sunshine-strum":
        events = sunshine_strum_events(args.tempo)
    elif args.piece == "sunshine-strum-low":
        events = sunshine_strum_events(args.tempo, low_frets=True)
    else:
        events = demo_events()
    if args.seconds is not None:
        limit = max(0.0, args.seconds)
        events = [
            (note, velocity, start, min(duration, limit - start))
            for note, velocity, start, duration in events
            if start < limit
        ]
    guitar = AcousticGuitar(args.rate, args.technique)
    audio = guitar.render(events)
    write_wav(args.out, audio, args.rate)
    print(f"rendered {len(events)} notes, {len(audio) / args.rate:.2f} s -> {args.out}")


if __name__ == "__main__":
    main()
