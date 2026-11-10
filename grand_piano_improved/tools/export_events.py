#!/usr/bin/env python3
"""Export physical MIDI key and pedal transitions for hardware playback.

Format is one event per line:

    on|off|ped  <time_seconds>  <pitch_or_pedal_state>  <velocity>

Equal-time source order and physical key releases are retained so the hardware
can apply its own sustain and damper-travel behavior.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from midi_events import clip_events, hardware_events, load_performance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--midi", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--sustain-all", action="store_true")
    args = parser.parse_args()

    try:
        performance = load_performance(args.midi, args.sustain_all)
        events, spans = clip_events(performance.events, performance.pedal_spans, args.seconds)
        unsupported = sorted({p for p, _, _, _ in events if not 21 <= p <= 108})
        if unsupported:
            raise ValueError(f"hardware supports MIDI notes 21..108; found {unsupported}")
        transitions = hardware_events(performance, args.seconds)
        lines = [f"{kind} {time:.9f} {pitch} {velocity}"
                 for kind, time, pitch, velocity in transitions]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    for warning in performance.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    print(f"{len(events)} notes, {len(spans)} pedal spans, "
          f"{len(transitions)} transitions -> {args.out}")


if __name__ == "__main__":
    main()
