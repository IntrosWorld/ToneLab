"""Conservative hand-performance preparation for tablature events.

This layer changes touch and the finite travel of a chord stroke. It does not
add notes, fill rests, retrigger ties, move the melodic grid, or invent an
alternating strum pattern. Source timings remain available to the string engine
so a short chord sweep cannot accidentally turn a connected note into a rest.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy

import numpy as np


def prepare_performance_events(events, role, seed):
    """Return independent event dictionaries with reproducible playing touch.

    Velocity offsets are limited to eight MIDI units around each written
    dynamic. The phrase contour is deliberately small; adjacent strings share
    one hand-force variation. Chord spacing follows physical string positions,
    not list order, and is capped at 50 ms across the entire instrument.

    Optional ``stroke_direction`` values ``up`` and ``down`` describe physical
    travel from high-to-low and low-to-high, respectively. Unmarked rhythm
    chords retain the existing downstroke interpretation. A melodic note's
    start and duration are unchanged, preserving the score's HOPO junctions.
    """
    if role not in ("lead", "rhythm"):
        raise ValueError("role must be lead or rhythm")
    rng = np.random.default_rng(seed)
    grouped = defaultdict(list)
    for source in events:
        event = deepcopy(source)
        event["score_start"] = float(source["start"])
        event["score_duration"] = float(source["duration"])
        event["score_velocity"] = int(source["velocity"])
        grouped[(round(event["start"], 8), event.get("voice", 0))].append(event)
    groups = [grouped[key] for key in sorted(grouped)]
    if not groups:
        return []

    # The phrase boundaries follow written silence, not an invented bar grid.
    phrases, phrase, sounding_until = [], [], float("-inf")
    for group in groups:
        start = group[0]["score_start"]
        if phrase and start - sounding_until > .09:
            phrases.append(phrase)
            phrase = []
        phrase.append(group)
        sounding_until = max(sounding_until, max(
            e["score_start"] + e["score_duration"] for e in group))
    if phrase:
        phrases.append(phrase)

    result = []
    for phrase_index, phrase in enumerate(phrases):
        hand_touch = float(np.clip(rng.normal(0., 1.4), -2.5, 2.5))
        for group_index, group in enumerate(phrase):
            # Correlated wrist/finger force changes smoothly between strokes.
            hand_touch = float(np.clip(.65 * hand_touch + rng.normal(0., 1.1),
                                       -3., 3.))
            progress = group_index / max(1, len(phrase) - 1)
            contour = 1.8 * np.sin(np.pi * progress) - 1.5 * progress
            if group_index == 0:
                contour += 1.0
            marked = {e.get("stroke_direction") for e in group}
            direction = "up" if "up" in marked else "down"
            group.sort(key=lambda e: e["string_index"], reverse=direction == "up")
            sweep = role == "rhythm" and len(group) > 1
            string_span = abs(group[-1]["string_index"] - group[0]["string_index"])
            # Slight hand acceleration makes a stroke one gesture, avoiding a
            # chain of perfectly equally spaced, equally strong MIDI attacks.
            sweep_s = min(.050, (.0088 + rng.uniform(-.0012, .0012)) * string_span)
            sweep_s = min(sweep_s, min(e["score_duration"] for e in group) * .2)
            for order, event in enumerate(group):
                position = (abs(event["string_index"] - group[0]["string_index"])
                            / max(1, string_span))
                delay = sweep_s * position ** .92 if sweep else 0.
                event["start"] = event["score_start"] + delay
                event["duration"] = event["score_duration"] - delay
                event["phrase_index"] = phrase_index
                if sweep:
                    event["stroke_direction"] = direction
                    event["stroke_order"] = order
                    event["stroke_delay"] = delay
                individual = float(np.clip(rng.normal(0., .65), -1.5, 1.5))
                # At a light brush's leading edge the pick meets more string
                # resistance; retain some top-string voice without equalizing
                # all six attacks independently.
                stroke_touch = 1.5 - 3. * position if sweep else 0.
                offset = int(np.clip(np.rint(hand_touch + contour + individual
                                            + stroke_touch), -8, 8))
                event["velocity"] = int(np.clip(event["score_velocity"] + offset,
                                                 1, 127))
                result.append(event)
    result.sort(key=lambda e: (e["start"], e["string_index"]))
    return result
