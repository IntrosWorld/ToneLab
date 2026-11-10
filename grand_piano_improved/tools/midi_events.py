"""Shared MIDI timing for reference rendering and hardware playback.

The renderer needs sounding release times; hardware needs physical key releases
so its own pedal and damper logic can run. ``load_events`` keeps the original
renderer API, while ``load_performance`` retains both timings and source order.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import math
from pathlib import Path


@dataclass(frozen=True)
class Note:
    channel: int
    pitch: int
    velocity: int
    start: float
    key_release: float
    release: float


@dataclass(frozen=True)
class Message:
    """A physical key or pedal transition, in MIDI source order."""
    kind: str
    time: float
    channel: int
    pitch: int
    velocity: int = 0


@dataclass(frozen=True)
class Performance:
    notes: tuple[Note, ...]
    pedal_spans: tuple[tuple[float, float], ...]
    messages: tuple[Message, ...]
    duration: float
    warnings: tuple[str, ...] = ()

    @property
    def events(self):
        return [(n.pitch, n.velocity, n.start, n.release) for n in self.notes]


@dataclass
class _HeldNote:
    channel: int
    pitch: int
    velocity: int
    start: float
    key_release: float | None = None
    release: float | None = None


def _merge_spans(spans):
    """Union channel pedals for the instrument's shared sympathetic bank."""
    merged = []
    for down, up in sorted(spans):
        if up <= down:
            continue
        if merged and down <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(up, merged[-1][1]))
        else:
            merged.append((down, up))
    return merged


def load_performance(path, sustain_all=False):
    """Read format 0/1 MIDI with channel-local note and sustain state.

    Repeated note-ons on one (channel, pitch) pair with note-offs FIFO; MIDI
    1.0 has no note IDs, so this convention retains every strike. Equal-time
    messages keep mido's stable merged-track order. Dangling keys and pedals
    close at the final timestamp, including end-of-track silence.

    CC64 is sustain, CC123 releases held keys, CC121 resets sustain, and CC120
    ends sounding notes. Other controls and programs do not change this piano.
    Percussion channel 10 and notes outside the piano compass A0..C8 are
    ignored and counted in warnings. Type 2 and SMPTE files cannot define the
    tempo-based performance we need.
    """
    import mido

    path = Path(path)
    try:
        midi = mido.MidiFile(str(path))
    except (OSError, EOFError, ValueError, TypeError) as exc:
        raise ValueError(f"Cannot read MIDI '{path}': {exc}") from exc
    if midi.type == 2:
        raise ValueError("MIDI type 2 contains independent sequences; convert to type 0 or 1 first")
    if midi.ticks_per_beat <= 0:
        raise ValueError("SMPTE MIDI timing is unsupported; use a ticks-per-beat MIDI file")

    notes, messages, spans = [], [], []
    active = defaultdict(deque)
    pending = defaultdict(list)
    pedal = [False] * 16
    pedal_start = [None] * 16
    now = 0.0
    unmatched = forced = ignored_percussion = ignored_range = 0
    sound_off = False
    if sustain_all:
        messages.append(Message("ped", 0.0, 0, 1))

    def release_key(note):
        note.key_release = now
        messages.append(Message("off", now, note.channel, note.pitch))
        if sustain_all or pedal[note.channel]:
            pending[note.channel].append(note)
        else:
            note.release = now

    def set_pedal(channel, down):
        if pedal[channel] == down:
            return
        pedal[channel] = down
        messages.append(Message("ped", now, channel, int(down)))
        if down:
            pedal_start[channel] = now
        else:
            spans.append((pedal_start[channel], now))
            pedal_start[channel] = None
            for note in pending.pop(channel, []):
                note.release = now

    try:
        for msg in midi:
            if not math.isfinite(msg.time) or msg.time < 0:
                raise ValueError("MIDI contains a non-finite or negative message delay")
            now += msg.time
            if msg.is_meta:
                continue
            channel = getattr(msg, "channel", None)
            if channel is None:
                continue
            if channel == 9:
                ignored_percussion += int(msg.type == "note_on" and msg.velocity > 0)
                continue
            if msg.type in ("note_on", "note_off") and not 21 <= msg.note <= 108:
                ignored_range += int(msg.type == "note_on" and msg.velocity > 0)
                continue
            if msg.type == "note_on" and msg.velocity > 0:
                note = _HeldNote(channel, msg.note, msg.velocity, now)
                notes.append(note)
                active[channel, msg.note].append(note)
                messages.append(Message("on", now, channel, msg.note, msg.velocity))
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                queue = active.get((channel, msg.note))
                if queue:
                    release_key(queue.popleft())
                    if not queue:
                        del active[channel, msg.note]
                else:
                    unmatched += 1
            elif msg.type == "control_change":
                if msg.control == 64 and not sustain_all:
                    set_pedal(channel, msg.value >= 64)
                elif msg.control == 121 and not sustain_all:
                    set_pedal(channel, False)
                elif msg.control in (120, 123):
                    for key in [key for key in active if key[0] == channel]:
                        for note in active.pop(key):
                            release_key(note)
                    if msg.control == 120:
                        sound_off = True
                        for note in pending.pop(channel, []):
                            note.release = now
    except (EOFError, ValueError, TypeError, ZeroDivisionError) as exc:
        raise ValueError(f"Cannot decode MIDI '{path}': {exc}") from exc

    for queue in active.values():
        for note in queue:
            forced += 1
            release_key(note)
    for queue in pending.values():
        for note in queue:
            note.release = now
    for channel in range(16):
        if pedal[channel]:
            set_pedal(channel, False)
    if sustain_all:
        spans = [(0.0, now)] if now > 0.0 else []
        messages.append(Message("ped", now, 0, 0))

    warnings = []
    if unmatched:
        warnings.append(f"Ignored {unmatched} unmatched note-off message(s)")
    if forced:
        warnings.append(f"Released {forced} dangling note(s) at end of file")
    if ignored_percussion:
        warnings.append(f"Ignored {ignored_percussion} percussion note(s) on MIDI channel 10")
    if ignored_range:
        warnings.append(f"Ignored {ignored_range} note(s) outside piano range A0..C8 (21..108)")
    if sound_off:
        warnings.append("CC120 all-sound-off is approximated as key release by pitch-only hardware")
    final_notes = tuple(Note(n.channel, n.pitch, n.velocity, n.start,
                            n.key_release, n.release) for n in notes)
    return Performance(final_notes, tuple(_merge_spans(spans)), tuple(messages),
                       now, tuple(warnings))


def load_events(path, sustain_all=False):
    """Return ``(pitch, velocity, start, sounding_release)`` notes and pedal spans."""
    performance = load_performance(path, sustain_all)
    return performance.events, list(performance.pedal_spans)


def clip_events(events, spans, seconds):
    """Keep onsets before a finite nonnegative boundary and close at it.

    Zero seconds is empty; ``None`` keeps the full performance. Pedal spans
    remain positive and non-overlapping.
    """
    if seconds is None:
        return list(events), list(spans)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("seconds must be finite and nonnegative")
    return ([(p, v, s, min(r, seconds)) for p, v, s, r in events if s < seconds],
            _merge_spans((a, min(b, seconds)) for a, b in spans if a < seconds))


def hardware_events(performance, seconds=None):
    """Build ordered ``(kind, seconds, pitch_or_pedal, velocity)`` transitions.

    Hardware has a global pedal and no channel/note IDs. Channel pedal states
    are ORed, and note-off waits until a pitch's final held key is released.
    This prevents one channel cutting another's held note. Independent channel
    pedals cannot be represented exactly by this port. Physical key releases
    preserve hardware damper-travel and quick-repedal behavior.
    """
    if seconds is not None and (not math.isfinite(seconds) or seconds < 0):
        raise ValueError("seconds must be finite and nonnegative")
    held = defaultdict(int)
    pedal = [False] * 16
    out = []
    for msg in performance.messages:
        if seconds is not None and msg.time >= seconds:
            break
        if msg.kind == "on":
            held[msg.pitch] += 1
            out.append(("on", msg.time, msg.pitch, msg.velocity))
        elif msg.kind == "off":
            if held[msg.pitch] > 0:
                held[msg.pitch] -= 1
                if held[msg.pitch] == 0:
                    out.append(("off", msg.time, msg.pitch, 0))
        elif msg.kind == "ped":
            previous = any(pedal)
            pedal[msg.channel] = bool(msg.pitch)
            down = any(pedal)
            if down != previous:
                out.append(("ped", msg.time, int(down), 0))
    if seconds is not None:
        for pitch, count in held.items():
            if count:
                out.append(("off", seconds, pitch, 0))
        if any(pedal):
            out.append(("ped", seconds, 0, 0))
    return out
