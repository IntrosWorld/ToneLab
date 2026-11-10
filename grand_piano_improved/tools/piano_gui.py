#!/usr/bin/env python3
"""Responsive desktop MIDI renderer with isolated, cancellable render jobs."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import wave

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))
ROOMS = {"Dry (0%)": 0.0, "Light (18%)": 0.18,
         "Balanced (34%)": 0.34, "More room (50%)": 0.50}


class RenderCancelled(Exception):
    """The user cancelled before a finished WAV was committed."""


def finite_number(value, name, minimum, maximum, *, inclusive=True):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number.") from exc
    valid = minimum <= number <= maximum if inclusive else minimum < number <= maximum
    if not math.isfinite(number) or not valid:
        boundary = "at least" if inclusive else "greater than"
        raise ValueError(f"{name} must be {boundary} {minimum:g} and at most {maximum:g}.")
    return number


@dataclass(frozen=True)
class RenderOptions:
    midi: Path
    output: Path
    seconds: float | None = None
    sample_rate: int = 48000
    tail: float = 4.0
    room_mix: float = 0.34
    sustain_all: bool = False
    normalize: bool = True


def render_command(options, destination):
    executable = Path(sys.executable)
    if os.name == "nt" and executable.name.lower() == "pythonw.exe":
        executable = executable.with_name("python.exe")
    command = [str(executable), "-u", str(PROJECT_ROOT / "tools" / "render_model.py"),
               "--midi", str(options.midi), "--out", str(destination),
               "--rate", str(options.sample_rate), "--tail", str(options.tail),
               "--room-mix", str(options.room_mix)]
    if options.seconds is not None:
        command.extend(["--seconds", str(options.seconds)])
    if options.sustain_all:
        command.append("--sustain-all")
    if not options.normalize:
        command.append("--no-normalize")
    return command


def parse_progress(line):
    match = re.fullmatch(r"PROGRESS\s+(\d+(?:\.\d+)?)\s+(.*)", line.strip())
    if match:
        return min(100.0, max(0.0, float(match[1]))), match[2]
    return None


def run_subprocess(command, cancelled, progress):
    """Drain child output without blocking cancellation or touching Tk."""
    if cancelled.is_set():
        raise RenderCancelled()
    lines = queue.Queue()
    history = deque(maxlen=60)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                               errors="replace", bufsize=1, env=env, creationflags=flags)

    def read_output():
        try:
            for line in process.stdout:
                lines.put(line.rstrip())
        finally:
            lines.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    try:
        while True:
            if cancelled.is_set():
                raise RenderCancelled()
            try:
                line = lines.get(timeout=0.1)
            except queue.Empty:
                continue
            if line is None:
                break
            history.append(line)
            update = parse_progress(line)
            if update:
                progress(*update)
        code = process.wait()
        if cancelled.is_set():
            raise RenderCancelled()
        if code:
            detail = "\n".join(history).strip()
            raise RuntimeError(detail or f"Renderer exited with code {code}.")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        reader.join(timeout=2.0)
        process.stdout.close()


def render_file(options, cancelled, progress):
    """Commit a validated WAV only after a successful, uncancelled render."""
    began = time.monotonic()
    output = options.output.resolve()
    if output == options.midi.resolve():
        raise ValueError("The output must be different from the input MIDI file.")
    if output.suffix.lower() != ".wav":
        raise ValueError("Choose an output filename ending in .wav.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".piano-render-", dir=output.parent)).resolve()
    destination = temporary / "render.wav"
    try:
        run_subprocess(render_command(options, destination), cancelled, progress)
        with wave.open(str(destination), "rb") as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (2, 3, options.sample_rate):
                raise RuntimeError("The renderer produced an unexpected audio format.")
            frames = wav.getnframes()
            if not frames:
                raise RuntimeError("The renderer produced an empty WAV.")
            duration = frames / wav.getframerate()
            wav.setpos(frames - 1)
            if len(wav.readframes(1)) != 6:
                raise RuntimeError("The renderer produced an incomplete WAV.")
        if cancelled.is_set():
            raise RenderCancelled()
        os.replace(destination, output)
        return output, duration, time.monotonic() - began
    finally:
        # Only this job's generated, flat temporary directory is eligible.
        if temporary.parent == output.parent and temporary.name.startswith(".piano-render-"):
            for generated in temporary.iterdir():
                if generated.is_file():
                    generated.unlink(missing_ok=True)
            temporary.rmdir()


def error_log(detail):
    try:
        folder = PROJECT_ROOT / "out" / "logs"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "piano_gui.log"
        with path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}]\n{detail}\n")
        return path
    except OSError:
        return None


class PianoGUI:
    def __init__(self, root):
        self.root = root
        root.title("Grand Piano — MIDI to WAV")
        root.minsize(650, 490)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.report_callback_exception = self._callback_error
        self.midi_path = tk.StringVar()
        self.whole = tk.BooleanVar(value=True)
        self.seconds = tk.StringVar(value="30")
        self.tail = tk.StringVar(value="4")
        self.rate = tk.StringVar(value="48000")
        self.room = tk.StringVar(value="Balanced (34%)")
        self.sustain_all = tk.BooleanVar(value=False)
        self.normalize = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Choose a MIDI file to begin.")
        self.metadata = tk.StringVar(value="24-bit stereo WAV · 48 kHz by default")
        self.results = queue.Queue()
        self.cancelled = threading.Event()
        self.busy = False
        self.closing = False
        self.last_output = None
        self._metadata_token = 0
        self._metadata_after = None
        self._editable = []

        frame = ttk.Frame(root, padding=20)
        frame.grid(sticky="nsew")
        frame.columnconfigure(0, weight=1)
        ttk.Label(frame, text="Grand Piano", font=("Segoe UI", 20, "bold")).grid(sticky="w")
        ttk.Label(frame, text="Render MIDI performances to a stereo WAV.").grid(sticky="w", pady=(0, 16))

        source = ttk.LabelFrame(frame, text="MIDI performance", padding=12)
        source.grid(sticky="ew", pady=(0, 12))
        source.columnconfigure(0, weight=1)
        entry = ttk.Entry(source, textvariable=self.midi_path, width=58)
        entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        browse = ttk.Button(source, text="Browse…", command=self.pick_midi)
        browse.grid(row=0, column=1)
        self._editable.extend([entry, browse])
        ttk.Label(source, textvariable=self.metadata, wraplength=570).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))

        settings = ttk.LabelFrame(frame, text="Render settings", padding=12)
        settings.grid(sticky="ew")
        for column in range(4):
            settings.columnconfigure(column, weight=1)
        whole_button = ttk.Radiobutton(settings, text="Whole file", variable=self.whole,
                                       value=True, command=self._sync)
        whole_button.grid(row=0, column=0, sticky="w")
        preview_button = ttk.Radiobutton(settings, text="First", variable=self.whole,
                                         value=False, command=self._sync)
        preview_button.grid(row=0, column=1, sticky="e", padx=8)
        self.spin = ttk.Spinbox(settings, from_=0.1, to=3600, increment=1,
                                width=8, textvariable=self.seconds)
        self.spin.grid(row=0, column=2, sticky="w")
        ttk.Label(settings, text="seconds + tail").grid(row=0, column=3, sticky="w")
        ttk.Label(settings, text="Release tail (s)").grid(row=1, column=0, sticky="w", pady=(12, 0))
        tail = ttk.Spinbox(settings, from_=0, to=30, width=8, textvariable=self.tail)
        tail.grid(row=1, column=1, sticky="w", pady=(12, 0))
        ttk.Label(settings, text="Sample rate (Hz)").grid(row=1, column=2, sticky="w", pady=(12, 0))
        rate = ttk.Combobox(settings, values=("44100", "48000", "96000"), width=10,
                            textvariable=self.rate, state="readonly")
        rate.grid(row=1, column=3, sticky="w", pady=(12, 0))
        ttk.Label(settings, text="Room amount").grid(row=2, column=0, sticky="w", pady=(12, 0))
        room = ttk.Combobox(settings, values=tuple(ROOMS), textvariable=self.room,
                            state="readonly", width=19)
        room.grid(row=2, column=1, columnspan=2, sticky="w", pady=(12, 0))
        normalize = ttk.Checkbutton(settings, text="Normalize level", variable=self.normalize)
        normalize.grid(row=2, column=3, sticky="w", pady=(12, 0))
        sustain = ttk.Checkbutton(settings, text="Hold sustain throughout (override MIDI pedal)",
                                  variable=self.sustain_all)
        sustain.grid(row=3, column=0, columnspan=4, sticky="w", pady=(12, 0))
        self._editable.extend([whole_button, preview_button, self.spin, tail, rate, room,
                               normalize, sustain])

        actions = ttk.Frame(frame)
        actions.grid(sticky="ew", pady=(16, 10))
        actions.columnconfigure(2, weight=1)
        self.render_button = ttk.Button(actions, text="Render to WAV…", command=self.render)
        self.render_button.grid(row=0, column=0, padx=(0, 8))
        self.cancel_button = ttk.Button(actions, text="Cancel", command=self.cancel, state="disabled")
        self.cancel_button.grid(row=0, column=1)
        self.play_button = ttk.Button(actions, text="Play WAV", command=self.play, state="disabled")
        self.play_button.grid(row=0, column=3, padx=(8, 0))
        self.folder_button = ttk.Button(actions, text="Open folder", command=self.open_folder, state="disabled")
        self.folder_button.grid(row=0, column=4, padx=(8, 0))
        self.progress = ttk.Progressbar(frame, mode="determinate", maximum=100)
        self.progress.grid(sticky="ew")
        ttk.Label(frame, textvariable=self.status, wraplength=600).grid(sticky="w", pady=(8, 0))
        self.midi_path.trace_add("write", self._source_changed)
        self._sync()
        root.update_idletasks()
        root.minsize(max(650, root.winfo_reqwidth()), max(490, root.winfo_reqheight()))
        self.root.after(100, self._poll)

    def _sync(self):
        for widget in self._editable:
            widget.state(["disabled" if self.busy else "!disabled"])
        self.spin.state(["disabled" if self.busy or self.whole.get() else "!disabled"])
        self.render_button.state(["disabled" if self.busy else "!disabled"])
        self.cancel_button.state(["!disabled" if self.busy and not self.cancelled.is_set() else "disabled"])

    def _source_changed(self, *_):
        self._metadata_token += 1
        if self._metadata_after is not None:
            self.root.after_cancel(self._metadata_after)
        self._metadata_after = self.root.after(300, self._inspect_source)

    def _inspect_source(self):
        self._metadata_after = None
        path = Path(self.midi_path.get().strip())
        token = self._metadata_token
        if not path.is_file():
            self.metadata.set("Choose an existing .mid or .midi file.")
            return
        self.metadata.set("Reading MIDI…")

        def inspect():
            try:
                from midi_events import load_events
                events, spans = load_events(path)
                duration = max((r for _, _, _, r in events), default=0.0)
                pedals = f"{len(spans)} pedal spans" if spans else "no sustain pedal data"
                result = f"{len(events):,} notes · {duration:.1f} seconds · {pedals}"
            except Exception as exc:
                result = f"Could not read MIDI: {exc}"
            self.results.put(("metadata", token, result))

        threading.Thread(target=inspect, daemon=True).start()

    def pick_midi(self):
        current = Path(self.midi_path.get()).parent if self.midi_path.get() else PROJECT_ROOT / "midi"
        path = filedialog.askopenfilename(parent=self.root, title="Choose a MIDI file",
            initialdir=current, filetypes=[("MIDI files", "*.mid *.midi"), ("All files", "*.*")])
        if path:
            self.midi_path.set(path)

    def render(self):
        if self.busy:
            return
        try:
            midi = Path(self.midi_path.get().strip()).resolve()
            if not midi.is_file():
                raise ValueError("Choose an existing MIDI file first.")
            if midi.suffix.lower() not in (".mid", ".midi"):
                raise ValueError("The input filename must end in .mid or .midi.")
            seconds = None if self.whole.get() else finite_number(
                self.seconds.get(), "Preview duration", 0, 86400, inclusive=False)
            tail = finite_number(self.tail.get(), "Release tail", 0, 30)
            sample_rate = int(self.rate.get())
            if sample_rate not in (44100, 48000, 96000):
                raise ValueError("Choose a sample rate from the list.")
            room_mix = ROOMS[self.room.get()]
        except (ValueError, KeyError) as exc:
            messagebox.showerror("Check render settings", str(exc), parent=self.root)
            return

        suffix = f"_{seconds:g}s" if seconds is not None else ""
        out_dir = PROJECT_ROOT / "out"
        out = filedialog.asksaveasfilename(parent=self.root, title="Save stereo WAV",
            defaultextension=".wav", initialdir=out_dir,
            initialfile=f"{midi.stem}{suffix}_improved.wav", filetypes=[("WAV audio", "*.wav")])
        if not out:
            return
        options = RenderOptions(midi, Path(out), seconds, sample_rate, tail, room_mix,
                                self.sustain_all.get(), self.normalize.get())
        if options.output.suffix.lower() != ".wav":
            messagebox.showerror("Check output filename", "Use a filename ending in .wav.", parent=self.root)
            return
        self.cancelled.clear()
        self.busy = True
        self.progress["value"] = 0
        self.status.set("Starting renderer…")
        self._sync()
        threading.Thread(target=self._worker, args=(options,), daemon=True).start()

    def _worker(self, options):
        try:
            result = render_file(options, self.cancelled,
                lambda percent, message: self.results.put(("progress", percent, message)))
            self.results.put(("ok", *result))
        except RenderCancelled:
            self.results.put(("cancelled",))
        except Exception as exc:
            log = error_log(traceback.format_exc())
            detail = next((line.strip() for line in reversed(str(exc).splitlines())
                           if line.strip()), type(exc).__name__)[:1200]
            if log:
                detail += f"\n\nDetails saved to:\n{log}"
            self.results.put(("error", detail))

    def _poll(self):
        for _ in range(200):
            try:
                item = self.results.get_nowait()
            except queue.Empty:
                break
            kind = item[0]
            if kind == "metadata":
                if item[1] == self._metadata_token:
                    self.metadata.set(item[2])
            elif kind == "progress":
                if self.busy and not self.cancelled.is_set():
                    self.progress["value"] = item[1]
                    self.status.set(f"{item[1]:.0f}% · {item[2]}")
            else:
                self.busy = False
                self._sync()
                if kind == "ok":
                    _, output, duration, elapsed = item
                    self.last_output = output
                    self.progress["value"] = 100
                    self.status.set(f"Saved {output.name}\n{duration:.1f} s of audio · rendered in {elapsed:.1f} s")
                    self.play_button.state(["!disabled"])
                    self.folder_button.state(["!disabled"])
                elif kind == "cancelled":
                    self.progress["value"] = 0
                    self.status.set("Render cancelled. Your existing output was preserved.")
                else:
                    self.progress["value"] = 0
                    self.status.set("Render failed. Check the error details and try again.")
                    if not self.closing:
                        messagebox.showerror("Render failed", item[1], parent=self.root)
        if self.closing and not self.busy:
            self.root.destroy()
            return
        self.root.after(100, self._poll)

    def cancel(self):
        if self.busy:
            self.cancelled.set()
            self.status.set("Cancelling render…")
            self._sync()

    def close(self):
        if self.busy:
            self.closing = True
            self.cancel()
            self.status.set("Cancelling render and closing…")
        else:
            self.root.destroy()

    def _callback_error(self, kind, value, trace):
        log = error_log("".join(traceback.format_exception(kind, value, trace)))
        detail = str(value)
        if log:
            detail += f"\n\nDetails saved to:\n{log}"
        if not self.closing:
            messagebox.showerror("Grand Piano error", detail, parent=self.root)

    def _open(self, path):
        try:
            if not path.exists():
                raise FileNotFoundError(f"File or folder no longer exists:\n{path}")
            if os.name == "nt":
                os.startfile(str(path))
            else:
                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])
        except OSError as exc:
            messagebox.showerror("Could not open output", str(exc), parent=self.root)

    def play(self):
        if self.last_output:
            self._open(self.last_output)

    def open_folder(self):
        if self.last_output:
            self._open(self.last_output.parent)


def check_dependencies():
    import numpy  # noqa: F401
    import scipy.signal  # noqa: F401
    import mido  # noqa: F401
    if not (PROJECT_ROOT / "rtl" / "soundboard_modes.json").is_file():
        raise FileNotFoundError("Missing rtl/soundboard_modes.json. Keep the project folder intact.")


def main():
    try:
        check_dependencies()
        if "--check" in sys.argv:
            print("Grand Piano dependencies are ready.")
            return
        if "--launch" in sys.argv and os.name == "nt":
            windowed = Path(sys.executable).with_name("pythonw.exe")
            if windowed.is_file():
                subprocess.Popen([str(windowed), str(Path(__file__).resolve())],
                                 cwd=PROJECT_ROOT, creationflags=subprocess.CREATE_NO_WINDOW)
                return
        root = tk.Tk()
        PianoGUI(root)
        root.mainloop()
    except Exception as exc:
        log = error_log(traceback.format_exc())
        message = f"Grand Piano could not start: {exc}\n\nRun Setup.bat to install dependencies."
        if log:
            message += f"\n\nDetails: {log}"
        if "--check" in sys.argv:
            print(message, file=sys.stderr)
        else:
            try:
                messagebox.showerror("Grand Piano startup error", message)
            except tk.TclError:
                pass
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
