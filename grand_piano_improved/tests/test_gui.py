"""Desktop job isolation: cancelled/failed jobs must never replace user audio."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import threading
import time
import wave

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import piano_gui as gui


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "0", "-1", "words"])
def test_preview_rejects_nonfinite_or_nonpositive_duration(value):
    with pytest.raises(ValueError):
        gui.finite_number(value, "Preview duration", 0, 86400, inclusive=False)


def test_progress_and_child_error_are_reported():
    updates = []
    command = [sys.executable, "-u", "-c",
               "print('PROGRESS 42 Rendering strings', flush=True); "
               "raise ValueError('The MIDI is invalid')"]
    with pytest.raises(RuntimeError, match="The MIDI is invalid"):
        gui.run_subprocess(command, threading.Event(), lambda *args: updates.append(args))
    assert updates == [(42, "Rendering strings")]


def test_cancel_terminates_a_silent_render_process(monkeypatch):
    real_popen = subprocess.Popen
    children = []

    def launch(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(gui.subprocess, "Popen", launch)
    cancelled = threading.Event()
    command = [sys.executable, "-u", "-c",
               "import time; print('PROGRESS 1 Started', flush=True); time.sleep(60)"]
    began = time.monotonic()
    with pytest.raises(gui.RenderCancelled):
        gui.run_subprocess(command, cancelled, lambda *_: cancelled.set())
    assert time.monotonic() - began < 8
    assert len(children) == 1 and children[0].poll() is not None


@pytest.mark.parametrize("failure", ["nonzero", "invalid_wav", "cancel"])
def test_failed_or_cancelled_job_preserves_existing_output(tmp_path, monkeypatch, failure):
    output = tmp_path / "existing song.wav"
    output.write_bytes(b"precious prior audio")
    cancelled = threading.Event()

    def command(options, destination):
        program = "from pathlib import Path; import sys; Path(sys.argv[1]).write_bytes(b'broken'); "
        if failure == "nonzero":
            program += "sys.exit(9)"
        elif failure == "cancel":
            program += "print('PROGRESS 90 Finishing', flush=True)"
        return [sys.executable, "-u", "-c", program, str(destination)]

    monkeypatch.setattr(gui, "render_command", command)
    options = gui.RenderOptions(tmp_path / "input.mid", output)
    with pytest.raises((RuntimeError, EOFError, wave.Error, gui.RenderCancelled)):
        gui.render_file(options, cancelled,
                        lambda *_: cancelled.set() if failure == "cancel" else None)
    assert output.read_bytes() == b"precious prior audio"
    assert list(tmp_path.glob(".piano-render-*")) == []


def test_successful_job_commits_validated_wav(tmp_path, monkeypatch):
    output = tmp_path / "piano with spaces.wav"
    output.write_bytes(b"old audio")

    def command(options, destination):
        program = (
            "import sys,wave; "
            "f=wave.open(sys.argv[1],'wb'); f.setnchannels(2); f.setsampwidth(3); "
            "f.setframerate(48000); f.writeframes(bytes(480*6)); f.close(); "
            "print('PROGRESS 100 Done',flush=True)"
        )
        return [sys.executable, "-u", "-c", program, str(destination)]

    monkeypatch.setattr(gui, "render_command", command)
    result, duration, elapsed = gui.render_file(
        gui.RenderOptions(tmp_path / "input.mid", output), threading.Event(), lambda *_: None)
    assert result == output
    assert duration == pytest.approx(0.01)
    assert elapsed > 0
    with wave.open(str(output)) as wav:
        assert wav.getnframes() == 480
    assert list(tmp_path.glob(".piano-render-*")) == []


def test_tk_render_updates_and_controls_stay_responsive(tmp_path, monkeypatch):
    try:
        root = gui.tk.Tk()
    except gui.tk.TclError:
        pytest.skip("Tk display is unavailable")
    root.withdraw()
    app = gui.PianoGUI(root)
    midi = tmp_path / "song.mid"
    midi.write_bytes(b"test fixture")
    output = tmp_path / "song.wav"
    monkeypatch.setattr(gui.filedialog, "asksaveasfilename", lambda **_: str(output))
    release = threading.Event()

    def render(options, cancelled, progress):
        progress(25, "Synthesizing notes")
        release.wait(4)
        return options.output, 5.0, 0.1

    monkeypatch.setattr(gui, "render_file", render)
    try:
        app.midi_path.set(str(midi))
        app.whole.set(False)
        app.seconds.set("1.5")
        app.render()
        assert app.busy
        assert "disabled" in app.render_button.state()
        assert "disabled" not in app.cancel_button.state()
        root.update()
        release.set()
        deadline = time.monotonic() + 4
        while app.busy and time.monotonic() < deadline:
            root.update()
            time.sleep(0.01)
        assert not app.busy
        assert app.last_output == output
        assert app.progress["value"] == 100
        assert "disabled" not in app.play_button.state()
        assert "disabled" not in app.spin.state()
        assert "Saved song.wav" in app.status.get()
    finally:
        release.set()
        root.destroy()
