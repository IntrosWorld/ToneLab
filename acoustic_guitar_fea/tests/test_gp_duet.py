"""Export checks for the requested two-part acoustic Guitar Pro excerpt."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import wave
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import render_gp_duet as duet  # noqa: E402
from test_gp_score import _score  # noqa: E402


def test_exact_thirty_second_excerpt_trims_without_tail_and_fades():
    frames = 30 * 48_000
    source = np.ones((frames + 2000, 2))
    clipped = duet.exact_excerpt(source, frames, 48_000)
    assert clipped.shape == (1_440_000, 2)
    np.testing.assert_array_equal(clipped[:frames - 1200], 1.0)
    assert np.all(np.diff(clipped[-1200:, 0]) <= 0)
    assert np.max(np.abs(clipped[-1])) < 1e-30


def test_short_excerpt_is_padded_with_exact_silence():
    source = np.ones((1000, 2)) * .25
    padded = duet.exact_excerpt(source, 48_000, 48_000)
    np.testing.assert_array_equal(padded[:1000], source)
    assert np.count_nonzero(padded[1000:]) == 0


def _duet_fixture(tmp_path):
    path = _score(tmp_path, [[("Quarter", None),
                              ("Quarter", {"fret": 3}),
                              ("Half", {"fret": 5})]])
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("Content/score.gpif"))
    # Add a sustained rhythm note on the D string; original first track starts
    # after a quarter rest. A third track is present but is never selected.
    ET.SubElement(ET.SubElement(root.find("Bars"), "Bar", id="unused0"),
                  "Voices").text = "rhythm -1 -1 -1"
    ET.SubElement(ET.SubElement(root.find("Voices"), "Voice", id="rhythm"),
                  "Beats").text = "rhythm"
    beat = ET.SubElement(root.find("Beats"), "Beat", id="rhythm")
    ET.SubElement(beat, "Rhythm", ref="Whole")
    ET.SubElement(beat, "Notes").text = "rhythm"
    note = ET.SubElement(root.find("Notes"), "Note", id="rhythm")
    properties = ET.SubElement(note, "Properties")
    for name, child, value in (("String", "String", 2), ("Fret", "Fret", 2),
                               ("Midi", "Number", 52)):
        ET.SubElement(ET.SubElement(properties, "Property", name=name),
                      child).text = str(value)
    excluded = deepcopy(root.find("./Tracks/Track"))
    excluded.set("id", "9")
    excluded.find("Name").text = "Excluded accompaniment"
    root.find("Tracks").append(excluded)
    root.find("./MasterTrack/Tracks").text = "7 8 9"
    root.find("./MasterBars/MasterBar/Bars").text += " unrendered"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("Content/score.gpif", ET.tostring(root))
    return path


def _read_pcm24(path):
    with wave.open(str(path), "rb") as handle:
        assert handle.getnchannels() == 2
        assert handle.getsampwidth() == 3
        assert handle.getframerate() == 12_000
        assert handle.getnframes() == 9000
        packed = np.frombuffer(handle.readframes(handle.getnframes()),
                               dtype=np.uint8).reshape(-1, 3).astype(np.int32)
    values = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
    values = (values ^ 0x800000) - 0x800000
    return values.reshape(-1, 2)


def test_real_duet_cli_exports_only_selected_parts_with_reconstructible_stems(
        tmp_path, monkeypatch):
    source = _duet_fixture(tmp_path)
    output = tmp_path / "duet.wav"
    monkeypatch.setattr(sys, "argv", ["render_gp_duet.py", str(source),
                                    "--seconds", ".75", "--rate", "12000",
                                    "--out", str(output)])
    duet.main()
    manifest = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert manifest["frames"] == 9000
    tracks = manifest["score"]["tracks"]
    assert [track["track_id"] for track in tracks] == ["7", "8"]
    assert [track["render_role"] for track in tracks] == ["lead", "rhythm"]
    assert all(track["render_instrument"] == "FEA steel-string acoustic guitar"
               for track in tracks)
    assert [track["track_id"] for track in manifest["score"]["excluded_tracks"]] == ["9"]
    audio = {role: _read_pcm24(info["path"])
             for role, info in manifest["outputs"].items()}
    assert set(audio) == {"duet", "lead", "rhythm"}
    for rendered in audio.values():
        assert np.count_nonzero(rendered) > 0
        assert np.max(np.abs(rendered)) < 0x7FFFFF
        np.testing.assert_array_equal(rendered[-1], [0, 0])
    # The lead's opening rest survives the complete import/render/export path.
    assert np.count_nonzero(audio["lead"][:6000]) == 0
    assert np.count_nonzero(audio["rhythm"][:6000]) > 0
    error = audio["duet"] - audio["lead"] - audio["rhythm"]
    assert np.max(np.abs(error)) <= 1  # Only independent PCM quantization.
    assert max(info["peak_dbfs"] for info in manifest["outputs"].values()) == pytest.approx(-1.4)
