"""Audit the actual exported WAV files, not just the pre-export render metadata."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import wave

import numpy as np

from tone_lab import dsp

HERE = Path(__file__).resolve().parent


class AudioLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "audio":
            self.links.append(attributes["src"])
        if tag == "button" and "data-file" in attributes:
            self.links.append("out/"+attributes["data-file"]+".wav")


def main():
    manifest = json.loads((HERE/"out/manifest.json").read_text())
    reports = {}
    for name, entry in manifest["renders"].items():
        path = Path(entry["file"])
        with wave.open(str(path)) as wav:
            assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (2, 3, dsp.SR), name
        audio = dsp.read_wav(path)
        actual = dsp.metrics(audio)
        assert actual["finite"], name
        assert actual["true_peak_4x_dbtp"] <= -1.49, name
        assert actual["dc_max"] < .001, name
        assert np.max(np.abs(audio[-1])) < 4e-7, name
        assert abs(actual["seconds"]-entry["seconds"]) < 1/dsp.SR, name
        assert abs(actual["active_rms_dbfs"]-entry["active_rms_dbfs"]) < .01, name
        reports[name] = {**actual, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    assert len(reports) == 16, "Run the full render suite before auditing."
    for instrument in ("bass", "drums"):
        new_name = "gp2_bass_finger" if instrument == "bass" else "gp2_drums_acoustic"
        old_name = f"gp2_{instrument}_legacy_matched"
        assert abs(reports[new_name]["active_rms_dbfs"]-reports[old_name]["active_rms_dbfs"]) < .1
    parser = AudioLinks()
    parser.feed((HERE/"listen.html").read_text(encoding="utf-8"))
    assert all((HERE/link).is_file() for link in parser.links), "Listening page has missing audio."
    result = {"result": "pass", "wav_files": len(reports), "page_audio_links": len(parser.links),
              "checks": ["PCM24/48k/stereo", "finite", "true_peak_4x <= -1.49 dBTP", "DC < .001",
                         "fade ends at silence", "manifest matches PCM", "A/B levels within .1 dB", "all page audio exists"],
              "perceptual_limit": "Technical validation only; no claim of recorded-instrument equivalence.",
              "files": reports}
    (HERE/"out/validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"PASS: {len(reports)} exported PCM24 WAVs, {len(parser.links)} listening-page audio links; A/B pairs matched within 0.1 dB.")


if __name__ == "__main__":
    main()
