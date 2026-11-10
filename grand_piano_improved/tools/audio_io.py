"""Validated, atomic, vectorised 24-bit stereo WAV export."""
from __future__ import annotations

import os
import tempfile
import wave
from pathlib import Path

import numpy as np


def write_wav(path, stereo, sample_rate):
    data = np.asarray(stereo)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("WAV audio must have shape (frames, 2)")
    if not np.isfinite(data).all():
        raise ValueError("WAV audio contains non-finite samples")
    if int(sample_rate) != sample_rate or not 8000 <= sample_rate <= 192000:
        raise ValueError("sample rate must be an integer from 8000 to 192000 Hz")
    if data.size * 3 > 0xFFFFFFFF - 36:
        raise ValueError("audio exceeds the 4 GiB WAV limit; render shorter sections")
    path = Path(path)
    if path.suffix.lower() != ".wav":
        raise ValueError("output filename must end in .wav")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        with wave.open(temporary, "wb") as output:
            output.setnchannels(2)
            output.setsampwidth(3)
            output.setframerate(int(sample_rate))
            # Bounded working memory even for multi-minute files.
            for begin in range(0, len(data), 65536):
                ints = np.rint(np.clip(data[begin:begin + 65536], -1.0, 1.0)
                              * ((1 << 23) - 1)).astype("<i4")
                packed = ints.view(np.uint8).reshape(-1, 4)[:, :3].copy()
                output.writeframesraw(packed.tobytes())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def audio_metrics(stereo, sample_rate):
    data = np.asarray(stereo, dtype=np.float64)
    peak = float(np.max(np.abs(data), initial=0.0))
    rms = float(np.sqrt(np.mean(data * data))) if data.size else 0.0
    return {
        "sample_rate": int(sample_rate), "frames": len(data),
        "duration_seconds": len(data) / sample_rate,
        "peak_dbfs": float(20 * np.log10(max(peak, 1e-15))),
        "rms_dbfs": float(20 * np.log10(max(rms, 1e-15))),
        "clipped_samples": int(np.count_nonzero(np.abs(data) >= 1.0)),
        "finite": bool(np.isfinite(data).all()),
        "dc_offset": data.mean(axis=0).tolist() if len(data) else [0.0, 0.0],
    }
