"""Audio I/O helpers: decode anything libsndfile/ffmpeg can read, encode MP3 with tags."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def load(path: str | Path) -> tuple[np.ndarray, int]:
    """Return float32 samples shaped (n, channels) and the sample rate."""
    path = Path(path)
    try:
        y, sr = sf.read(str(path), always_2d=True, dtype="float32")
        return y, int(sr)
    except Exception:
        if not shutil.which("ffmpeg"):
            raise
    # libsndfile could not decode it (e.g. m4a) - go through ffmpeg
    tmp = path.with_suffix(".decoded.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-f", "wav", "-acodec", "pcm_f32le", str(tmp)],
        check=True,
    )
    y, sr = sf.read(str(tmp), always_2d=True, dtype="float32")
    tmp.unlink(missing_ok=True)
    return y, int(sr)


def to_mono(y: np.ndarray) -> np.ndarray:
    return y.mean(axis=1) if y.ndim == 2 else y


def write_wav(path: str | Path, y: np.ndarray, sr: int) -> None:
    sf.write(str(path), y, sr, subtype="FLOAT")


def write_mp3(
    path: str | Path,
    y: np.ndarray,
    sr: int,
    bitrate_kbps: int = 320,
    copy_tags_from: str | Path | None = None,
    bpm: float | None = None,
) -> None:
    """Encode MP3. Prefers ffmpeg/libmp3lame (CBR, tag copy, TBPM tag); falls back to libsndfile."""
    path = Path(path)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.999:
        y = y * (0.999 / peak)
    if shutil.which("ffmpeg"):
        tmp = path.with_suffix(".tmp.wav")
        write_wav(tmp, y, sr)
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp)]
        if copy_tags_from and Path(copy_tags_from).exists():
            cmd += ["-i", str(copy_tags_from), "-map", "0:a", "-map_metadata", "1"]
        cmd += ["-c:a", "libmp3lame", "-b:a", f"{bitrate_kbps}k", "-id3v2_version", "3"]
        if bpm is not None:
            cmd += ["-metadata", f"TBPM={bpm:g}"]
        cmd += [str(path)]
        subprocess.run(cmd, check=True)
        tmp.unlink(missing_ok=True)
        return
    sf.write(str(path), y, sr, format="MP3")
