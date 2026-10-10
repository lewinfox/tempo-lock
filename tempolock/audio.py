"""Audio I/O helpers: decode anything libsndfile/ffmpeg can read, encode MP3 with tags."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)

# The rates libmp3lame will accept. Anything else is snapped to the nearest of these.
MP3_BITRATES = (32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320)
# Tags worth showing, in display order, mapped from the names ffprobe reports (lowercased).
TAG_FIELDS = ("title", "artist", "album", "album_artist", "date", "genre", "track")
LOSSLESS_CODECS = {
    "flac",
    "alac",
    "pcm_s16le",
    "pcm_s24le",
    "pcm_s32le",
    "pcm_f32le",
    "pcm_f64le",
    "pcm_s16be",
    "pcm_s24be",
}


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
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-f",
            "wav",
            "-acodec",
            "pcm_f32le",
            str(tmp),
        ],
        check=True,
    )
    y, sr = sf.read(str(tmp), always_2d=True, dtype="float32")
    tmp.unlink(missing_ok=True)
    return y, int(sr)


def probe(path: str | Path) -> dict:
    """What ffprobe knows about a file: its tags plus the technical shape of the audio.

    Everything here is best-effort - a WAV off a DAW has no tags at all, and an MP3 with a
    broken header may report no bitrate. Callers get {} rather than an exception."""
    if not shutil.which("ffprobe"):
        return {}
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                "-select_streams",
                "a:0",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
        data = json.loads(out)
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        log.warning("ffprobe failed for %s", path, exc_info=True)
        return {}

    fmt = data.get("format") or {}
    stream = (data.get("streams") or [{}])[0]
    # ID3 keys arrive in whatever case the tagger wrote them; the stream carries tags too
    # for some containers (m4a), so merge with the container's taking precedence.
    tags = {k.lower(): v for k, v in (stream.get("tags") or {}).items()}
    tags.update({k.lower(): v for k, v in (fmt.get("tags") or {}).items()})

    def num(*candidates):
        for c in candidates:
            try:
                return float(c)
            except (TypeError, ValueError):
                continue
        return None

    bitrate = num(stream.get("bit_rate"), fmt.get("bit_rate"))
    return {
        "tags": {k: str(tags[k]).strip() for k in TAG_FIELDS if tags.get(k)},
        "codec": stream.get("codec_name") or "",
        "container": (fmt.get("format_name") or "").split(",")[0],
        "bitrate_kbps": round(bitrate / 1000) if bitrate else None,
        "sample_rate": int(num(stream.get("sample_rate")) or 0) or None,
        "channels": stream.get("channels"),
        "duration": num(stream.get("duration"), fmt.get("duration")),
        "lossless": (stream.get("codec_name") or "") in LOSSLESS_CODECS,
    }


def mp3_bitrate_for(info: dict | None, default: int = 320) -> int:
    """Bitrate to re-encode at so the output is no worse than the input.

    A lossy source is matched to its own (average) rate snapped to a rate lame accepts; a
    lossless one has no meaningful MP3 equivalent, so it gets the default."""
    if not info or info.get("lossless") or not info.get("bitrate_kbps"):
        return default
    want = info["bitrate_kbps"]
    return min(MP3_BITRATES, key=lambda b: (abs(b - want), b))


def to_mono(y: np.ndarray) -> np.ndarray:
    return y.mean(axis=1) if y.ndim == 2 else y


def write_wav(path: str | Path, y: np.ndarray, sr: int) -> None:
    sf.write(str(path), y, sr, subtype="FLOAT")


def straightened(name: str, bpm: float) -> str:
    """The title and filename stem of a straightened track, so it can't be mistaken for the
    original in a music library or a downloads folder."""
    return f"{name} (straightened {bpm:g} bpm)"


def write_mp3(
    path: str | Path,
    y: np.ndarray,
    sr: int,
    bitrate_kbps: int | None = None,
    copy_tags_from: str | Path | None = None,
    bpm: float | None = None,
    title: str | None = None,
) -> None:
    """Encode MP3. Prefers ffmpeg/libmp3lame (CBR, tag copy, TBPM and title tags); falls back
    to libsndfile, which writes no tags.

    bitrate_kbps defaults to matching the source read from `copy_tags_from`, so a 128 kbps
    input does not come back as a 320 kbps file that is 2.5x the size and no better."""
    if bitrate_kbps is None:
        bitrate_kbps = mp3_bitrate_for(
            probe(copy_tags_from) if copy_tags_from else None
        )
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
        if title:
            cmd += ["-metadata", f"title={title}"]
        cmd += [str(path)]
        subprocess.run(cmd, check=True)
        tmp.unlink(missing_ok=True)
        return
    sf.write(str(path), y, sr, format="MP3")
