"""Beat / downbeat detection back-ends.

Primary: Beat This! (CPJKU, ISMIR 2024) - transformer beat tracker, robust to tempo drift,
outputs beats and downbeats. Fallback: librosa's dynamic-programming tracker, which is
always installable but assumes a near-constant tempo and knows nothing about downbeats.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class RawBeats:
    beats: np.ndarray  # seconds
    downbeats: np.ndarray  # seconds (may be empty)
    detector: str


_beat_this_lock = threading.Lock()
_beat_this_model = None


def beat_this_available() -> bool:
    try:
        import beat_this  # noqa: F401
        import torch  # noqa: F401
    except Exception:  # pragma: no cover - depends on environment
        return False
    return True


def _get_beat_this(checkpoint: str = "final0", dbn: bool = False):
    """Load (once) and cache the Beat This! model. Loading takes several seconds."""
    global _beat_this_model
    with _beat_this_lock:
        if _beat_this_model is None:
            import torch
            from beat_this.inference import Audio2Beats

            device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info("loading Beat This! checkpoint %s on %s", checkpoint, device)
            _beat_this_model = Audio2Beats(checkpoint_path=checkpoint, device=device, dbn=dbn)
        return _beat_this_model


def detect_beat_this(mono: np.ndarray, sr: int, dbn: bool = False) -> RawBeats:
    model = _get_beat_this(dbn=dbn)
    with _beat_this_lock:
        beats, downbeats = model(mono.astype(np.float32), sr)
    return RawBeats(np.asarray(beats, float), np.asarray(downbeats, float), "beat_this")


def detect_librosa(mono: np.ndarray, sr: int) -> RawBeats:
    """Fallback tracker. tightness is lowered from librosa's default of 100 so the DP is
    allowed to follow a drifting tempo instead of forcing a fixed one."""
    import librosa

    tempo, beats = librosa.beat.beat_track(y=mono, sr=sr, hop_length=256, tightness=40, units="time", trim=False)
    return RawBeats(np.asarray(beats, float), np.array([], float), "librosa")


def detect(mono: np.ndarray, sr: int, backend: str = "auto") -> RawBeats:
    if backend == "auto":
        backend = "beat_this" if beat_this_available() else "librosa"
    if backend == "beat_this":
        return detect_beat_this(mono, sr)
    if backend == "librosa":
        return detect_librosa(mono, sr)
    raise ValueError(f"unknown detector backend {backend!r}")
