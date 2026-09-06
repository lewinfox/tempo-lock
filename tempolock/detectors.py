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
# cold -> loading -> ready | failed. Read without the lock: a plain string assignment is
# atomic, and callers only use this to decide what to display.
_beat_this_state = "cold"


def beat_this_state() -> str:
    """Where the model load has got to, for /api/health."""
    if not beat_this_available():
        return "unavailable"
    return _beat_this_state


def beat_this_available() -> bool:
    try:
        import beat_this  # noqa: F401
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure means unavailable  # pragma: no cover
        return False
    return True


def _get_beat_this(checkpoint: str = "final0", dbn: bool = False):
    """Load (once) and cache the Beat This! model. Loading takes several seconds."""
    global _beat_this_model, _beat_this_state
    with _beat_this_lock:
        if _beat_this_model is None:
            # Flip the state before the imports: `import torch` is itself seconds of the
            # wait, and reporting "cold" through it makes the page look idle.
            _beat_this_state = "loading"
            try:
                import torch
                from beat_this.inference import Audio2Beats

                device = "cuda" if torch.cuda.is_available() else "cpu"
                log.info("loading Beat This! checkpoint %s on %s", checkpoint, device)
                _beat_this_model = Audio2Beats(
                    checkpoint_path=checkpoint, device=device, dbn=dbn
                )
            except Exception:
                _beat_this_state = "failed"
                raise
            _beat_this_state = "ready"
        return _beat_this_model


def warm_beat_this() -> None:
    """Load the model ahead of the first analysis, which otherwise pays ~7 s for it while
    the user waits. Idempotent, and safe to call from a background thread: a real analysis
    arriving mid-load simply blocks on the same lock rather than loading a second copy."""
    if not beat_this_available():
        return
    try:
        _get_beat_this()
    except Exception:
        log.exception(
            "pre-loading Beat This! failed; the first analysis will try again"
        )


def detect_beat_this(mono: np.ndarray, sr: int, dbn: bool = False) -> RawBeats:
    model = _get_beat_this(dbn=dbn)
    with _beat_this_lock:
        beats, downbeats = model(mono.astype(np.float32), sr)
    return RawBeats(np.asarray(beats, float), np.asarray(downbeats, float), "beat_this")


def detect_librosa(mono: np.ndarray, sr: int) -> RawBeats:
    """Fallback tracker. tightness is lowered from librosa's default of 100 so the DP is
    allowed to follow a drifting tempo instead of forcing a fixed one."""
    import librosa

    _tempo, beats = librosa.beat.beat_track(
        y=mono, sr=sr, hop_length=256, tightness=40, units="time", trim=False
    )
    return RawBeats(np.asarray(beats, float), np.array([], float), "librosa")


def detect(mono: np.ndarray, sr: int, backend: str = "auto") -> RawBeats:
    if backend == "auto":
        backend = "beat_this" if beat_this_available() else "librosa"
    if backend == "beat_this":
        return detect_beat_this(mono, sr)
    if backend == "librosa":
        return detect_librosa(mono, sr)
    raise ValueError(f"unknown detector backend {backend!r}")
