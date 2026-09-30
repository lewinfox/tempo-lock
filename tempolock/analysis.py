"""Turn raw detector output into a clean, indexed beat list plus tempo statistics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import pairwise

import numpy as np

from .detectors import RawBeats, detect


@dataclass
class Analysis:
    sr: int
    duration: float
    detector: str
    beats: np.ndarray  # refined beat times (s)
    beat_index: (
        np.ndarray
    )  # integer beat count for each beat (accounts for missed beats)
    is_downbeat: np.ndarray  # bool per beat
    bpm_times: np.ndarray  # midpoint of each inter-beat interval (s)
    bpm_curve: np.ndarray  # instantaneous BPM per interval
    median_bpm: float
    min_bpm: float
    max_bpm: float
    suggested_bpm: float
    dropped_beats: list = field(
        default_factory=list
    )  # spurious detections we discarded (s)
    sections: list = field(
        default_factory=list
    )  # tempo sections split at abrupt jumps (see segment_tempo)
    tempo_changes: list = field(default_factory=list)  # the abrupt jumps themselves

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Analysis:
        """Inverse of to_dict, for an analysis saved to disk as JSON."""
        d = dict(d)
        d["beats"] = np.asarray(d["beats"], float)
        d["beat_index"] = np.asarray(d["beat_index"], int)
        d["is_downbeat"] = np.asarray(d["is_downbeat"], bool)
        d["bpm_times"] = np.asarray(d["bpm_times"], float)
        d["bpm_curve"] = np.asarray(d["bpm_curve"], float)
        return cls(**d)


def onset_envelope(
    mono: np.ndarray, sr: int, hop: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    import librosa

    env = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=hop)
    times = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=hop)
    return env, times


def refine_to_onsets(
    beats: np.ndarray, env: np.ndarray, times: np.ndarray, window: float = 0.035
) -> np.ndarray:
    """Beat trackers emit frame-quantised times (Beat This!: 20 ms). Snap each beat to the
    strongest onset within +/-window so the grid lands on the actual drum transient."""
    out = np.empty_like(beats)
    for i, b in enumerate(beats):
        lo, hi = np.searchsorted(times, [b - window, b + window])
        if hi <= lo:
            out[i] = b
            continue
        seg = env[lo:hi]
        k = int(np.argmax(seg))
        # only snap when there is a clear transient; a flat window would send argmax to its edge
        if seg[k] > 0 and seg[k] > 1.5 * np.median(seg) + 1e-9:
            out[i] = times[lo + k]
        else:
            out[i] = b
    return out


def clean_beats(
    beats: np.ndarray, env: np.ndarray, times: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list]:
    """Assign an integer beat index to each detection, tolerating both missed beats
    (index jumps by 2+) and spurious extra beats (dropped). Also drops detections that
    sit in near-silence, e.g. a phantom beat at t=0 before the music starts.

    Returns (beats, indices, dropped)."""
    beats = np.sort(np.asarray(beats, float))
    if len(beats) < 2:
        return beats, np.arange(len(beats)), []

    strength = np.array(
        [env[min(len(env) - 1, int(np.searchsorted(times, b)))] for b in beats]
    )
    audible = strength > 0.02 * np.max(strength)
    dropped = beats[~audible].tolist()
    beats = beats[audible]
    if len(beats) < 2:
        return beats, np.arange(len(beats)), dropped

    ibi = np.diff(beats)
    period = float(np.median(ibi))
    kept = [beats[0]]
    index = [0]
    recent = [period] * 4  # rolling local period estimate
    for b in beats[1:]:
        local = float(np.median(recent))
        gap = b - kept[-1]
        if gap < 0.6 * local:
            dropped.append(float(b))  # too soon: spurious double detection
            continue
        steps = max(1, round(gap / local))
        kept.append(b)
        index.append(index[-1] + steps)
        recent.append(gap / steps)
        recent = recent[-8:]
    return np.array(kept), np.array(index, int), dropped


def _rolling_median(x: np.ndarray, width: int) -> np.ndarray:
    if len(x) < 3:
        return x.copy()
    width = min(width, len(x) if len(x) % 2 else len(x) - 1)
    half = width // 2
    pad = np.pad(x, half, mode="edge")
    return np.array([np.median(pad[i : i + width]) for i in range(len(x))])


def segment_tempo(
    bpm: np.ndarray,
    beat_times: np.ndarray,
    min_beats: int = 8,
    jump_frac: float = 0.05,
    context: int = 8,
) -> tuple[list, list]:
    """Find tempo changes that look deliberate rather than like a drummer drifting.

    A change counts as deliberate when the tempo settles at a new value within a beat or
    two and the step between the last `context` beats and the next `context` beats is at
    least `jump_frac` of the tempo. Gradual ramps (a ritardando, a song that creeps up
    3% over four minutes) do not trip this: their step between adjacent windows is small
    even though the overall range is large.

    Returns (sections, changes). Each section is a dict with beat/time bounds and its median
    BPM; each change has the beat index, the time, the BPM before and after and the ratio.
    `bpm[i]` is the tempo of the interval between beat i and beat i+1, and `beat_times`
    has one more entry than `bpm`."""
    n = len(bpm)
    if n < 2 * min_beats:
        med = float(np.median(bpm)) if n else 0.0
        return (
            [
                {
                    "start_beat": 0,
                    "end_beat": n,
                    "start": float(beat_times[0]),
                    "end": float(beat_times[-1]),
                    "bpm": med,
                }
            ]
            if n
            else []
        ), []

    smooth = _rolling_median(bpm, 3)  # kills single-beat push/pull

    # Discontinuity score at each boundary k: fit a line to the `context` beats on each
    # side and measure the gap between the two lines at the boundary. A gradual ramp has
    # continuous lines (gap ~ 0) however steep it is; a real step shows the whole jump.
    def fit_at(seg: np.ndarray, x_eval: float) -> tuple[float, float]:
        """(value of the fitted line at x_eval, RMS residual of the fit)"""
        x = np.arange(len(seg), dtype=float)
        if len(seg) < 3:
            return float(np.median(seg)), 0.0
        slope, intercept = np.polyfit(x, seg, 1)
        resid = seg - (slope * x + intercept)
        return float(slope * x_eval + intercept), float(np.sqrt(np.mean(resid**2)))

    cuts = []
    for k in range(min_beats, n - min_beats + 1):
        before, r1 = fit_at(smooth[k - context : k], context - 0.5)
        after, r2 = fit_at(smooth[k : k + context], -0.5)
        if before <= 0 or after <= 0:
            continue
        ratio = after / before
        if abs(ratio - 1) >= jump_frac:
            # a window that straddles the real step fits badly; rank by gap over misfit so
            # the run of adjacent candidates collapses onto the true boundary
            cuts.append((k, abs(np.log(ratio)) / (r1 + r2 + 0.02 * before)))
    # every boundary within `context` beats of a real step is a candidate (one of its
    # windows straddles the step), so cluster candidates that lie within `context` of each
    # other and keep the best-scoring one per cluster; then enforce the minimum section length
    chosen: list[int] = []
    i = 0
    while i < len(cuts):
        j = i
        while j + 1 < len(cuts) and cuts[j + 1][0] - cuts[j][0] <= context:
            j += 1
        k = max(cuts[i : j + 1], key=lambda c: c[1])[0]
        if not chosen or k - chosen[-1] >= min_beats:
            chosen.append(k)
        i = j + 1

    # verify each cut with the actual section medians (a candidate can vanish once
    # neighbouring sections are known, e.g. two cuts that bracket one fill)
    bounds = [0] + chosen + [n]
    sections = []
    changes = []
    for a, b in pairwise(bounds):
        sections.append(
            {
                "start_beat": a,
                "end_beat": b,
                "start": float(beat_times[a]),
                "end": float(beat_times[b]),
                "bpm": float(np.median(bpm[a:b])),
            }
        )
    merged = [sections[0]]
    for sec in sections[1:]:
        prev = merged[-1]
        if abs(sec["bpm"] / prev["bpm"] - 1) < jump_frac:
            prev["end_beat"], prev["end"] = sec["end_beat"], sec["end"]
            prev["bpm"] = float(np.median(bpm[prev["start_beat"] : prev["end_beat"]]))
        else:
            merged.append(sec)
    for prev, sec in pairwise(merged):
        k = sec["start_beat"]
        ratio = sec["bpm"] / prev["bpm"]
        kind = (
            "half-time"
            if abs(ratio - 0.5) < 0.06
            else "double-time"
            if abs(ratio - 2) < 0.12
            else "tempo change"
        )
        changes.append(
            {
                "beat": k,
                "time": float(beat_times[k]),
                "from_bpm": prev["bpm"],
                "to_bpm": sec["bpm"],
                "ratio": float(ratio),
                "kind": kind,
            }
        )
    return merged, changes


def analyse(
    y: np.ndarray, sr: int, backend: str = "auto", raw: RawBeats | None = None
) -> Analysis:
    mono = y.mean(axis=1) if y.ndim == 2 else y
    if raw is None:
        raw = detect(mono, sr, backend=backend)
    env, times = onset_envelope(mono, sr)
    beats = refine_to_onsets(raw.beats, env, times)
    beats, index, dropped = clean_beats(beats, env, times)

    is_down = np.zeros(len(beats), bool)
    if len(raw.downbeats):
        for d in raw.downbeats:
            if len(beats) == 0:
                break
            k = int(np.argmin(np.abs(beats - d)))
            if abs(beats[k] - d) < 0.06:
                is_down[k] = True

    if len(beats) >= 2:
        ibi = np.diff(beats) / np.diff(index)
        bpm = 60.0 / ibi
        bpm_times = (beats[1:] + beats[:-1]) / 2
        median = float(np.median(bpm))
        sections, changes = segment_tempo(bpm, beats)
        return Analysis(
            sr=sr,
            duration=len(y) / sr,
            detector=raw.detector,
            beats=beats,
            beat_index=index,
            is_downbeat=is_down,
            bpm_times=bpm_times,
            bpm_curve=bpm,
            median_bpm=median,
            min_bpm=float(np.percentile(bpm, 2)),
            max_bpm=float(np.percentile(bpm, 98)),
            suggested_bpm=float(round(median)),
            dropped_beats=dropped,
            sections=sections,
            tempo_changes=changes,
        )
    return Analysis(
        sr,
        len(y) / sr,
        raw.detector,
        beats,
        index,
        is_down,
        np.array([]),
        np.array([]),
        0.0,
        0.0,
        0.0,
        0.0,
        dropped,
    )


def waveform_peaks(y: np.ndarray, buckets: int = 4000) -> dict:
    """Min/max per bucket, quantised to int8, for drawing the waveform in the browser."""
    mono = y.mean(axis=1) if y.ndim == 2 else y
    n = len(mono)
    buckets = min(buckets, n) or 1
    edges = np.linspace(0, n, buckets + 1).astype(int)
    mins = np.zeros(buckets, np.float32)
    maxs = np.zeros(buckets, np.float32)
    for i in range(buckets):
        seg = mono[edges[i] : max(edges[i] + 1, edges[i + 1])]
        mins[i] = seg.min()
        maxs[i] = seg.max()
    scale = max(1e-9, float(max(np.abs(mins).max(), np.abs(maxs).max())))
    return {
        "min": np.round(mins / scale * 127).astype(int).tolist(),
        "max": np.round(maxs / scale * 127).astype(int).tolist(),
    }
