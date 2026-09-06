import numpy as np

from tempolock.analysis import clean_beats, refine_to_onsets


def _flat_env(duration=10.0, hop_s=0.005):
    times = np.arange(0, duration, hop_s)
    return np.ones_like(times), times


def test_clean_beats_fills_missed_beat_with_index_gap():
    env, times = _flat_env()
    beats = np.array([1.0, 1.5, 2.0, 3.0, 3.5])  # beat at 2.5 was missed
    kept, idx, dropped = clean_beats(beats, env, times)
    assert list(kept) == [1.0, 1.5, 2.0, 3.0, 3.5]
    assert list(idx) == [0, 1, 2, 4, 5]
    assert dropped == []


def test_clean_beats_drops_spurious_double_detection():
    env, times = _flat_env()
    beats = np.array([1.0, 1.5, 1.6, 2.0, 2.5])  # 1.6 is a phantom
    kept, idx, dropped = clean_beats(beats, env, times)
    assert list(kept) == [1.0, 1.5, 2.0, 2.5]
    assert list(idx) == [0, 1, 2, 3]
    assert dropped == [1.6]


def test_clean_beats_drops_beats_in_silence():
    times = np.arange(0, 10, 0.005)
    env = np.ones_like(times)
    env[times < 0.9] = 0.0  # silence before the music starts
    kept, _, dropped = clean_beats(np.array([0.0, 1.0, 1.5, 2.0]), env, times)
    assert list(kept) == [1.0, 1.5, 2.0]
    assert dropped == [0.0]


def test_refine_snaps_to_strongest_onset_in_window():
    times = np.arange(0, 5, 0.005)
    env = np.zeros_like(times)
    env[np.searchsorted(times, 2.013)] = 5.0
    out = refine_to_onsets(np.array([2.0]), env, times, window=0.035)
    assert abs(out[0] - 2.013) < 0.003
    # nothing within the window: unchanged
    out = refine_to_onsets(np.array([4.0]), env, times, window=0.035)
    assert out[0] == 4.0


from tempolock.analysis import segment_tempo


def _curve(bpms, jitter=0.6, seed=1):
    rng = np.random.default_rng(seed)
    bpm = np.array(bpms, float) + rng.normal(0, jitter, len(bpms))
    times = np.concatenate([[0], np.cumsum(60.0 / bpm)])
    return bpm, times


def test_segment_flags_abrupt_sustained_jump():
    bpm, times = _curve([120] * 64 + [140] * 64)
    sections, changes = segment_tempo(bpm, times)
    assert len(sections) == 2 and len(changes) == 1
    assert abs(changes[0]["beat"] - 64) <= 2
    assert abs(changes[0]["from_bpm"] - 120) < 1.5 and abs(changes[0]["to_bpm"] - 140) < 1.5
    assert changes[0]["kind"] == "tempo change"
    assert abs(sections[0]["end"] - times[changes[0]["beat"]]) < 1e-9


def test_segment_ignores_gradual_drift():
    ramp = np.linspace(116, 128, 200)  # +10% but spread over 200 beats
    bpm, times = _curve(ramp)
    sections, changes = segment_tempo(bpm, times)
    assert changes == []
    assert len(sections) == 1


def test_segment_ignores_short_fill():
    bpm, times = _curve([120] * 60 + [150, 150, 150] + [120] * 60)  # 3-beat push
    _, changes = segment_tempo(bpm, times)
    assert changes == []


def test_segment_labels_half_time():
    bpm, times = _curve([120] * 48 + [60] * 48 + [120] * 48, jitter=0.3)
    _, changes = segment_tempo(bpm, times)
    assert [c["kind"] for c in changes] == ["half-time", "double-time"]


def test_segment_handles_wobbly_but_constant_tempo():
    rng = np.random.default_rng(3)
    bpm, times = _curve(120 + 3 * np.sin(np.arange(300) / 10) + rng.normal(0, 1.0, 300))
    _, changes = segment_tempo(bpm, times)
    assert changes == []
