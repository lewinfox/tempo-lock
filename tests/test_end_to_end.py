"""Straighten a synthetic drifting-tempo track and check the output really is on a grid."""

import numpy as np
import pytest
from synth import live_drums

from tempolock.analysis import analyse
from tempolock.render import build_grid, render, rubberband_binary, write_timemap


@pytest.fixture(scope="module")
def track():
    return live_drums(duration=30.0, drift_bpm=5.0)


@pytest.fixture(scope="module")
def analysis(track):
    y, sr, _ = track
    return analyse(y, sr, backend="auto")


def test_detection_matches_ground_truth(track, analysis):
    _, _, truth = track
    err = np.array([np.abs(analysis.beats - b).min() for b in truth])
    assert (err < 0.05).mean() > 0.95, (
        f"only {(err < 0.05).mean():.0%} of beats found within 50 ms"
    )
    assert abs(analysis.median_bpm - 120) < 2
    assert analysis.max_bpm - analysis.min_bpm > 5  # the drift is really there


def test_timemap_is_strictly_increasing_without_leading_zero(track, analysis, tmp_path):
    y, sr, _ = track
    grid = build_grid(analysis, 120.0, len(y), sr)
    path = tmp_path / "map.txt"
    write_timemap(grid, sr, path)
    pairs = [tuple(map(int, line.split())) for line in path.read_text().splitlines()]
    assert pairs[0] != (0, 0)
    assert all(a < b for a, b in zip([p[0] for p in pairs], [p[0] for p in pairs][1:]))
    assert all(a < b for a, b in zip([p[1] for p in pairs], [p[1] for p in pairs][1:]))
    assert pairs[-1] == (len(y), grid.n_out)


def test_beat_level_scales_grid_period(track, analysis):
    y, sr, _ = track
    g1 = build_grid(analysis, 120.0, len(y), sr, level=1.0)
    g2 = build_grid(
        analysis, 240.0, len(y), sr, level=2.0
    )  # detector "found half-time" of a 240 track
    np.testing.assert_allclose(g1.target_beats, g2.target_beats)


@pytest.mark.skipif(rubberband_binary() is None, reason="rubberband CLI not installed")
def test_render_produces_regular_grid(track, analysis):
    y, sr, _ = track
    target = 120.0
    z, grid = render(y, sr, analysis, target)
    assert z.shape[1] == y.shape[1]
    assert abs(len(z) - grid.n_out) < sr * 0.05
    # re-detect on the output: inter-beat intervals should now be nearly constant
    again = analyse(z, sr, backend="auto")
    ibi = np.diff(again.beats) / np.diff(again.beat_index)
    period = 60.0 / target
    assert abs(np.median(ibi) - period) < 0.004
    assert np.std(ibi) < 0.006, f"IBI std {np.std(ibi) * 1000:.1f} ms"
    assert again.max_bpm - again.min_bpm < 3.0
    # the grid we claim to have produced is where the beats really are
    err = np.array([np.abs(again.beats - g).min() for g in grid.target_beats])
    assert np.median(err) < 0.006 and np.percentile(err, 95) < 0.02


def test_deliberate_tempo_change_is_flagged():
    y, sr, _ = live_drums(
        duration=40.0, drift_bpm=2.0, step_at=20.0, step_bpm=20.0, seed=4
    )
    a = analyse(y, sr, backend="auto")
    assert len(a.tempo_changes) == 1, a.tempo_changes
    c = a.tempo_changes[0]
    assert abs(c["time"] - 20.0) < 1.5
    assert abs(c["from_bpm"] - 120) < 3 and abs(c["to_bpm"] - 140) < 3
    assert len(a.sections) == 2


def test_plain_drift_is_not_flagged():
    y, sr, _ = live_drums(duration=30.0, drift_bpm=5.0)
    a = analyse(y, sr, backend="auto")
    assert a.tempo_changes == []
    assert len(a.sections) == 1
