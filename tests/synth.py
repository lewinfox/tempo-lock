"""Synthesise a 'live drummer': kick/snare/hat pattern whose tempo drifts, with known beat times."""
import numpy as np


def live_drums(duration=30.0, sr=44100, base_bpm=120.0, drift_bpm=5.0, seed=0, lead_in=0.5, step_at=None, step_bpm=0.0):
    """step_at/step_bpm add a deliberate tempo change: from step_at seconds on, base tempo is base_bpm + step_bpm."""
    rng = np.random.default_rng(seed)

    def tempo_at(t):
        base = base_bpm + (step_bpm if step_at is not None and t >= step_at else 0.0)
        return base + drift_bpm * np.sin(2 * np.pi * t / 25) + 0.4 * drift_bpm * np.sin(2 * np.pi * t / 7.3)

    beats = [lead_in]
    t = lead_in
    while True:
        t += 60.0 / tempo_at(t) + rng.normal(0, 0.003)
        if t > duration - 0.5:
            break
        beats.append(t)
    beats = np.array(beats)
    n = int(duration * sr)
    y = np.zeros(n)

    def env(length, decay):
        return np.exp(-np.arange(length) / (decay * sr))

    def kick():
        L = int(0.25 * sr)
        tt = np.arange(L) / sr
        f = 150 * np.exp(-tt * 30) + 45
        return np.sin(2 * np.pi * np.cumsum(f) / sr) * env(L, 0.08) * 0.9

    def snare():
        L = int(0.2 * sr)
        return rng.normal(0, 1, L) * env(L, 0.05) * 0.5 + np.sin(2 * np.pi * 190 * np.arange(L) / sr) * env(L, 0.03) * 0.4

    def hat():
        L = int(0.06 * sr)
        return rng.normal(0, 1, L) * env(L, 0.012) * 0.25

    def add(sig, at):
        i = int(at * sr)
        L = min(len(sig), n - i)
        if L > 0:
            y[i:i + L] += sig[:L]

    for k, b in enumerate(beats):
        add(kick() if k % 2 == 0 else snare(), b)
        add(hat(), b)
        if k + 1 < len(beats):
            add(hat(), (b + beats[k + 1]) / 2)
        if k % 4 == 0:
            L = int(0.4 * sr)
            add(np.sin(2 * np.pi * 55 * np.arange(L) / sr) * env(L, 0.2) * 0.4, b)
    y = np.clip(y, -1, 1).astype(np.float32)
    return np.stack([y, y], axis=1), sr, beats
