"""The volume must not fill up: files go away when the track is downloaded, expires, or
when the data dir blows past its cap."""

import time
from pathlib import Path

import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from synth import live_drums

from tempolock.render import rubberband_binary

needs_rubberband = pytest.mark.skipif(
    rubberband_binary() is None, reason="rubberband CLI not installed"
)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TEMPOLOCK_DATA", str(tmp_path))
    import importlib

    from tempolock import server

    server = importlib.reload(server)
    with TestClient(server.app) as c:
        c.server = server
        yield c


@pytest.fixture(scope="module")
def wav_bytes(tmp_path_factory):
    y, sr, _ = live_drums(duration=10.0, drift_bpm=4.0)
    path = tmp_path_factory.mktemp("src") / "drums.wav"
    sf.write(str(path), y, sr, subtype="PCM_16")
    return path.read_bytes()


def _upload(client, wav_bytes):
    r = client.post(
        "/api/tracks", files={"file": ("drums.wav", wav_bytes, "audio/wav")}
    )
    assert r.status_code == 200
    tid = r.json()["id"]
    for _ in range(600):
        state = client.get(f"/api/tracks/{tid}").json()
        if state["status"] in ("ready", "error"):
            break
        time.sleep(0.5)
    assert state["status"] == "ready", state["error"]
    return tid, state


def _render(client, tid, bpm):
    assert (
        client.post(f"/api/tracks/{tid}/render", json={"target_bpm": bpm}).status_code
        == 200
    )
    for _ in range(600):
        state = client.get(f"/api/tracks/{tid}").json()
        if state["status"] in ("rendered", "error"):
            break
        time.sleep(0.5)
    assert state["status"] == "rendered", state["error"]


def _files(data: Path):
    return sorted(p.name for p in data.rglob("*") if p.is_file())


@needs_rubberband
def test_download_frees_the_wavs_but_keeps_the_mp3(client, wav_bytes, tmp_path):
    tid, state = _upload(client, wav_bytes)
    _render(client, tid, round(state["analysis"]["median_bpm"]))
    before = _files(tmp_path)
    assert any(f.endswith("_rendered.wav") for f in before)
    assert any(f.endswith("_original.wav") for f in before)

    r = client.get(f"/api/tracks/{tid}/download")
    assert r.status_code == 200 and r.content[:2] in (b"ID", b"\xff\xfb")

    after = _files(tmp_path)
    assert after == [f"{tid}_rendered.mp3"], after
    # a second click on the same link still works
    assert client.get(f"/api/tracks/{tid}/download").status_code == 200
    # ...but the WAVs are gone, so the player asks for them and is told so plainly
    assert client.get(f"/api/tracks/{tid}/audio/original").status_code == 410
    assert client.get(f"/api/tracks/{tid}/audio/rendered").status_code == 404


@needs_rubberband
def test_rerender_after_download_regenerates_the_files(client, wav_bytes, tmp_path):
    tid, state = _upload(client, wav_bytes)
    bpm = round(state["analysis"]["median_bpm"])
    _render(client, tid, bpm)
    client.get(f"/api/tracks/{tid}/download")
    _render(client, tid, bpm + 1)
    assert client.get(f"/api/tracks/{tid}/audio/rendered").status_code == 200


def test_idle_tracks_are_reaped(client, wav_bytes, tmp_path):
    tid, _ = _upload(client, wav_bytes)
    client.server.TTL_SECONDS = 0
    client.server._reap()
    assert _files(tmp_path) == []
    assert client.get(f"/api/tracks/{tid}").status_code == 404


def test_orphans_from_a_previous_process_are_swept(client, tmp_path):
    stale = tmp_path / "deadbeef_rendered.wav"
    stale.write_bytes(b"x" * 1024)
    assert client.server._sweep_orphans(grace_seconds=0) == 1024
    assert not stale.exists()


def test_data_cap_evicts_the_oldest_track(client, wav_bytes, tmp_path):
    first, _ = _upload(client, wav_bytes)
    second, _ = _upload(client, wav_bytes)
    client.server.TRACKS[first].touched = 0.0
    client.server.MAX_DATA_BYTES = client.server._data_bytes() - 1
    assert client.server._enforce_cap() > 0
    assert client.get(f"/api/tracks/{first}").status_code == 404
    assert client.get(f"/api/tracks/{second}").status_code == 200


def test_oversized_upload_is_rejected_and_leaves_nothing_behind(client, tmp_path):
    client.server.MAX_UPLOAD_BYTES = 4096
    r = client.post(
        "/api/tracks", files={"file": ("big.wav", b"\0" * 200_000, "audio/wav")}
    )
    assert r.status_code == 413
    assert _files(tmp_path) == []
