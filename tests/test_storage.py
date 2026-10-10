"""Each upload is a job folder that survives restarts until it is a day old or the disk
runs low on space."""

import json
import os
import time
from pathlib import Path
from urllib.parse import unquote

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
    monkeypatch.setenv(
        "TEMPOLOCK_WARM_MODEL", "0"
    )  # these tests are not about the model
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
    return sorted(str(p.relative_to(data)) for p in data.rglob("*") if p.is_file())


def _restart(tmp_path, monkeypatch):
    """A fresh server process on the same data dir."""
    monkeypatch.setenv("TEMPOLOCK_DATA", str(tmp_path))
    monkeypatch.setenv("TEMPOLOCK_WARM_MODEL", "0")
    import importlib

    from tempolock import server

    server = importlib.reload(server)
    c = TestClient(server.app)
    c.server = server
    return c


def _measured(client, tid):
    for _ in range(600):
        a = client.get(f"/api/tracks/{tid}").json()["rendered"]["analysis"]
        if a:
            return a
        time.sleep(0.5)


def test_each_job_gets_its_own_folder(client, wav_bytes, tmp_path):
    tid, _ = _upload(client, wav_bytes)
    assert _files(tmp_path) == [
        f"{tid}/analysis.json",
        f"{tid}/job.json",
        f"{tid}/original.wav",
        f"{tid}/upload.wav",
    ]


@needs_rubberband
def test_download_keeps_every_file(client, wav_bytes, tmp_path):
    tid, state = _upload(client, wav_bytes)
    _render(client, tid, round(state["analysis"]["median_bpm"]))
    _measured(client, tid)
    before = _files(tmp_path)
    r = client.get(f"/api/tracks/{tid}/download")
    assert r.status_code == 200 and r.content[:2] in (b"ID", b"\xff\xfb")
    assert _files(tmp_path) == before
    assert client.get(f"/api/tracks/{tid}/audio/original").status_code == 200
    assert client.get(f"/api/tracks/{tid}/audio/rendered").status_code == 200


@needs_rubberband
def test_rendered_track_is_reanalysed_and_saved(client, wav_bytes, tmp_path):
    tid, state = _upload(client, wav_bytes)
    bpm = round(state["analysis"]["median_bpm"])
    _render(client, tid, bpm)
    a = _measured(client, tid)
    assert "error" not in a, a
    # how straight it is gets checked in test_end_to_end; a 10 s clip is too short for that
    assert abs(a["median_bpm"] - bpm) < 1
    saved = json.loads((tmp_path / tid / "render.json").read_text())
    assert saved["analysis"]["median_bpm"] == a["median_bpm"]
    assert saved["params"] == {"target_bpm": bpm, "engine": "r3", "level": 1.0}


@needs_rubberband
def test_rerender_reloads_samples_dropped_from_ram(client, wav_bytes):
    tid, state = _upload(client, wav_bytes)
    client.server.JOBS[tid].y = None
    _render(client, tid, round(state["analysis"]["median_bpm"]))


def test_job_survives_a_restart(client, wav_bytes, tmp_path, monkeypatch):
    """The bug that started this: a restart wiped a track the user had not downloaded yet."""
    tid, state = _upload(client, wav_bytes)
    with _restart(tmp_path, monkeypatch) as c:
        again = c.get(f"/api/tracks/{tid}").json()
        assert again["status"] == "ready"
        assert again["analysis"]["beats"] == state["analysis"]["beats"]
        assert again["name"] == "drums.wav"
        assert c.get(f"/api/tracks/{tid}/audio/original").status_code == 200
        assert [j["id"] for j in c.get("/api/tracks").json()["jobs"]] == [tid]


@needs_rubberband
def test_old_job_can_be_rerun_after_a_restart(client, wav_bytes, tmp_path, monkeypatch):
    tid, state = _upload(client, wav_bytes)
    bpm = round(state["analysis"]["median_bpm"])
    _render(client, tid, bpm)
    _measured(client, tid)
    with _restart(tmp_path, monkeypatch) as c:
        again = c.get(f"/api/tracks/{tid}").json()
        assert again["status"] == "rendered"
        assert again["rendered"]["bpm"] == bpm and again["rendered"]["analysis"]
        r = c.get(f"/api/tracks/{tid}/download")
        assert r.status_code == 200
        assert f"drums (straightened {bpm} bpm).mp3" in unquote(r.headers["content-disposition"])
        _render(c, tid, bpm + 1)
        assert c.get("/api/tracks").json()["jobs"][0]["render"]["target_bpm"] == bpm + 1


@needs_rubberband
def test_redetecting_drops_the_old_render(client, wav_bytes, tmp_path):
    tid, state = _upload(client, wav_bytes)
    _render(client, tid, round(state["analysis"]["median_bpm"]))
    _measured(client, tid)
    assert client.post(f"/api/tracks/{tid}/analyse?detector=librosa").status_code == 200
    for _ in range(600):
        state = client.get(f"/api/tracks/{tid}").json()
        if state["status"] != "analysing":
            break
        time.sleep(0.5)
    assert state["status"] == "ready" and "rendered" not in state
    assert state["analysis"]["detector"] == "librosa"
    assert not (tmp_path / tid / "rendered.mp3").exists()


def _age(path: Path, seconds):
    t = time.time() - seconds
    for p in [path, *path.rglob("*")]:
        os.utime(p, (t, t))


def test_jobs_older_than_max_age_are_deleted(client, wav_bytes, tmp_path):
    tid, _ = _upload(client, wav_bytes)
    (tmp_path / "lost+found").mkdir()
    _age(tmp_path / "lost+found", 25 * 3600)
    _age(tmp_path / tid, 25 * 3600)
    client.server._sweep()
    assert not (tmp_path / tid).exists()
    assert (tmp_path / "lost+found").exists()  # not ours, never touched
    assert client.get(f"/api/tracks/{tid}").status_code == 404


def test_low_disk_deletes_oldest_first(client, wav_bytes, tmp_path, monkeypatch):
    first, _ = _upload(client, wav_bytes)
    second, _ = _upload(client, wav_bytes)
    _age(tmp_path / first, 3600)
    _age(tmp_path / second, 1800)
    # low on the first check, fine once one job has gone
    readings = iter([0.05])
    monkeypatch.setattr(client.server, "_free_fraction", lambda: next(readings, 0.5))
    assert client.server._sweep() > 0
    assert client.get(f"/api/tracks/{first}").status_code == 404
    assert client.get(f"/api/tracks/{second}").status_code == 200


def test_job_list_and_file_download(client, wav_bytes):
    tid, _ = _upload(client, wav_bytes)
    listing = client.get("/api/tracks").json()
    [job] = listing["jobs"]
    assert job["id"] == tid and job["name"] == "drums.wav"
    assert job["median_bpm"] and job["render"] is None
    kinds = {f["kind"]: f for f in job["files"]}
    assert set(kinds) == {"upload", "playback WAV", "beat data"}
    r = client.get(kinds["upload"]["url"])
    assert r.status_code == 200 and r.content == wav_bytes
    assert 'filename="drums.wav"' in r.headers["content-disposition"]
    assert listing["storage"]["disk_total_bytes"] > 0


def test_file_download_rejects_other_paths(client, wav_bytes, tmp_path):
    tid, _ = _upload(client, wav_bytes)
    (tmp_path / "secret.txt").write_text("no")
    assert client.get(f"/api/tracks/{tid}/files/job.json").status_code == 404
    assert client.get(f"/api/tracks/{tid}/files/..%2Fsecret.txt").status_code == 404
    assert client.get("/api/tracks/..%2F..%2Fetc/files/passwd").status_code == 404


def test_delete_removes_the_folder(client, wav_bytes, tmp_path):
    tid, _ = _upload(client, wav_bytes)
    assert client.delete(f"/api/tracks/{tid}").status_code == 200
    assert not (tmp_path / tid).exists()
    assert client.get(f"/api/tracks/{tid}").status_code == 404


def test_oversized_upload_is_rejected_and_leaves_nothing_behind(client, tmp_path):
    client.server.MAX_UPLOAD_BYTES = 4096
    r = client.post(
        "/api/tracks", files={"file": ("big.wav", b"\0" * 200_000, "audio/wav")}
    )
    assert r.status_code == 413
    assert _files(tmp_path) == []
