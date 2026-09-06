"""The Beat This! checkpoint takes ~7 s to load. It should be loaded before the first
upload arrives, not while someone is waiting on it."""

import importlib

import pytest
from fastapi.testclient import TestClient

from tempolock import detectors


@pytest.fixture()
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("TEMPOLOCK_DATA", str(tmp_path))
    monkeypatch.setenv("TEMPOLOCK_WARM_MODEL", "0")  # each test starts the load itself
    from tempolock import server

    return importlib.reload(server)


needs_model = pytest.mark.skipif(
    not detectors.beat_this_available(), reason="Beat This! not installed"
)


def test_health_reports_where_the_model_load_has_got_to(server):
    with TestClient(server.app) as c:
        state = c.get("/api/health").json()["model"]
    assert state in ("cold", "loading", "ready", "failed", "unavailable")


@needs_model
def test_warm_endpoint_loads_the_model_and_is_idempotent(server):
    with TestClient(server.app) as c:
        assert c.post("/api/warm").json()["model"] in ("cold", "loading", "ready")
        detectors.warm_beat_this()  # blocks until the background load is done
        assert c.get("/api/health").json()["model"] == "ready"
        # a second call must reuse the loaded model rather than build another
        first = detectors._get_beat_this()
        assert c.post("/api/warm").json()["model"] == "ready"
        assert detectors._get_beat_this() is first


@needs_model
def test_startup_warms_the_model_when_enabled(server, monkeypatch):
    monkeypatch.setenv("TEMPOLOCK_WARM_MODEL", "1")
    server = importlib.reload(server)
    monkeypatch.setattr(detectors, "_beat_this_model", None)
    monkeypatch.setattr(detectors, "_beat_this_state", "cold")
    with TestClient(server.app):
        detectors.warm_beat_this()  # joins whatever the startup thread began
        assert detectors.beat_this_state() == "ready"
