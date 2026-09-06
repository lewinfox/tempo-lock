"""FastAPI app: upload a track, analyse it, render it straightened, stream both back."""

from __future__ import annotations

import logging
import os
import re
import shutil
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.background import BackgroundTask

from . import audio
from .analysis import Analysis, analyse, waveform_peaks
from .detectors import beat_this_available, beat_this_state, warm_beat_this
from .render import render, rubberband_binary

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("TEMPOLOCK_DATA", ROOT / "data"))
DATA.mkdir(parents=True, exist_ok=True)


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except ValueError:
        return default


# Storage is the scarce resource in the cloud: one 5-minute track costs ~120 MB of
# WAVs before it is downloaded. Three mechanisms keep the volume from filling up.
#   1. delete-on-download - the point of the app is the rendered MP3, so once the
#      browser has it the whole track is dead weight (the page holds its own decoded
#      copy in a Web Audio buffer, so playback survives the files going away).
#   2. an idle reaper - catches uploads that are analysed and then abandoned.
#   3. a hard cap on bytes in DATA - evicts oldest-first when the two above lose.
DELETE_AFTER_DOWNLOAD = os.environ.get("TEMPOLOCK_DELETE_AFTER_DOWNLOAD", "1") != "0"
TTL_SECONDS = _env_int("TEMPOLOCK_TTL_SECONDS", 3600)
SWEEP_SECONDS = _env_int("TEMPOLOCK_SWEEP_SECONDS", 120)
MAX_DATA_BYTES = _env_int("TEMPOLOCK_MAX_DATA_BYTES", 2 * 1024**3)
MAX_UPLOAD_BYTES = _env_int("TEMPOLOCK_MAX_UPLOAD_BYTES", 150 * 1024**2)

# Loading the Beat This! checkpoint takes ~7 s. Doing it at boot means the first upload
# does not wait for it - and on fly.io the machine wakes on the request that serves the
# page, so boot and page load are the same moment anyway.
WARM_MODEL = os.environ.get("TEMPOLOCK_WARM_MODEL", "1") != "0"


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # The job table lives in memory, so anything already on the volume is unreachable:
    # a restarted machine starts from a clean data dir.
    freed = _sweep_orphans(grace_seconds=0)
    if freed:
        log.info("startup sweep freed %d bytes left by a previous run", freed)
    threading.Thread(target=_reaper_loop, daemon=True).start()
    if WARM_MODEL:
        threading.Thread(target=warm_beat_this, daemon=True, name="warm-model").start()
    yield


app = FastAPI(title="tempo-lock", lifespan=_lifespan)


class Track:
    def __init__(self, id: str, name: str, path: Path):
        self.id = id
        self.name = name
        self.path = path
        self.status = "analysing"  # analysing | ready | rendering | rendered | error
        self.error: str | None = None
        self.y: np.ndarray | None = None
        self.sr: int = 0
        self.analysis: Analysis | None = None
        self.peaks: dict | None = None
        self.rendered: dict | None = None  # {bpm, peaks, grid, mp3, wav}
        self.info: dict = {}  # ffprobe: tags plus codec/bitrate/sample rate/channels
        self.playback_wav: Path | None = None
        self.downloaded = False
        self.touched = time.monotonic()
        self.lock = threading.Lock()

    def public(self) -> dict:
        d = {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "error": self.error,
            "info": self.info,
        }
        if self.analysis is not None:
            d["analysis"] = self.analysis.to_dict()
            d["peaks"] = self.peaks
        if self.rendered is not None:
            d["rendered"] = {
                k: v
                for k, v in self.rendered.items()
                if k in ("bpm", "peaks", "grid", "duration", "bitrate_kbps")
            }
        return d

    def files(self) -> list[Path]:
        """Every file this track owns. Everything it writes - the upload, the decoded
        playback WAV, the render, and ffmpeg's temporaries - is named `{id}...`, so the
        prefix claims work in flight that has not been recorded on the track yet."""
        return sorted(DATA.glob(f"{self.id}*"))

    def discard_files(self, keep_mp3: bool = False) -> int:
        """Delete this track's files and return the bytes freed. The in-memory record
        survives, so the user can still re-render from the samples we already decoded.
        `keep_mp3` spares the rendered MP3 - it is a few MB against ~100 MB of WAVs, and
        keeping it means a second click on the download link still works."""
        keep = (
            {self.rendered["mp3"]}
            if keep_mp3 and self.rendered and self.rendered.get("mp3")
            else set()
        )
        freed = 0
        for p in self.files():
            if p not in keep:
                freed += _unlink(p)
        if self.rendered is not None:
            self.rendered.pop("wav", None)
            if not keep_mp3:
                self.rendered.pop("mp3", None)
        return freed


TRACKS: dict[str, Track] = {}


def _unlink(path: Path) -> int:
    try:
        size = path.stat().st_size
        path.unlink()
        return size
    except OSError:
        return 0


def _data_bytes() -> int:
    return sum(p.stat().st_size for p in DATA.rglob("*") if p.is_file())


def _forget(track: Track) -> int:
    """Drop a track entirely: files off the volume, samples out of RAM."""
    freed = track.discard_files()
    TRACKS.pop(track.id, None)
    track.y = None
    track.rendered = None
    return freed


def _sweep_orphans(grace_seconds: float) -> int:
    """Delete files in DATA that no live track claims. The job table is in memory, so
    anything left behind by a previous process is unreachable and must go."""
    live = tuple(TRACKS)
    cutoff = time.time() - grace_seconds
    freed = 0
    for p in DATA.rglob("*"):
        if not p.is_file() or p.name.startswith(live):
            continue
        try:
            if p.stat().st_mtime > cutoff:  # may still be being written
                continue
        except OSError:
            continue
        freed += _unlink(p)
    return freed


def _busy(track: Track) -> bool:
    return track.status in ("analysing", "rendering")


def _enforce_cap() -> int:
    """Evict least-recently-touched idle tracks until DATA is back under the cap."""
    used = _data_bytes()
    if used <= MAX_DATA_BYTES:
        return 0
    freed = 0
    for track in sorted(TRACKS.values(), key=lambda t: t.touched):
        if _busy(track):
            continue
        freed += _forget(track)
        log.warning(
            "evicted track %s to stay under the %d byte cap", track.id, MAX_DATA_BYTES
        )
        if used - freed <= MAX_DATA_BYTES:
            break
    return freed


def _reap() -> None:
    now = time.monotonic()
    for track in list(TRACKS.values()):
        if _busy(track) or now - track.touched < TTL_SECONDS:
            continue
        _forget(track)
        log.info("reaped idle track %s", track.id)
    _sweep_orphans(grace_seconds=300)
    _enforce_cap()


def _reaper_loop() -> None:
    while True:
        time.sleep(SWEEP_SECONDS)
        try:
            _reap()
        except Exception:  # pragma: no cover - a sweep failure must not kill the thread
            log.exception("storage sweep failed")


def _get(tid: str) -> Track:
    track = TRACKS.get(tid)
    if not track:
        raise HTTPException(404)
    track.touched = time.monotonic()
    return track


def _analyse_job(track: Track, backend: str):
    try:
        y, sr = audio.load(track.path)
        track.y, track.sr = y, sr
        track.peaks = waveform_peaks(y)
        # the browser plays a server-decoded PCM copy so its timeline matches ours exactly
        # (MP3 decoders disagree about encoder delay by ~25 ms, enough to misplace a grid)
        track.playback_wav = track.path.with_name(f"{track.id}_original.wav")
        sf.write(str(track.playback_wav), y, sr, subtype="PCM_16")
        track.analysis = analyse(y, sr, backend=backend)
        track.status = "ready"
    except Exception as e:  # pragma: no cover
        log.exception("analysis failed")
        track.status, track.error = "error", str(e)
    finally:
        track.touched = time.monotonic()


class RenderRequest(BaseModel):
    target_bpm: float
    engine: str = "r3"
    level: float = 1.0  # 1 = detected beats are the beat; 2 = half-time detected; 0.5 = double-time


def _render_job(track: Track, req: RenderRequest):
    try:
        assert track.y is not None and track.analysis is not None
        z, grid = render(
            track.y,
            track.sr,
            track.analysis,
            req.target_bpm,
            engine=req.engine,
            level=req.level,
        )
        wav = track.path.with_name(f"{track.id}_rendered.wav")
        mp3 = track.path.with_name(f"{track.id}_rendered.mp3")
        sf.write(str(wav), z, track.sr, subtype="PCM_16")
        bitrate = audio.mp3_bitrate_for(track.info)
        audio.write_mp3(
            mp3,
            z,
            track.sr,
            bitrate_kbps=bitrate,
            copy_tags_from=track.path,
            bpm=req.target_bpm,
        )
        track.rendered = {
            "bpm": req.target_bpm,
            "peaks": waveform_peaks(z),
            "grid": grid.to_dict(),
            "duration": len(z) / track.sr,
            "wav": wav,
            "mp3": mp3,
            "bitrate_kbps": bitrate,
        }
        track.downloaded = False
        track.status = "rendered"
    except Exception as e:
        log.exception("render failed")
        track.status, track.error = "error", str(e)
    finally:
        track.touched = time.monotonic()


@app.get("/api/health")
def health():
    return {
        "beat_this": beat_this_available(),
        "rubberband": rubberband_binary(),
        "ffmpeg": bool(shutil.which("ffmpeg")),
        "model": beat_this_state(),
        "storage": {
            "used_bytes": _data_bytes(),
            "max_bytes": MAX_DATA_BYTES,
            "tracks": len(TRACKS),
        },
    }


@app.post("/api/warm")
def warm():
    """Start the model load if it has not started. The page calls this so the wait happens
    while the user is still choosing a file rather than after they have dropped one."""
    if beat_this_state() in ("cold", "failed"):
        threading.Thread(target=warm_beat_this, daemon=True, name="warm-model").start()
    return {"model": beat_this_state()}


_TOO_BIG = "upload larger than {} MB"


@app.middleware("http")
async def _reject_oversized_bodies(request: Request, call_next):
    """Starlette spools a multipart body to a temp file before the route ever runs, so a
    size check inside the handler is too late to keep the bytes off the disk. This runs
    first and turns an oversized request away on its declared length alone."""
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
        return JSONResponse(
            {"detail": _TOO_BIG.format(MAX_UPLOAD_BYTES // 1024**2)}, status_code=413
        )
    return await call_next(request)


# Deliberately sync: FastAPI runs a `def` endpoint in a threadpool, so the file write
# cannot block the event loop for the length of the upload. Writing from a coroutine
# would stall every other request, including the polling that keeps the machine awake.
@app.post("/api/tracks")
def upload(file: UploadFile = File(...), detector: str = "auto"):
    # Backstop for a chunked request that declared no length for the middleware to see.
    if (file.size or 0) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, _TOO_BIG.format(MAX_UPLOAD_BYTES // 1024**2))
    _enforce_cap()
    tid = uuid.uuid4().hex[:12]
    suffix = Path(file.filename or "track.mp3").suffix.lower() or ".mp3"
    dest = DATA / f"{tid}{suffix}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    track = Track(tid, file.filename or dest.name, dest)
    track.info = audio.probe(dest)
    TRACKS[tid] = track
    threading.Thread(target=_analyse_job, args=(track, detector), daemon=True).start()
    return {"id": tid, "status": track.status}


@app.get("/api/tracks/{tid}")
def get_track(tid: str):
    return JSONResponse(_get(tid).public())


@app.delete("/api/tracks/{tid}")
def delete_track(tid: str):
    return {"freed_bytes": _forget(_get(tid))}


@app.post("/api/tracks/{tid}/render")
def start_render(tid: str, req: RenderRequest):
    track = _get(tid)
    if track.analysis is None:
        raise HTTPException(409, "analysis not finished")
    if track.y is None:
        raise HTTPException(410, "track expired; upload it again")
    if rubberband_binary() is None:
        raise HTTPException(500, "rubberband CLI not installed on the server")
    if not (20 <= req.target_bpm <= 400):
        raise HTTPException(422, "target_bpm out of range")
    if req.level not in (0.5, 1.0, 2.0):
        raise HTTPException(422, "level must be 0.5, 1 or 2")
    detected = track.analysis.median_bpm / req.level
    if detected and abs(req.target_bpm / detected - 1) > 0.3:
        raise HTTPException(
            422,
            f"target {req.target_bpm:g} BPM is more than 30% away from the detected {detected:.1f} BPM; "
            "that would change the speed of the track rather than straighten it. If the detector "
            "locked onto half- or double-time, change the beat level instead.",
        )
    with track.lock:
        if track.status == "rendering":
            raise HTTPException(409, "already rendering")
        track.status, track.error = "rendering", None
    threading.Thread(target=_render_job, args=(track, req), daemon=True).start()
    return {"id": tid, "status": track.status}


@app.get("/api/tracks/{tid}/audio/{which}")
def get_audio(tid: str, which: str):
    track = _get(tid)
    if which == "original":
        path = track.playback_wav or track.path
        if not path.exists():
            raise HTTPException(410, "track files have been cleaned up")
        return FileResponse(path, media_type="audio/wav")
    if which == "rendered" and track.rendered and track.rendered.get("wav"):
        return FileResponse(track.rendered["wav"], media_type="audio/wav")
    raise HTTPException(404)


def _download_stem(track: Track) -> str:
    """Name the download after the tags when the file carries them, the uploaded filename
    otherwise. Slashes and control characters would break the Content-Disposition header."""
    tags = track.info.get("tags") or {}
    stem = (
        " - ".join(x for x in (tags.get("artist"), tags.get("title")) if x)
        or Path(track.name).stem
    )
    return re.sub(r'[\x00-\x1f/\\:*?"<>|]', "_", stem).strip() or "track"


@app.get("/api/tracks/{tid}/download")
def download(tid: str):
    track = _get(tid)
    if not track.rendered or not track.rendered.get("mp3"):
        raise HTTPException(404)
    stem = _download_stem(track)
    bpm = track.rendered["bpm"]

    def cleanup() -> None:
        # Runs once the MP3 has actually gone out over the wire. The page keeps its own
        # decoded copies of both waveforms, so it carries on working; a re-render just
        # regenerates the WAVs from the samples still in RAM.
        track.downloaded = True
        if DELETE_AFTER_DOWNLOAD:
            freed = track.discard_files(keep_mp3=True)
            log.info("track %s downloaded, freed %d bytes", track.id, freed)

    return FileResponse(
        track.rendered["mp3"],
        media_type="audio/mpeg",
        filename=f"{stem} [{bpm:g} BPM].mp3",
        background=BackgroundTask(cleanup),
    )


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
