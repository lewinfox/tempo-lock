"""FastAPI app: upload a track, analyse it, render it straightened, stream both back.

Each upload is a job with its own folder on the data volume, so jobs survive a restart
and can be reopened and re-run from the page's job browser."""

from __future__ import annotations

import json
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


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


# Job folders stay on the volume until one of two things happens, so they survive a
# restart (on fly.io the machine stops itself a minute or two after the last request):
#   1. every file in the job is older than MAX_AGE_SECONDS, or
#   2. the volume has less than MIN_FREE_FRACTION free - then whole jobs go, oldest
#      first, until it is back above the line.
# Jobs mid-analysis or mid-render are never deleted.
MAX_AGE_SECONDS = _env_int("TEMPOLOCK_MAX_AGE_SECONDS", 24 * 3600)
MIN_FREE_FRACTION = _env_float("TEMPOLOCK_MIN_FREE_FRACTION", 0.10)
# Decoded samples of a job nobody has touched for this long are dropped from RAM; a
# re-render reloads them from the upload on disk.
TTL_SECONDS = _env_int("TEMPOLOCK_TTL_SECONDS", 3600)
SWEEP_SECONDS = _env_int("TEMPOLOCK_SWEEP_SECONDS", 120)
MAX_UPLOAD_BYTES = _env_int("TEMPOLOCK_MAX_UPLOAD_BYTES", 150 * 1024**2)

# Loading the Beat This! checkpoint takes ~7 s. Doing it at boot means the first upload
# does not wait for it - and on fly.io the machine wakes on the request that serves the
# page, so boot and page load are the same moment anyway.
WARM_MODEL = os.environ.get("TEMPOLOCK_WARM_MODEL", "1") != "0"


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    _sweep()
    threading.Thread(target=_reaper_loop, daemon=True).start()
    if WARM_MODEL:
        threading.Thread(target=warm_beat_this, daemon=True, name="warm-model").start()
    yield


app = FastAPI(title="tempo-lock", lifespan=_lifespan)

_JOB_ID = re.compile(r"[0-9a-f]{12}")


def _write_json(path: Path, obj) -> None:
    """Write via a temp file and rename, so a crash never leaves half a JSON file."""
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


class Job:
    """One upload and everything made from it, kept in DATA/{id}/:

        job.json        name, ffprobe info, and a summary for the job list
        upload.<ext>    the file as uploaded
        original.wav    decoded copy the browser plays (its timeline matches ours exactly;
                        MP3 decoders disagree about encoder delay by ~25 ms)
        analysis.json   detected beats, tempo stats and waveform peaks
        render.json     the last straighten: its settings, grid, peaks, and the beats
                        detected again in the output
        rendered.wav    the straightened audio, and the same as an MP3
        rendered.mp3

    The object in memory caches that folder and adds the live status of any work running."""

    def __init__(self, id: str):
        self.id = id
        self.dir = DATA / id
        self.name = ""
        self.upload_name = ""
        self.info: dict = {}  # ffprobe: tags plus codec/bitrate/sample rate/channels
        self.created = time.time()
        self.status = "analysing"  # analysing | ready | rendering | rendered | error
        self.error: str | None = None
        self.y: np.ndarray | None = None
        self.sr: int = 0
        self.analysis: Analysis | None = None
        self.peaks: dict | None = None
        self.rendered: dict | None = None  # the contents of render.json
        self.touched = time.monotonic()
        self.lock = threading.Lock()

    upload = property(lambda self: self.dir / self.upload_name)
    playback = property(lambda self: self.dir / "original.wav")
    rendered_wav = property(lambda self: self.dir / "rendered.wav")
    rendered_mp3 = property(lambda self: self.dir / "rendered.mp3")

    @classmethod
    def load(cls, id: str) -> Job | None:
        """Rebuild a job from its folder, e.g. after a restart."""
        meta = _read_json(DATA / id / "job.json")
        if not meta:
            return None
        job = cls(id)
        job.name = meta.get("name", id)
        job.upload_name = meta.get("upload", "")
        job.info = meta.get("info") or {}
        job.created = meta.get("created", job.created)
        saved = _read_json(job.dir / "analysis.json")
        if saved:
            job.analysis = Analysis.from_dict(saved["analysis"])
            job.peaks = saved.get("peaks")
            job.status = "ready"
        else:
            job.status = "error"
            job.error = "beat detection did not finish; run it again"
        rendered = _read_json(job.dir / "render.json")
        if saved and rendered and job.rendered_mp3.exists():
            if rendered.get("analysis") is None:
                rendered["analysis"] = {
                    "error": "measuring was interrupted; straighten again to measure"
                }
            job.rendered = rendered
            job.status = "rendered"
        return job

    def save(self) -> None:
        """job.json: enough to label the job in the list without reading anything else."""
        a, r = self.analysis, self.rendered
        _write_json(
            self.dir / "job.json",
            {
                "id": self.id,
                "name": self.name,
                "upload": self.upload_name,
                "info": self.info,
                "created": self.created,
                "detector": a.detector if a else None,
                "median_bpm": a.median_bpm if a else None,
                "render": r["params"] if r else None,
            },
        )

    def public(self) -> dict:
        d = {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "error": self.error,
            "info": self.info,
            "created": self.created,
        }
        if self.analysis is not None:
            d["analysis"] = self.analysis.to_dict()
            d["peaks"] = self.peaks
        if self.rendered is not None:
            d["rendered"] = self.rendered
        return d

    def drop_render(self) -> None:
        self.rendered = None
        for p in (self.dir / "render.json", self.rendered_wav, self.rendered_mp3):
            _unlink(p)


JOBS: dict[str, Job] = {}
_jobs_lock = threading.Lock()


def _unlink(path: Path) -> int:
    try:
        size = path.stat().st_size
        path.unlink()
        return size
    except OSError:
        return 0


def _dir_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def _data_bytes() -> int:
    return sum(_dir_bytes(d) for d in _job_dirs())


def _free_fraction() -> float:
    usage = shutil.disk_usage(DATA)
    return usage.free / usage.total if usage.total else 1.0


def _job_dirs() -> list[Path]:
    """Every job folder. Anything else in DATA (lost+found on a fresh volume) is left alone."""
    return [d for d in DATA.iterdir() if d.is_dir() and _JOB_ID.fullmatch(d.name)]


def _newest_mtime(path: Path) -> float:
    times = [path.stat().st_mtime]
    for p in path.iterdir():
        try:
            times.append(p.stat().st_mtime)
        except OSError:
            pass
    return max(times)


def _busy(job: Job) -> bool:
    return job.status in ("analysing", "rendering")


def _delete_job(tid: str) -> int:
    job = JOBS.pop(tid, None)
    if job is not None:
        job.y = None
    path = DATA / tid
    freed = _dir_bytes(path)
    shutil.rmtree(path, ignore_errors=True)
    return freed


# A folder written to in the last minute may belong to an upload still streaming in,
# before its job is registered as busy. Low disk space never deletes one of those.
_WRITE_GRACE = 60


def _sweep() -> int:
    """Delete jobs whose newest file is older than MAX_AGE_SECONDS, then, while the volume
    is under MIN_FREE_FRACTION free, the oldest remaining jobs. Returns bytes freed."""
    now = time.time()
    jobs = [
        (d.name, _newest_mtime(d))
        for d in _job_dirs()
        if not (d.name in JOBS and _busy(JOBS[d.name]))
    ]
    jobs.sort(key=lambda j: j[1])
    freed = 0
    for tid, mtime in jobs:
        if now - mtime > MAX_AGE_SECONDS:
            reason = f"older than {MAX_AGE_SECONDS} s"
        elif _free_fraction() < MIN_FREE_FRACTION and now - mtime > _WRITE_GRACE:
            reason = f"disk under {MIN_FREE_FRACTION:.0%} free"
        else:
            break  # sorted oldest first, so nothing later is due either
        n = _delete_job(tid)
        freed += n
        log.info("deleted job %s (%d bytes): %s", tid, n, reason)
    return freed


def _reap() -> None:
    now = time.monotonic()
    for job in list(JOBS.values()):
        if not _busy(job) and job.y is not None and now - job.touched > TTL_SECONDS:
            job.y = None  # frees RAM only; the files stay
    _sweep()


def _reaper_loop() -> None:
    while True:
        time.sleep(SWEEP_SECONDS)
        try:
            _reap()
        except Exception:  # pragma: no cover - a sweep failure must not kill the thread
            log.exception("storage sweep failed")


def _get(tid: str) -> Job:
    if not _JOB_ID.fullmatch(tid):
        raise HTTPException(404)
    with _jobs_lock:
        job = JOBS.get(tid)
        if job is None:
            job = Job.load(tid)
            if job is None:
                raise HTTPException(404)
            JOBS[tid] = job
    job.touched = time.monotonic()
    return job


def _samples(job: Job) -> np.ndarray:
    if job.y is None:  # never loaded in this process, or dropped from RAM while idle
        job.y, job.sr = audio.load(job.upload)
    return job.y


def _analyse_job(job: Job, backend: str):
    try:
        y = _samples(job)
        peaks = waveform_peaks(y)
        if not job.playback.exists():
            sf.write(str(job.playback), y, job.sr, subtype="PCM_16")
        a = analyse(y, job.sr, backend=backend)
        _write_json(
            job.dir / "analysis.json", {"analysis": a.to_dict(), "peaks": peaks}
        )
        job.analysis, job.peaks = a, peaks
        job.save()
        job.status = "ready"
    except Exception as e:  # pragma: no cover
        log.exception("analysis failed")
        job.status, job.error = "error", str(e)
    finally:
        job.touched = time.monotonic()


class RenderRequest(BaseModel):
    target_bpm: float
    engine: str = "r3"
    level: float = 1.0  # 1 = detected beats are the beat; 2 = half-time detected; 0.5 = double-time


def _render_job(job: Job, req: RenderRequest):
    try:
        assert job.analysis is not None
        y = _samples(job)
        z, grid = render(
            y,
            job.sr,
            job.analysis,
            req.target_bpm,
            engine=req.engine,
            level=req.level,
        )
        sf.write(str(job.rendered_wav), z, job.sr, subtype="PCM_16")
        bitrate = audio.mp3_bitrate_for(job.info)
        audio.write_mp3(
            job.rendered_mp3,
            z,
            job.sr,
            bitrate_kbps=bitrate,
            copy_tags_from=job.upload,
            bpm=req.target_bpm,
            title=audio.straightened(
                (job.info.get("tags") or {}).get("title") or Path(job.name).stem,
                req.target_bpm,
            ),
        )
        rendered = {
            "params": req.model_dump(),
            "bpm": req.target_bpm,
            "level": req.level,
            "peaks": waveform_peaks(z),
            "grid": grid.to_dict(),
            "duration": len(z) / job.sr,
            "bitrate_kbps": bitrate,
            "analysis": None,
        }
        _write_json(job.dir / "render.json", rendered)
        job.rendered = rendered
        job.save()
        job.status = "rendered"
    except Exception as e:
        log.exception("render failed")
        job.status, job.error = "error", str(e)
        return
    finally:
        job.touched = time.monotonic()
    # Re-detect the beats in the output so the page can show how straight it really is.
    # This runs after the MP3 is ready, so the download does not wait for it. If the job
    # is re-rendered or re-analysed meanwhile, the result is dropped.
    try:
        result = analyse(z, job.sr, backend=job.analysis.detector).to_dict()
    except Exception as e:  # pragma: no cover
        log.exception("analysis of the rendered track failed")
        result = {"error": str(e)}
    with job.lock:
        if job.rendered is rendered:
            rendered["analysis"] = result
            _write_json(job.dir / "render.json", rendered)


@app.get("/api/health")
def health():
    return {
        "beat_this": beat_this_available(),
        "rubberband": rubberband_binary(),
        "ffmpeg": bool(shutil.which("ffmpeg")),
        "model": beat_this_state(),
        "storage": _storage(),
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
    _sweep()
    job = Job(uuid.uuid4().hex[:12])
    job.name = file.filename or "track.mp3"
    job.upload_name = "upload" + (Path(job.name).suffix.lower() or ".mp3")
    job.dir.mkdir()
    try:
        with open(job.upload, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except BaseException:
        shutil.rmtree(job.dir, ignore_errors=True)
        raise
    job.info = audio.probe(job.upload)
    job.save()
    JOBS[job.id] = job
    threading.Thread(target=_analyse_job, args=(job, detector), daemon=True).start()
    return {"id": job.id, "status": job.status}


def _list_status(live: Job | None, d: Path) -> str:
    """What a job is doing, for its badge. A job not loaded in this process is idle, so
    its files say where it got to. "measuring" is the re-detection that runs after a
    render, once the MP3 is already there."""
    if live is None:
        if not (d / "analysis.json").exists():
            return "error"
        return "rendered" if (d / "rendered.mp3").exists() else "ready"
    if (
        live.status == "rendered"
        and live.rendered
        and live.rendered["analysis"] is None
    ):
        return "measuring"
    return live.status


@app.get("/api/tracks")
def list_jobs():
    """Every job on the volume, newest first, with a download link for each file."""
    out = []
    for d in _job_dirs():
        meta = _read_json(d / "job.json")
        if not meta:
            continue
        tid = d.name
        live = JOBS.get(tid)
        files = []
        for p in sorted(d.iterdir()):
            kind = _KINDS.get(p.name) or (
                "upload" if p.name == meta.get("upload") else None
            )
            if kind is None or not p.is_file():
                continue
            files.append(
                {
                    "kind": kind,
                    "bytes": p.stat().st_size,
                    "url": f"/api/tracks/{tid}/files/{p.name}",
                    "download_name": _download_name(p.name, meta),
                }
            )
        tags = (meta.get("info") or {}).get("tags") or {}
        out.append(
            {
                "id": tid,
                "name": meta.get("name") or tid,
                "title": " - ".join(
                    x for x in (tags.get("artist"), tags.get("title")) if x
                ),
                "created": meta.get("created"),
                "updated": _newest_mtime(d),
                "status": _list_status(live, d),
                "detector": meta.get("detector"),
                "median_bpm": meta.get("median_bpm"),
                "render": meta.get("render"),
                "files": files,
            }
        )
    out.sort(key=lambda j: j["created"] or 0, reverse=True)
    return {"storage": _storage(), "jobs": out}


@app.get("/api/tracks/{tid}")
def get_track(tid: str):
    return JSONResponse(_get(tid).public())


@app.delete("/api/tracks/{tid}")
def delete_track(tid: str):
    job = _get(tid)
    if _busy(job):
        raise HTTPException(409, "the job is still working; wait for it to finish")
    return {"freed_bytes": _delete_job(tid)}


@app.post("/api/tracks/{tid}/analyse")
def start_analysis(tid: str, detector: str = "auto"):
    """Detect the beats again, e.g. with a different detector. The old straightened file
    was built on the old beats, so it goes."""
    job = _get(tid)
    if not job.upload.exists():
        raise HTTPException(410, "the uploaded file is gone; upload it again")
    with job.lock:
        if _busy(job):
            raise HTTPException(409, "the job is still working; wait for it to finish")
        job.status, job.error = "analysing", None
        job.analysis = None
        job.drop_render()
    threading.Thread(target=_analyse_job, args=(job, detector), daemon=True).start()
    return {"id": tid, "status": job.status}


@app.post("/api/tracks/{tid}/render")
def start_render(tid: str, req: RenderRequest):
    job = _get(tid)
    if job.analysis is None:
        raise HTTPException(409, "analysis not finished")
    if not job.upload.exists():
        raise HTTPException(410, "the uploaded file is gone; upload it again")
    if rubberband_binary() is None:
        raise HTTPException(500, "rubberband CLI not installed on the server")
    if not (20 <= req.target_bpm <= 400):
        raise HTTPException(422, "target_bpm out of range")
    if req.level not in (0.5, 1.0, 2.0):
        raise HTTPException(422, "level must be 0.5, 1 or 2")
    detected = job.analysis.median_bpm / req.level
    if detected and abs(req.target_bpm / detected - 1) > 0.3:
        raise HTTPException(
            422,
            f"target {req.target_bpm:g} BPM is more than 30% away from the detected {detected:.1f} BPM; "
            "that would change the speed of the track rather than straighten it. If the detector "
            "locked onto half- or double-time, change the beat level instead.",
        )
    with job.lock:
        if _busy(job):
            raise HTTPException(409, "the job is still working; wait for it to finish")
        job.status, job.error = "rendering", None
    threading.Thread(target=_render_job, args=(job, req), daemon=True).start()
    return {"id": tid, "status": job.status}


@app.get("/api/tracks/{tid}/audio/{which}")
def get_audio(tid: str, which: str):
    job = _get(tid)
    if which == "original":
        path = job.playback if job.playback.exists() else job.upload
        if not path.exists():
            raise HTTPException(410, "the job's files have been cleaned up")
        return FileResponse(path, media_type="audio/wav")
    if which == "rendered" and job.rendered and job.rendered_wav.exists():
        return FileResponse(job.rendered_wav, media_type="audio/wav")
    raise HTTPException(404)


def _download_stem(name: str, info: dict) -> str:
    """Name the download after the tags when the file carries them, the uploaded filename
    otherwise. Slashes and control characters would break the Content-Disposition header."""
    tags = info.get("tags") or {}
    stem = (
        " - ".join(x for x in (tags.get("artist"), tags.get("title")) if x)
        or Path(name).stem
    )
    return re.sub(r'[\x00-\x1f/\\:*?"<>|]', "_", stem).strip() or "track"


# The files in a job folder a user may want, and what to call each one in the list.
_KINDS = {
    "rendered.mp3": "straightened MP3",
    "rendered.wav": "straightened WAV",
    "original.wav": "playback WAV",
    "analysis.json": "beat data",
}


def _download_name(filename: str, meta: dict) -> str:
    """A readable filename for a file in a job folder."""
    stem = _download_stem(meta.get("name", filename), meta.get("info") or {})
    bpm = (meta.get("render") or {}).get("target_bpm")
    if filename.startswith("rendered.") and bpm:
        return f"{audio.straightened(stem, bpm)}{Path(filename).suffix}"
    if filename == "original.wav":
        return f"{stem} (decoded).wav"
    if filename == "analysis.json":
        return f"{stem} (beats).json"
    if filename == meta.get("upload"):
        return f"{Path(meta.get('name', filename)).stem}{Path(filename).suffix}"
    return filename


def _storage() -> dict:
    usage = shutil.disk_usage(DATA)
    return {
        "used_bytes": _data_bytes(),
        "disk_total_bytes": usage.total,
        "disk_free_bytes": usage.free,
        "min_free_fraction": MIN_FREE_FRACTION,
        "max_age_seconds": MAX_AGE_SECONDS,
        "jobs_in_memory": len(JOBS),
    }


@app.get("/api/tracks/{tid}/download")
def download(tid: str):
    job = _get(tid)
    if not job.rendered or not job.rendered_mp3.exists():
        raise HTTPException(404)
    return download_file(tid, job.rendered_mp3.name)


@app.get("/api/tracks/{tid}/files/{name}")
def download_file(tid: str, name: str):
    job = _get(tid)
    path = job.dir / name
    allowed = name in _KINDS or name == job.upload_name
    if not allowed or not path.is_file():
        raise HTTPException(404)
    meta = _read_json(job.dir / "job.json") or {}
    media = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".json": "application/json"}
    return FileResponse(
        path,
        media_type=media.get(path.suffix.lower(), "application/octet-stream"),
        filename=_download_name(name, meta),
    )


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
