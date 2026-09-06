# tempo-lock

Take a track with live drums (so the tempo wanders), find every beat, and re-time the
audio so the beats sit on a fixed-BPM grid, without changing pitch. Makes live
recordings behave like quantised tracks when DJing.

```
MP3 in ──► Beat This! (beats + downbeats)
        ──► snap each beat to the drum transient, fix missed/spurious beats
        ──► build a fixed grid at the target BPM
        ──► Rubber Band R3 time map (variable stretch, pitch untouched)
        ──► MP3 out (tags copied, TBPM set)
```

Web app: waveform + beat grid for the original and the straightened version, a
tempo-over-time chart, A/B playback that keeps your place when you switch, and a click
track on the grid so you can hear whether the beats really land.

## Run it

```bash
cd tempo-lock
./setup.sh                      # apt/brew: rubberband-cli + ffmpeg; then uv sync
uv run tempolock serve          # http://127.0.0.1:8000
```

Dependencies are managed with [uv](https://docs.astral.sh/uv/): `pyproject.toml` declares
them, `uv.lock` pins them, and `uv sync` builds `.venv` from the lock so everyone (and the
Docker image, and CI) gets the same versions. `[tool.uv.sources]` points torch at PyTorch's
CPU index - the default wheels bundle CUDA and are ~10x bigger for no benefit here. Add a
dependency with `uv add <pkg>`; run anything in the environment with `uv run ...`.

Docker (no Python or system packages needed on the host):

```bash
cd tempo-lock
docker compose up --build        # http://127.0.0.1:8000
# or without compose:
docker build -t tempo-lock .
docker run --rm -p 8000:8000 -v tempo-lock-data:/data tempo-lock
```

The image is CPU-only (`python:3.11-slim` + rubberband-cli + ffmpeg + CPU torch, installed
with `uv sync --frozen` from the same `uv.lock` you develop against) and bakes
in the Beat This! checkpoint, so it needs no network at run time. Expect ~2 GB. Uploads
and rendered files go to the `/data` volume; the job table is in memory, so a container
restart forgets tracks (and wipes the volume on the way back up - see
[Storage](#storage)). The container starts as root only long enough to chown the mounted
volume, then runs the server as an unprivileged user. The CLI works in the container too:

```bash
docker run --rm --user "$(id -u)" -v "$PWD:/work" tempo-lock \
  tempolock render /work/track.mp3 -o /work/out.mp3
```

Command line:

```bash
uv run tempolock analyse track.mp3              # beats, downbeats, tempo stats as JSON
uv run tempolock render track.mp3               # writes track_124bpm.mp3 at the rounded median tempo
uv run tempolock render track.mp3 --bpm 124.5 -o out.mp3 --grid grid.json
uv run tempolock render track.mp3 --level 2     # detector locked onto half-time
uv run pytest tests                             # ~40 s, synthesises its own test track
```

First analysis downloads the Beat This! checkpoint (~80 MB) and loads the model (~7 s).
After that a 5-minute track takes roughly 20 s to analyse and 15 s to render on a
4-core CPU. A GPU is used automatically if torch sees one.

## How it works

**Detection.** [Beat This!](https://github.com/CPJKU/beat_this) (CPJKU, ISMIR 2024)
runs on the mono mix and gives beat and downbeat times at 20 ms resolution. Its output is
plain peak-picking with no tempo-continuity prior, which is exactly what you want when
the tempo genuinely moves. We deliberately do *not* enable its optional DBN
post-processor: the DBN enforces smooth tempo and is the part that fails on unstable
tempo (see the "SMC Blind Spot" paper below).

**Refinement** (`tempolock/analysis.py`). 20 ms is too coarse for a DJ grid, so each
beat is snapped to the strongest onset within ±35 ms of it, computed from a
librosa onset-strength envelope at 5.8 ms hops. Then beats are walked in order with a
rolling local period: a gap of ~2 periods means a missed beat (the beat *index* jumps by
2 so the grid stays honest), a beat arriving at <0.6 periods is a spurious double and is
dropped, and detections in near-silence (a phantom beat at t=0 is common) are discarded.
Dropped beats are shown as small red ticks in the UI.

**Deliberate tempo changes** (`segment_tempo`). A drummer drifting and a song that
changes tempo look the same to a single-BPM grid, but they should not be treated the
same: flattening a deliberate 120→140 change slows a whole section by 14%. So the
per-beat tempo curve is scanned for *steps*: at each candidate boundary a line is fitted
to the 8 beats on either side and the gap between the two lines at the boundary is
measured. A ramp, however steep, has continuous lines and no gap; a real step shows the
whole jump. Steps of 5% or more that hold for at least 8 beats become section
boundaries. Fills and push/pull are absorbed by a 3-beat median and the minimum section
length. The UI shows a warning with the time and size of each change (click to jump
there), shades the sections on the tempo chart, and draws a histogram of per-beat BPM
coloured by section, so two tempos show up as two humps. Rendering is deliberately one
BPM for the whole track: the point of the tool is a file you can beatmatch end to end.
The warning exists so you know when that choice is stretching a whole section rather
than tidying drift, and can pick the target BPM with that in mind.

**Grid.** Target beat k lands at `t_first + k · 60/BPM`, so the intro before the first
beat and the tail after the last are left untouched (ratio 1). Default BPM is the
rounded median of the instantaneous tempo. The UI refuses a target more than 30% from
the detected tempo because that would change the *speed* of the track rather than
straighten it; if the detector locked onto half- or double-time, set "Detected beats
are" instead of typing 2× the number.

**Stretch** (`tempolock/render.py`). Rubber Band's CLI accepts a time map: pairs of
`source_frame target_frame` between which it varies the stretch ratio smoothly. We
give it one pair per beat plus the end of file and run the R3 engine (`-3`). Pitch is not
touched. Gotcha found the hard way: don't include a leading `0 0` pair, R3 divides by
zero on it and silently skips stretching those segments.

**Verification.** `tests/test_end_to_end.py` synthesises a drum track whose tempo
swings ±5 BPM around 120, straightens it, re-detects the beats on the output and asserts
the inter-beat interval is constant to within a few ms. On that material the rendered
grid is within ~2 ms (median) of where we said the beats would be.

**Playback alignment.** The browser plays a server-decoded PCM copy of both versions.
MP3 decoders disagree about the encoder delay by ~25 ms, which is enough to make a
correct grid look wrong, so the browser and the analyser must share one decode.

## Audio quality

The download path is float32 end to end: libsndfile decodes to float, Rubber Band takes a
`FLOAT` WAV and hands one back, and the encoder gets a `FLOAT` WAV. The 16-bit writes in
`server.py` are the browser's playback copies and never feed the download.

What you do lose:

- **A second lossy generation.** An MP3 in means an MP3 out, and decode/re-encode always
  costs something even at a matched bitrate. Nothing avoids this short of a lossless
  download, which the pipeline could offer unchanged.
- **Rubber Band's resynthesis.** It is a phase-vocoder-family transform, not a
  sample-preserving edit. That is the job.
- **Peak normalisation**, but only when the stretched signal exceeds 0.999.

The output is encoded at the source's own bitrate, snapped to a rate libmp3lame accepts
(`audio.mp3_bitrate_for`) - a 128 kbps upload comes back at 128 kbps rather than a 320 kbps
file 2.5x the size and no better. Lossless sources have no MP3 equivalent, so they get 320.

`ffprobe` supplies the track's tags and technical details on upload; the UI shows the title,
artist, album and year, falling back to the filename when the file carries no tags, and the
download is named from the tags where they exist. ID3 text is escaped before it reaches the
DOM - anyone can craft an MP3.

## Beat tracker research (Sept 2026)

The requirement is offline, per-beat timestamps on music with drifting tempo, ideally
with downbeats. Summary of what is realistically installable:

| Library | GTZAN beat F1 / downbeat F1 | Handles drifting tempo | Downbeats | Weight | Install on py3.11 | Licence |
|---|---|---|---|---|---|---|
| **Beat This!** (CPJKU 2024) | **0.891 / 0.783** | Yes, by design (no tempo prior, trained with ±20% speed aug.) | Yes | torch, 80 MB ckpt | `pip install beat_this` | MIT |
| madmom RNN+DBN | ~0.88 / – | DBN enforces tempo continuity, 55–215 BPM, 3/4 or 4/4 | Yes | numpy only | broken on PyPI (2018); works from git | BSD, models CC-BY-NC |
| BeatNet / BeatNet+ | 0.75 / 0.47 (0.81 / 0.57) | online particle filter; offline uses madmom DBN | Yes | torch + madmom | PyPI unresolvable on 3.11 | CC-BY |
| All-In-One (Kim 2023) | Harmonix 0.958 / 0.915, no GTZAN | madmom DBN | Yes + segments | torch, Demucs, NATTEN | fragile, unmaintained since 2023 | MIT |
| Essentia RhythmExtractor2013 | not published as F1 | whole-track statistics | No | C++ wheels | `pip install essentia` | AGPL |
| librosa `beat_track` / `plp` | classic DP baseline, well below | `beat_track` assumes one tempo; `plp` is local | No | none extra | trivial | ISC |

Beat This! wins on every axis that matters here. The 2026 masked-diffusion follow-up
(GTZAN 0.897 / 0.795) has no released inference code yet. librosa is kept as a fallback
so the app still runs without torch, but expect it to miss fills and drift.

Sources: [Beat This! paper](https://arxiv.org/abs/2407.21658) ·
[SMC Blind Spot failure analysis](https://arxiv.org/abs/2605.12287) ·
[madmom numpy-2 PR](https://github.com/CPJKU/madmom/pull/540) ·
[BeatNet+](https://transactions.ismir.net/articles/10.5334/tismir.198) ·
[All-In-One](https://arxiv.org/abs/2307.16425) ·
[Rubber Band CLI](https://breakfastquay.com/rubberband/usage.txt).

### Why Rubber Band and not a phase vocoder

Drums are transients. Phase-vocoder stretchers (librosa, most Python TSM packages)
smear them and only do constant ratios anyway. Rubber Band R3 handles transients well,
is GPL, ships in every distro, and its time-map mode is the only off-the-shelf way to do
a *continuously varying* stretch without stitching segments together yourself. R2 with
`--crisp 6` is offered as the faster option and is worth an A/B on very dry drum-only
material.

## Layout

```
tempolock/analysis.py   detector output -> refined, indexed beats + tempo curve
tempolock/detectors.py  Beat This! (cached model) and librosa fallback
tempolock/render.py     grid, Rubber Band time map, stretch
tempolock/audio.py      decode anything, MP3 encode with tag copy + TBPM
tempolock/server.py     FastAPI: upload, analyse, render, stream, download
tempolock/cli.py        analyse | render | serve
static/                 index.html, app.js, style.css (no build step)
tests/                  unit tests + synthetic end-to-end test
pyproject.toml          dependencies, CPU-torch index, the `tempolock` entry point
uv.lock                 the pinned resolution used by dev, Docker and CI alike
```

## Storage

One 5-minute track costs roughly 120 MB on disk while you work on it: the upload, a
decoded WAV the browser plays so its timeline matches the server's, the rendered WAV, and
the rendered MP3. On a small cloud volume that adds up fast, so the server cleans up after
itself in four ways:

| When | What goes |
| --- | --- |
| The browser finishes downloading the rendered MP3 | Every file for that track except the MP3 itself - the WAVs are ~95% of the bytes, and keeping the MP3 means a second click on the download link still works. The page holds its own decoded copies in Web Audio buffers, so playback carries on. |
| A track goes untouched for `TEMPOLOCK_TTL_SECONDS` | Everything, including the decoded samples held in RAM. This is what catches uploads that are analysed and then abandoned. |
| `/data` exceeds `TEMPOLOCK_MAX_DATA_BYTES` | Least-recently-touched idle tracks, oldest first, until it is back under the cap. Tracks mid-analysis or mid-render are never evicted. |
| Startup | The whole data dir. The job table is in memory, so files from a previous process are unreachable anyway. |

Uploads bigger than `TEMPOLOCK_MAX_UPLOAD_BYTES` are rejected with a 413 while streaming,
before the whole body lands on the volume. `GET /api/health` reports current usage, and
`DELETE /api/tracks/{id}` drops a track on demand.

| Variable | Default | Meaning |
| --- | --- | --- |
| `TEMPOLOCK_DATA` | `./data` | Where uploads and renders live |
| `TEMPOLOCK_DELETE_AFTER_DOWNLOAD` | `1` | Set to `0` to keep files after download |
| `TEMPOLOCK_TTL_SECONDS` | `3600` | Idle lifetime of a track |
| `TEMPOLOCK_MAX_DATA_BYTES` | `2147483648` | Hard cap on `/data` |
| `TEMPOLOCK_MAX_UPLOAD_BYTES` | `157286400` | Largest accepted upload |
| `TEMPOLOCK_SWEEP_SECONDS` | `120` | How often the reaper runs |

## Deploy to Fly.io

`fly.toml` is set up for a single machine in `syd` with a 3 GB volume mounted at `/data`.
One-off bootstrap:

```bash
fly launch --no-deploy --copy-config --name tempo-lock-lewinfox   # claim the name
fly deploy --dockerfile Dockerfile                                # build from source, this once
fly open
```

After that, `.github/workflows/deploy.yml` ships every merge to `main`: it builds the
image, pushes it to GHCR, runs the test suite *inside that image*, and only then runs
`flyctl deploy --image ...@sha256:...` pinned to the digest it just tested. A red test
blocks the deploy. `workflow_dispatch` lets you re-run it by hand.

Two bits of setup, both one-off:

- Add a deploy token as the `FLY_IO_API_KEY` repo secret:
  `fly tokens create deploy -a tempo-lock-lewinfox`.
- Make the GHCR package public (Packages → tempo-lock → Package settings → Change
  visibility). Fly pulls the image with no registry credentials.

`[build] image` in `fly.toml` points at `:latest` so a manual `fly deploy` ships whatever
CI last built; pass `--dockerfile Dockerfile` to build from your working tree instead.

The image is ~2 GB because of torch and the baked-in checkpoint, so the first build is
slow; later deploys reuse the buildx GHA cache. `[[vm]]` asks for `shared-cpu-4x` / 4 GB - Beat
This! is a torch model and Rubber Band R3 is CPU-hungry, and a 1x machine takes minutes per
track.

The machine has `auto_stop_machines = "stop"` and `min_machines_running = 0`, so it sleeps
when nobody is using it and cold-starts on the next request (a few seconds to load torch).
The browser polls while a track is analysing or rendering, which keeps the machine awake
for the duration; leave the tab open.

Tune the storage guard rails in the `[env]` block of `fly.toml`, and watch the volume with:

```bash
fly ssh console -C "df -h /data"
curl -s https://tempo-lock-lewinfox.fly.dev/api/health | jq .storage
```

## Known limits / ideas

- Tracks with a deliberate tempo change are flattened to one BPM by design. If you ever
  want each section kept at its own steady tempo, the time map already supports it; only
  the grid builder and a per-section BPM control would change.
- No manual grid editing yet (nudge a beat, insert/delete one, set the downbeat). The
  data model supports it; it is mostly UI work.
- Tracks that change time signature or have long free-time sections will get a grid
  that is technically right but musically odd. Rendering only a section is not
  supported yet.
- Uploaded files live in `tempo-lock/data/` and the job table is in memory. Restarting
  the server forgets tracks and clears the data dir.
- Beat This! processes 30 s chunks, so a beat straddling a chunk border can very
  occasionally be missed; the index logic covers that.
