# tempo-lock: CPU-only image with Beat This!, Rubber Band and ffmpeg baked in.
#   docker build -t tempo-lock .
#   docker run --rm -p 8000:8000 -v tempo-lock-data:/data tempo-lock
# Or: docker compose up --build
FROM python:3.11-slim-bookworm

# uv resolves and installs from the committed uv.lock, so the image gets exactly the
# versions that were tested. Pinned so a rebuild is reproducible.
COPY --from=ghcr.io/astral-sh/uv:0.12.2 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    TORCH_HOME=/models \
    TEMPOLOCK_DATA=/data \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:$PATH

# rubberband-cli 3.x (R3 engine) + ffmpeg (decode anything, encode MP3 with tags)
RUN apt-get update \
 && apt-get install -y --no-install-recommends rubberband-cli ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, without the project itself, so editing tempolock/ does not
# re-download torch. pyproject pins torch to PyTorch's CPU index: ~10x smaller than the
# default CUDA wheels and plenty fast for this.
# The dev group (pytest, httpx) is deliberately included - CI runs the suite inside this
# image so it tests the artefact it is about to ship.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# bake the Beat This! checkpoint (~80 MB) so the container works offline and the first
# analysis does not stall on a download
RUN python -c "from beat_this.inference import load_checkpoint; load_checkpoint('final0')"

COPY README.md ./
COPY tempolock ./tempolock
COPY static ./static
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

RUN useradd --create-home --uid 1000 app \
 && mkdir -p /data \
 && chown -R app:app /app /data /models \
 && chmod +x /usr/local/bin/docker-entrypoint.sh
VOLUME ["/data"]

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health').status==200 else 1)"

# entrypoint chowns the mounted volume as root, then drops to the app user
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["tempolock", "serve", "--host", "0.0.0.0", "--port", "8000"]
