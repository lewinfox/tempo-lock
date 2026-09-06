#!/usr/bin/env bash
# One-shot setup for tempo-lock on Debian/Ubuntu or macOS. Re-runnable.
set -euo pipefail
cd "$(dirname "$0")"

if command -v apt-get >/dev/null; then
  echo ">> system packages (rubberband-cli, ffmpeg)"
  sudo apt-get install -y rubberband-cli ffmpeg
elif command -v brew >/dev/null; then
  echo ">> system packages (rubberband, ffmpeg)"
  brew install rubberband ffmpeg
else
  echo "!! install the Rubber Band CLI and ffmpeg yourself, then re-run" >&2
fi

if ! command -v uv >/dev/null; then
  echo ">> installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# Creates .venv, fetches the Python in .python-version, installs from uv.lock.
# torch comes from PyTorch's CPU index (see [tool.uv.sources] in pyproject.toml):
# ~10x smaller than the CUDA build and plenty fast for this.
echo ">> syncing the environment"
uv sync

echo
echo "done. run the web app with:   uv run tempolock serve"
echo "or from the command line:     uv run tempolock render some_track.mp3"
