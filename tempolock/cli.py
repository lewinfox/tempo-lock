"""Command line: tempolock analyse|render|serve."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import audio
from .analysis import analyse
from .render import render


def cmd_analyse(args):
    y, sr = audio.load(args.input)
    a = analyse(y, sr, backend=args.detector)
    print(
        json.dumps(
            {
                k: v
                for k, v in a.to_dict().items()
                if k not in ("bpm_times", "bpm_curve")
            },
            indent=None if args.compact else 2,
        )
    )


def cmd_render(args):
    y, sr = audio.load(args.input)
    a = analyse(y, sr, backend=args.detector)
    bpm = args.bpm or round(a.median_bpm / args.level)
    print(
        f"detector={a.detector} beats={len(a.beats)} median={a.median_bpm:.2f} range={a.min_bpm:.1f}-{a.max_bpm:.1f} -> target {bpm:g} BPM",
        file=sys.stderr,
    )
    z, grid = render(y, sr, a, bpm, engine=args.engine, level=args.level)
    out = (
        Path(args.output)
        if args.output
        else Path(args.input).with_name(Path(args.input).stem + f"_{bpm:g}bpm.mp3")
    )
    if out.suffix.lower() == ".mp3":
        audio.write_mp3(out, z, sr, copy_tags_from=args.input, bpm=bpm)
    else:
        audio.write_wav(out, z, sr)
    if args.grid:
        Path(args.grid).write_text(json.dumps(grid.to_dict()))
    print(str(out))


def cmd_serve(args):
    import uvicorn

    uvicorn.run(
        "tempolock.server:app", host=args.host, port=args.port, reload=args.reload
    )


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    p = argparse.ArgumentParser(prog="tempolock", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser(
        "analyse", help="detect beats and print tempo statistics as JSON"
    )
    a.add_argument("input")
    a.add_argument(
        "--detector", default="auto", choices=["auto", "beat_this", "librosa"]
    )
    a.add_argument("--compact", action="store_true")
    a.set_defaults(fn=cmd_analyse)

    r = sub.add_parser("render", help="straighten a track to a fixed BPM")
    r.add_argument("input")
    r.add_argument(
        "-o",
        "--output",
        help="output file (.mp3 or .wav); default <input>_<bpm>bpm.mp3",
    )
    r.add_argument(
        "--bpm",
        type=float,
        help="target BPM (default: rounded median of the detected tempo)",
    )
    r.add_argument(
        "--detector", default="auto", choices=["auto", "beat_this", "librosa"]
    )
    r.add_argument(
        "--level",
        type=float,
        default=1.0,
        choices=[0.5, 1.0, 2.0],
        help="2 if the detector found half-time, 0.5 for double-time",
    )
    r.add_argument(
        "--engine",
        default="r3",
        choices=["r3", "r2"],
        help="Rubber Band engine (r3 = best quality)",
    )
    r.add_argument(
        "--grid", help="also write the source/target beat grid to this JSON file"
    )
    r.set_defaults(fn=cmd_render)

    s = sub.add_parser("serve", help="run the web app")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(fn=cmd_serve)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
