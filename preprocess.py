import subprocess
from pathlib import Path
import math
from argparse import ArgumentParser


def make_sliding_clips_ffmpeg(
        input_path: str, output_dir: str,
        clip_len: float = 5.0, step: float = 1.0,
        start: float = 0, end: float = 60):

    input_path = Path(input_path)
    output_dir = Path(output_dir) if output_dir else input_path.parent / f"{input_path.stem}_split"
    output_dir.mkdir(parents=True, exist_ok=True)

    last_start = max(0.0, end - clip_len)
    n = int(math.floor(last_start / step)) + 1
#    print(f"Generating {n} clips...")

    for i in range(n):
        start = i * step
        out_file = output_dir / f"{start:03d}_{start+clip_len:03d}.mp4"

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", input_path,
            "-t", str(clip_len),
            "-c", "copy",
            "-map", "0",
            "-avoid_negative_ts", "make_zero",
            str(out_file),
        ]

        # for less console noise, set stdout/stderr to DEVNULL
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    ap = ArgumentParser("This program splits a video into short clips")
    ap.add_argument("-i", "--file", type=str)
    ap.add_argument("--clip_len", type=float, default=5)
    ap.add_argument("--step", type=float, default=2)
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--end", type=float, default=60)
    ap.add_argument("--outdir", type=str, default="")
    args = ap.parse_args()

    # todo: check end time vs total length

    make_sliding_clips_ffmpeg(args.file, args.outdir, clip_len=args.clip_len, step=args.step, start=args.start, end=args.end)
