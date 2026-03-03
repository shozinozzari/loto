#!/usr/bin/env python3
r"""
Download the first video of the first product from product_video_results.json.

Default input:
  <project_root>/data/product_video_results.json

Default output:
  <project_root>/data/downloaded_videos/<ASIN>/<ASIN>_video_01.mp4
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        if (parent / "requirements.txt").exists():
            return parent
    return here.parent


PROJECT_ROOT = _project_root()
DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_INPUT_FILE = DATA_DIR / "product_video_results.json"
DEFAULT_DOWNLOAD_DIR = DATA_DIR / "downloaded_videos"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the first video of the first product in product_video_results.json."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_FILE),
        help="Path to product_video_results.json",
    )
    parser.add_argument(
        "--download-dir",
        default=str(DEFAULT_DOWNLOAD_DIR),
        help="Base folder for downloaded videos.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=600,
        help="ffmpeg timeout in seconds.",
    )
    return parser.parse_args()


def load_first_video_info(input_path: Path) -> tuple[str, str]:
    payload = json.loads(input_path.read_text(encoding="utf-8-sig"))
    results = payload.get("results", [])
    if not isinstance(results, list) or not results:
        raise ValueError("No products found in results.")

    first = results[0]
    if not isinstance(first, dict):
        raise ValueError("First product entry is invalid.")

    asin = str(first.get("asin", "")).strip() or "unknown"
    video_urls = first.get("video_urls", [])
    if not isinstance(video_urls, list) or not video_urls:
        raise ValueError("First product has no video URLs.")

    first_video_url = str(video_urls[0]).strip()
    if not first_video_url:
        raise ValueError("First video URL is empty.")

    return asin, first_video_url


def download_with_ffmpeg(ffmpeg_path: str, video_url: str, output_file: Path, timeout_seconds: int) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_path,
        "-y",
        "-nostdin",
        "-loglevel",
        "error",
        "-user_agent",
        DEFAULT_USER_AGENT,
        "-i",
        video_url,
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(output_file),
    ]
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=max(60, int(timeout_seconds)),
        check=False,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip() or f"ffmpeg exited with code {completed.returncode}"
        raise RuntimeError(stderr)
    if not output_file.exists() or output_file.stat().st_size <= 0:
        raise RuntimeError("ffmpeg finished but output file is missing or empty.")


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    ffmpeg_path = shutil.which("ffmpeg") or ""
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg is required but was not found in PATH.")

    asin, first_video_url = load_first_video_info(input_path)
    download_dir = Path(args.download_dir).expanduser().resolve()
    output_file = download_dir / asin / f"{asin}_video_01.mp4"

    print(f"Input: {input_path}")
    print(f"ASIN: {asin}")
    print(f"Video URL: {first_video_url}")
    print(f"Output: {output_file}")

    download_with_ffmpeg(
        ffmpeg_path=ffmpeg_path,
        video_url=first_video_url,
        output_file=output_file,
        timeout_seconds=args.timeout_seconds,
    )

    print(f"Downloaded: {output_file}")
    print(f"Size: {output_file.stat().st_size} bytes")


if __name__ == "__main__":
    main()
