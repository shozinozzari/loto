#!/usr/bin/env python3
"""
Overlay a QR image on top of a video for the first N seconds.

Default behavior:
  - Input video: CTA YT.mp4
  - QR image: amazon_product_qr.png
  - Position: top-right
  - Overlay duration: first 2.9 seconds
  - Output: CTA YT_with_qr.mp4

Example:
  python add_qr_to_cta_video.py --qr-width 260 --margin-x 24 --margin-y 24
  python add_qr_to_cta_video.py --qr-scale 0.40
"""

from __future__ import annotations

import argparse
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
ASSETS_VIDEO_DIR = PROJECT_ROOT / "assets" / "video"
ASSETS_IMAGE_DIR = PROJECT_ROOT / "assets" / "images"

DEFAULT_VIDEO = ASSETS_VIDEO_DIR / "CTA YT.mp4"
DEFAULT_QR_IMAGE = ASSETS_IMAGE_DIR / "amazon_product_qr.png"
DEFAULT_OUTPUT = ASSETS_VIDEO_DIR / "CTA YT_with_qr.mp4"
DEFAULT_QR_WIDTH = 470
DEFAULT_MARGIN_X = 0
DEFAULT_MARGIN_Y = 0

# Hardcoded size override (edit these values directly when needed).
# When True, CLI size flags are ignored.
USE_HARDCODED_QR_SIZE = True
HARDCODED_QR_WIDTH = 470
HARDCODED_QR_HEIGHT = 0  # 0 keeps aspect ratio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add a QR code overlay to the beginning of a video using ffmpeg."
    )
    parser.add_argument(
        "--video",
        default=str(DEFAULT_VIDEO),
        help=f"Input video file (default: {DEFAULT_VIDEO})",
    )
    parser.add_argument(
        "--qr-image",
        default=str(DEFAULT_QR_IMAGE),
        help=f"QR image file (default: {DEFAULT_QR_IMAGE})",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Output video file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--overlay-seconds",
        type=float,
        default=2.9,
        help="Show QR from t=0 to this many seconds (default: 2.9).",
    )
    parser.add_argument(
        "--position",
        choices=["top-right", "top-left", "bottom-right", "bottom-left"],
        default="top-right",
        help="QR position on video frame (default: top-right).",
    )
    parser.add_argument(
        "--qr-width",
        type=int,
        default=DEFAULT_QR_WIDTH,
        help=f"QR width in pixels (default: {DEFAULT_QR_WIDTH}).",
    )
    parser.add_argument(
        "--qr-scale",
        type=float,
        default=0.0,
        help=(
            "Optional QR width as a fraction of video width (0 disables). "
            "Example: 0.40 makes QR 40% of frame width."
        ),
    )
    parser.add_argument(
        "--qr-height",
        type=int,
        default=0,
        help="QR height in pixels. Use 0 to preserve aspect ratio (default: 0).",
    )
    parser.add_argument(
        "--margin-x",
        type=int,
        default=DEFAULT_MARGIN_X,
        help=f"Horizontal margin from edge in pixels (default: {DEFAULT_MARGIN_X}).",
    )
    parser.add_argument(
        "--margin-y",
        type=int,
        default=DEFAULT_MARGIN_Y,
        help=f"Vertical margin from edge in pixels (default: {DEFAULT_MARGIN_Y}).",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=20,
        help="libx264 CRF quality (lower is higher quality, default: 20).",
    )
    parser.add_argument(
        "--preset",
        default="medium",
        help="libx264 preset (default: medium).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print ffmpeg command without executing.",
    )
    return parser.parse_args()


def overlay_xy(position: str, margin_x: int, margin_y: int) -> tuple[str, str]:
    if position == "top-right":
        return f"main_w-overlay_w-{margin_x}", f"{margin_y}"
    if position == "top-left":
        return f"{margin_x}", f"{margin_y}"
    if position == "bottom-right":
        return f"main_w-overlay_w-{margin_x}", f"main_h-overlay_h-{margin_y}"
    return f"{margin_x}", f"main_h-overlay_h-{margin_y}"


def get_video_width(ffprobe_path: str, input_video: Path) -> int:
    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(input_video),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    width_text = (result.stdout or "").strip()
    width = int(width_text)
    if width <= 0:
        raise ValueError(f"Invalid video width from ffprobe: {width_text!r}")
    return width


def build_ffmpeg_cmd(
    args: argparse.Namespace,
    ffmpeg_path: str,
    qr_width: int,
    qr_height: int,
) -> list[str]:
    x_expr, y_expr = overlay_xy(args.position, args.margin_x, args.margin_y)
    scale_h = qr_height if qr_height > 0 else -1
    overlay_duration = max(0.0, float(args.overlay_seconds))

    filter_complex = (
        f"[1:v]scale=w={qr_width}:h={scale_h}[qr];"
        f"[0:v][qr]overlay=x={x_expr}:y={y_expr}:"
        f"enable='between(t,0,{overlay_duration:.3f})'[v]"
    )

    return [
        ffmpeg_path,
        "-y",
        "-i",
        str(Path(args.video).expanduser().resolve()),
        "-i",
        str(Path(args.qr_image).expanduser().resolve()),
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        str(args.preset),
        "-crf",
        str(args.crf),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        str(Path(args.output).expanduser().resolve()),
    ]


def main() -> None:
    args = parse_args()

    input_video = Path(args.video).expanduser().resolve()
    qr_image = Path(args.qr_image).expanduser().resolve()
    output_video = Path(args.output).expanduser().resolve()

    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")
    if not qr_image.exists():
        raise FileNotFoundError(f"QR image not found: {qr_image}")
    if args.qr_width <= 0:
        raise ValueError("--qr-width must be > 0")
    if args.qr_scale < 0:
        raise ValueError("--qr-scale must be >= 0")
    if args.qr_height < 0:
        raise ValueError("--qr-height must be >= 0")
    if args.margin_x < 0 or args.margin_y < 0:
        raise ValueError("--margin-x and --margin-y must be >= 0")

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg is required in PATH.")
    ffprobe_path = shutil.which("ffprobe")
    if not ffprobe_path:
        raise RuntimeError("ffprobe is required in PATH.")

    if USE_HARDCODED_QR_SIZE:
        if HARDCODED_QR_WIDTH <= 0:
            raise ValueError("HARDCODED_QR_WIDTH must be > 0")
        if HARDCODED_QR_HEIGHT < 0:
            raise ValueError("HARDCODED_QR_HEIGHT must be >= 0")
        effective_qr_width = int(HARDCODED_QR_WIDTH)
        effective_qr_height = int(HARDCODED_QR_HEIGHT)
        size_source = "hardcoded constants"
    else:
        effective_qr_width = int(args.qr_width)
        if args.qr_scale > 0:
            video_width = get_video_width(ffprobe_path, input_video)
            effective_qr_width = max(16, int(round(video_width * float(args.qr_scale))))
        effective_qr_height = int(args.qr_height)
        size_source = "CLI flags"

    output_video.parent.mkdir(parents=True, exist_ok=True)

    cmd = build_ffmpeg_cmd(args, ffmpeg_path, effective_qr_width, effective_qr_height)
    print(f"Input video: {input_video}")
    print(f"QR image: {qr_image}")
    print(f"Output video: {output_video}")
    print(
        f"Overlay: position={args.position}, size={effective_qr_width}x"
        f"{effective_qr_height if effective_qr_height > 0 else 'auto'}, "
        f"margins=({args.margin_x},{args.margin_y}), duration={args.overlay_seconds:.2f}s"
    )
    print(f"QR size source: {size_source}")

    if args.dry_run:
        print("Dry run command:")
        print(" ".join(f'"{part}"' if " " in part else part for part in cmd))
        return

    subprocess.run(cmd, check=True)
    print("Done.")


if __name__ == "__main__":
    main()
