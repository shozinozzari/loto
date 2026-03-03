#!/usr/bin/env python3
r"""
Create a 1:1 square reel by centering the source video and applying external audio.
Video duration is auto-adjusted to audio length by changing video speed.
Can optionally append a CTA clip with QR overlay to produce a complete reel.

Example:
  python make_square_reel.py ^
    --video "C:\path\input.mp4" ^
    --audio "C:\path\narration.wav" ^
    --output "C:\path\output_square.mp4"

  python make_square_reel.py ^
    --base-reel "C:\path\existing_square_reel.mp4" ^
    --complete-output "C:\path\existing_square_reel_complete.mp4"

Run Python File mode:
  - Auto-picks latest downloaded *_video_01.mp4
  - Auto-picks matching/latest *_promo_ml.wav
  - Writes to <project_root>/data/reel/<video_stem>_square_reel.mp4
"""

from __future__ import annotations

import argparse
from datetime import datetime
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        if (parent / "requirements.txt").exists():
            return parent
    return here.parent


PROJECT_ROOT = _project_root()
ASSETS_DIR = PROJECT_ROOT / "assets"
DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_DOWNLOADED_DIR = DATA_DIR / "downloaded_videos"
DEFAULT_GEMINI_OUTPUTS_DIR = DATA_DIR / "gemini_outputs"
DEFAULT_REEL_DIR = DATA_DIR / "reel"
DEFAULT_CTA_VIDEO = ASSETS_DIR / "video" / "CTA YT.mp4"
DEFAULT_QR_IMAGE = ASSETS_DIR / "images" / "amazon_product_qr.png"
DEFAULT_FINAL_MUSIC = ASSETS_DIR / "audio" / "reel Music.mp3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a video to centered 1:1 square reel, combine with external audio, "
            "and optionally append a QR CTA clip."
        )
    )
    parser.add_argument(
        "--base-reel",
        default="",
        help=(
            "Existing square reel to finalize. If set, square generation from --video/--audio "
            "is skipped and this file is used as the main reel."
        ),
    )
    parser.add_argument(
        "--video",
        default="",
        help=(
            "Input video path. If omitted, uses latest *_video_01.mp4 from "
            f"{DEFAULT_DOWNLOADED_DIR}."
        ),
    )
    parser.add_argument(
        "--audio",
        default="",
        help=(
            "Input audio path (wav/mp3/m4a). If omitted, tries matching <ASIN>_promo_ml.wav "
            f"from {DEFAULT_GEMINI_OUTPUTS_DIR}, then latest *_promo_ml.wav."
        ),
    )
    parser.add_argument(
        "--output",
        default="",
        help=(
            "Square reel output path (.mp4). If omitted, writes to "
            f"{DEFAULT_REEL_DIR}\\<video_stem>_square_reel.mp4."
        ),
    )
    parser.add_argument(
        "--complete-output",
        default="",
        help=(
            "Final complete reel output path after CTA append. If omitted, writes to "
            "<square_reel_stem>_complete.mp4 in the same folder."
        ),
    )
    parser.add_argument("--size", type=int, default=1080, help="Square output size in pixels.")
    parser.add_argument("--fps", type=int, default=30, help="Output frame rate.")
    parser.add_argument("--crf", type=int, default=20, help="libx264 CRF quality (lower is better).")
    parser.add_argument("--preset", default="medium", help="libx264 preset (ultrafast..veryslow).")
    parser.add_argument(
        "--background",
        choices=["blur", "black"],
        default="blur",
        help="Background style behind centered video.",
    )
    parser.add_argument(
        "--audio-volume",
        type=float,
        default=1.0,
        help="External audio gain multiplier (1.0 = unchanged).",
    )
    parser.add_argument(
        "--skip-cta-append",
        action="store_true",
        help="If set, do not append CTA clip. Only produce/use square reel.",
    )
    parser.add_argument(
        "--cta-video",
        default=str(DEFAULT_CTA_VIDEO),
        help=f"CTA clip appended at the end (default: {DEFAULT_CTA_VIDEO}).",
    )
    parser.add_argument(
        "--qr-image",
        default=str(DEFAULT_QR_IMAGE),
        help=f"QR image to overlay on CTA clip (default: {DEFAULT_QR_IMAGE}).",
    )
    parser.add_argument(
        "--cta-overlay-first-end",
        type=float,
        default=2.9,
        help="Show QR on CTA clip from t=0 up to this timestamp in seconds (default: 2.9).",
    )
    parser.add_argument(
        "--cta-overlay-start",
        type=float,
        default=7.9,
        help="Show QR on CTA clip again starting at this timestamp in seconds (default: 7.3).",
    )
    parser.add_argument(
        "--cta-overlay-end",
        type=float,
        default=-1.0,
        help="Show QR until this timestamp in seconds. Use < 0 to keep it visible till CTA ends.",
    )
    parser.add_argument(
        "--cta-position",
        choices=["top-right", "top-left", "bottom-right", "bottom-left"],
        default="top-right",
        help="QR position on CTA clip (default: top-right).",
    )
    parser.add_argument(
        "--cta-qr-width",
        type=int,
        default=470,
        help="QR width in pixels on CTA clip (default: 470).",
    )
    parser.add_argument(
        "--cta-qr-scale",
        type=float,
        default=0.6,
        help=(
            "Optional QR width as fraction of square frame width (0 disables). "
            "Example: 0.5 means 50%% of frame width, 1.0 means full frame width."
        ),
    )
    parser.add_argument(
        "--cta-qr-height",
        type=int,
        default=0,
        help="QR height in pixels on CTA clip. Use 0 to preserve aspect ratio.",
    )
    parser.add_argument(
        "--cta-margin-x",
        type=int,
        default=190,
        help="Horizontal margin for CTA QR overlay in pixels (higher moves QR more toward center).",
    )
    parser.add_argument(
        "--cta-margin-y",
        type=int,
        default=0,
        help="Vertical margin for CTA QR overlay in pixels.",
    )
    parser.add_argument(
        "--final-music",
        default="",
        help="Optional background music file to mix into the final complete reel.",
    )
    parser.add_argument(
        "--use-default-final-music",
        action="store_true",
        help=f"Use default final music track: {DEFAULT_FINAL_MUSIC}",
    )
    parser.add_argument(
        "--no-final-music",
        action="store_true",
        help="Disable adding background music in the final complete reel.",
    )
    parser.add_argument(
        "--final-music-volume",
        type=float,
        default=0.7,
        help="Background music volume in final reel mix (default: 1.0).",
    )
    parser.add_argument(
        "--final-voice-volume",
        type=float,
        default=1.0,
        help="Voice/primary audio volume in final reel mix (default: 1.0).",
    )
    return parser.parse_args()


def find_latest_file(base_dir: Path, pattern: str) -> Path | None:
    if not base_dir.exists():
        return None
    files = [p for p in base_dir.rglob(pattern) if p.is_file()]
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def infer_asin(path: Path) -> str:
    text = f"{path.parent.name} {path.stem}".upper()
    match = re.search(r"\b([A-Z0-9]{10})\b", text)
    return match.group(1) if match else ""


def run_command(cmd: list[str]) -> str:
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        detail = stderr or stdout or f"exit code {completed.returncode}"
        raise RuntimeError(f"Command failed: {' '.join(cmd)} | {detail[:800]}")
    return (completed.stdout or "").strip()


def can_write_output_path(path: Path) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            with open(path, "a+b"):
                pass
        else:
            with open(path, "xb"):
                pass
            path.unlink(missing_ok=True)
        return True
    except (PermissionError, OSError):
        return False


def resolve_writable_output_path(path: Path) -> Path:
    if can_write_output_path(path):
        return path

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for idx in range(1, 100):
        candidate = path.with_name(f"{path.stem}_{stamp}_{idx:02d}{path.suffix}")
        if can_write_output_path(candidate):
            return candidate

    raise RuntimeError(f"Could not find writable output filename near: {path}")


def has_audio_stream(ffprobe_path: str, media_path: Path) -> bool:
    out = run_command(
        [
            ffprobe_path,
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(media_path),
        ]
    )
    return bool(str(out).strip())


def get_duration_seconds(ffprobe_path: str, media_path: Path) -> float:
    out = run_command(
        [
            ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(media_path),
        ]
    )
    try:
        value = float(out.strip().splitlines()[-1].strip())
    except Exception as exc:
        raise RuntimeError(f"Could not parse duration for {media_path}: {out}") from exc
    if value <= 0:
        raise RuntimeError(f"Invalid duration for {media_path}: {value}")
    return value


def build_video_filter(size: int, fps: int, background: str, setpts_factor: float) -> str:
    pts = max(0.05, float(setpts_factor))
    if background == "black":
        return (
            f"color=c=black:s={size}x{size}:r={fps}[bg];"
            f"[0:v]setpts={pts:.8f}*PTS,scale={size}:{size}:force_original_aspect_ratio=decrease[fg];"
            "[bg][fg]overlay=(W-w)/2:(H-h)/2,format=yuv420p[v]"
        )

    return (
        f"[0:v]setpts={pts:.8f}*PTS,split=2[vb][vf];"
        f"[vb]scale={size}:{size}:force_original_aspect_ratio=increase,"
        f"crop={size}:{size},boxblur=20:2[bg];"
        f"[vf]scale={size}:{size}:force_original_aspect_ratio=decrease[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2,format=yuv420p[v]"
    )


def overlay_xy(position: str, margin_x: int, margin_y: int) -> tuple[str, str]:
    if position == "top-right":
        return f"main_w-overlay_w-{margin_x}", f"{margin_y}"
    if position == "top-left":
        return f"{margin_x}", f"{margin_y}"
    if position == "bottom-right":
        return f"main_w-overlay_w-{margin_x}", f"main_h-overlay_h-{margin_y}"
    return f"{margin_x}", f"main_h-overlay_h-{margin_y}"


def build_cta_qr_filter(
    size: int,
    fps: int,
    position: str,
    margin_x: int,
    margin_y: int,
    qr_width: int,
    qr_height: int,
    overlay_first_end_seconds: float,
    overlay_start_seconds: float,
    overlay_end_seconds: float,
) -> str:
    x_expr, y_expr = overlay_xy(position, margin_x, margin_y)
    scale_h = qr_height if qr_height > 0 else -1
    first_end_t = max(0.0, float(overlay_first_end_seconds))
    start_t = max(0.0, float(overlay_start_seconds))
    end_t = float(overlay_end_seconds)
    first_expr = f"between(t,0,{first_end_t:.3f})" if first_end_t > 0 else "0"
    if end_t >= 0:
        end_t = max(start_t, end_t)
        second_expr = f"between(t,{start_t:.3f},{end_t:.3f})"
    else:
        second_expr = f"gte(t,{start_t:.3f})"
    enable_expr = f"{first_expr}+{second_expr}"
    return (
        f"[0:v]fps={fps},scale={size}:{size}:force_original_aspect_ratio=decrease,"
        f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2:black[cta];"
        f"[1:v]scale=w={qr_width}:h={scale_h}[qr];"
        f"[cta][qr]overlay=x={x_expr}:y={y_expr}:"
        f"enable='{enable_expr}',format=yuv420p[v]"
    )


def build_concat_filter(size: int, fps: int) -> str:
    return (
        f"[0:v]fps={fps},scale={size}:{size}:force_original_aspect_ratio=decrease,"
        f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2:black,setsar=1[v0];"
        f"[1:v]fps={fps},scale={size}:{size}:force_original_aspect_ratio=decrease,"
        f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2:black,setsar=1[v1];"
        "[0:a]aformat=sample_rates=48000:channel_layouts=mono[a0];"
        "[1:a]aformat=sample_rates=48000:channel_layouts=mono[a1];"
        "[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]"
    )


def build_music_mix_filter(music_volume: float, voice_volume: float) -> str:
    vol = max(0.0, float(music_volume))
    voice_vol = max(0.0, float(voice_volume))
    return (
        f"[0:a]aformat=sample_rates=48000:channel_layouts=mono,volume={voice_vol:.3f}[voice];"
        f"[1:a]aformat=sample_rates=48000:channel_layouts=mono,volume={vol:.3f}[bg];"
        "[voice][bg]amix=inputs=2:duration=first:dropout_transition=0:weights='1 1':normalize=0,"
        "alimiter=limit=0.95[a]"
    )


def main() -> None:
    args = parse_args()

    if str(args.base_reel).strip():
        square_output_path = Path(args.base_reel).expanduser().resolve()
        if not square_output_path.exists():
            raise FileNotFoundError(f"Base reel not found: {square_output_path}")
        generated_square = False
    else:
        if str(args.video).strip():
            video_path = Path(args.video).expanduser().resolve()
        else:
            latest_video = find_latest_file(DEFAULT_DOWNLOADED_DIR, "*_video_01.mp4")
            if latest_video is None:
                raise FileNotFoundError(
                    "No input video found. Pass --video or place *_video_01.mp4 under "
                    f"{DEFAULT_DOWNLOADED_DIR}"
                )
            video_path = latest_video.resolve()

        if str(args.audio).strip():
            audio_path = Path(args.audio).expanduser().resolve()
        else:
            asin = infer_asin(video_path)
            audio_path = None
            if asin:
                audio_path = find_latest_file(DEFAULT_GEMINI_OUTPUTS_DIR, f"{asin}_promo_ml.wav")
            if audio_path is None:
                audio_path = find_latest_file(DEFAULT_GEMINI_OUTPUTS_DIR, "*_promo_ml.wav")
            if audio_path is None:
                raise FileNotFoundError(
                    "No promo audio found. Pass --audio or create *_promo_ml.wav under "
                    f"{DEFAULT_GEMINI_OUTPUTS_DIR}"
                )
            audio_path = audio_path.resolve()

        if str(args.output).strip():
            square_output_path = Path(args.output).expanduser().resolve()
        else:
            square_output_path = (DEFAULT_REEL_DIR / f"{video_path.stem}_square_reel.mp4").resolve()
        square_output_path.parent.mkdir(parents=True, exist_ok=True)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")
        generated_square = True

    ffmpeg_path = shutil.which("ffmpeg") or ""
    ffprobe_path = shutil.which("ffprobe") or ""
    if not ffmpeg_path or not ffprobe_path:
        raise RuntimeError("ffmpeg and ffprobe are required in PATH.")

    size = max(320, int(args.size))
    fps = max(1, int(args.fps))
    crf = max(0, int(args.crf))
    audio_volume = max(0.0, float(args.audio_volume))

    if generated_square:
        video_duration = get_duration_seconds(ffprobe_path, video_path)
        audio_duration = get_duration_seconds(ffprobe_path, audio_path)
        setpts_factor = audio_duration / max(0.001, video_duration)
        video_filter = build_video_filter(
            size=size,
            fps=fps,
            background=args.background,
            setpts_factor=setpts_factor,
        )
        audio_filter = f"volume={audio_volume:.3f}"

        square_cmd = [
            ffmpeg_path,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-filter_complex",
            video_filter,
            "-map",
            "[v]",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            str(args.preset),
            "-crf",
            str(crf),
            "-r",
            str(fps),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-af",
            audio_filter,
            "-t",
            f"{audio_duration:.3f}",
            "-movflags",
            "+faststart",
            str(square_output_path),
        ]

        print(f"Video: {video_path}")
        print(f"Audio: {audio_path}")
        print(f"Square output: {square_output_path}")
        print(f"Video duration: {video_duration:.3f}s")
        print(f"Audio duration: {audio_duration:.3f}s")
        print(f"Video setpts factor: {setpts_factor:.6f}")
        print(f"Output duration target (audio-led): {audio_duration:.3f}s")
        run_command(square_cmd)
        print("Square reel render complete.")
    else:
        print(f"Using existing base reel: {square_output_path}")

    if args.skip_cta_append:
        print("Skipping CTA append. Done.")
        return

    cta_video_path = Path(args.cta_video).expanduser().resolve()
    qr_image_path = Path(args.qr_image).expanduser().resolve()
    if not cta_video_path.exists():
        raise FileNotFoundError(f"CTA video not found: {cta_video_path}")
    if not qr_image_path.exists():
        raise FileNotFoundError(f"QR image not found: {qr_image_path}")

    cta_qr_scale = max(0.0, float(args.cta_qr_scale))
    cta_qr_width = max(1, int(args.cta_qr_width))
    if cta_qr_scale > 0:
        cta_qr_width = max(16, int(round(size * cta_qr_scale)))
    cta_qr_height = max(0, int(args.cta_qr_height))
    cta_margin_x = max(0, int(args.cta_margin_x))
    cta_margin_y = max(0, int(args.cta_margin_y))
    cta_overlay_first_end = max(0.0, float(args.cta_overlay_first_end))
    cta_overlay_start = max(0.0, float(args.cta_overlay_start))
    cta_overlay_end = float(args.cta_overlay_end)

    if str(args.complete_output).strip():
        complete_output_path = Path(args.complete_output).expanduser().resolve()
    else:
        complete_output_path = square_output_path.with_name(f"{square_output_path.stem}_complete.mp4")
    complete_output_path.parent.mkdir(parents=True, exist_ok=True)
    final_complete_output_path = resolve_writable_output_path(complete_output_path)
    if final_complete_output_path != complete_output_path:
        print(
            "Requested complete output is locked or not writable. "
            f"Using: {final_complete_output_path}"
        )

    if not has_audio_stream(ffprobe_path, square_output_path):
        raise RuntimeError(f"Base reel has no audio stream: {square_output_path}")
    if not has_audio_stream(ffprobe_path, cta_video_path):
        raise RuntimeError(f"CTA video has no audio stream: {cta_video_path}")

    final_music_path: Path | None = None
    if str(args.final_music).strip():
        final_music_path = Path(args.final_music).expanduser().resolve()
    elif args.no_final_music:
        final_music_path = None
    elif args.use_default_final_music or DEFAULT_FINAL_MUSIC.exists():
        final_music_path = DEFAULT_FINAL_MUSIC.resolve()
    if final_music_path is not None and not final_music_path.exists():
        raise FileNotFoundError(f"Final music not found: {final_music_path}")
    final_music_volume = max(0.0, float(args.final_music_volume))
    final_voice_volume = max(0.0, float(args.final_voice_volume))

    with tempfile.TemporaryDirectory(prefix="cta_qr_") as tmp_dir:
        cta_qr_path = Path(tmp_dir) / "cta_with_qr.mp4"
        cta_filter = build_cta_qr_filter(
            size=size,
            fps=fps,
            position=args.cta_position,
            margin_x=cta_margin_x,
            margin_y=cta_margin_y,
            qr_width=cta_qr_width,
            qr_height=cta_qr_height,
            overlay_first_end_seconds=cta_overlay_first_end,
            overlay_start_seconds=cta_overlay_start,
            overlay_end_seconds=cta_overlay_end,
        )

        cta_cmd = [
            ffmpeg_path,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(cta_video_path),
            "-i",
            str(qr_image_path),
            "-filter_complex",
            cta_filter,
            "-map",
            "[v]",
            "-map",
            "0:a:0",
            "-c:v",
            "libx264",
            "-preset",
            str(args.preset),
            "-crf",
            str(crf),
            "-r",
            str(fps),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-ac",
            "1",
            "-movflags",
            "+faststart",
            str(cta_qr_path),
        ]
        run_command(cta_cmd)

        concat_filter = build_concat_filter(size=size, fps=fps)
        concat_output_path = final_complete_output_path
        if final_music_path is not None:
            concat_output_path = Path(tmp_dir) / "complete_no_music.mp4"
        concat_cmd = [
            ffmpeg_path,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(square_output_path),
            "-i",
            str(cta_qr_path),
            "-filter_complex",
            concat_filter,
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-preset",
            str(args.preset),
            "-crf",
            str(crf),
            "-r",
            str(fps),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-movflags",
            "+faststart",
            str(concat_output_path),
        ]
        run_command(concat_cmd)

        if final_music_path is not None:
            music_filter = build_music_mix_filter(
                music_volume=final_music_volume,
                voice_volume=final_voice_volume,
            )
            music_cmd = [
                ffmpeg_path,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(concat_output_path),
                "-stream_loop",
                "-1",
                "-i",
                str(final_music_path),
                "-filter_complex",
                music_filter,
                "-map",
                "0:v:0",
                "-map",
                "[a]",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-ar",
                "48000",
                "-movflags",
                "+faststart",
                str(final_complete_output_path),
            ]
            run_command(music_cmd)

    print(f"Base reel: {square_output_path}")
    print(f"CTA source: {cta_video_path}")
    print(f"QR image: {qr_image_path}")
    print(
        "CTA QR overlay: "
        f"position={args.cta_position}, size={cta_qr_width}x"
        f"{cta_qr_height if cta_qr_height > 0 else 'auto'}, "
        f"margins=({cta_margin_x},{cta_margin_y}), window1=0-{cta_overlay_first_end:.2f}s, "
        f"window2_start={cta_overlay_start:.2f}s, "
        f"end={'video_end' if cta_overlay_end < 0 else f'{cta_overlay_end:.2f}s'}, "
        f"scale={cta_qr_scale:.3f}"
    )
    if final_music_path is not None:
        print(
            f"Final music: {final_music_path} "
            f"(music_volume={final_music_volume:.3f}, voice_volume={final_voice_volume:.3f})"
        )
    else:
        print("Final music: disabled")
    print(f"Complete output: {final_complete_output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
