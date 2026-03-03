#!/usr/bin/env python3
r"""
Check Amazon product pages for video, scrape title/description, and download video.

Default one-click run behavior:
  - Input:  <project_root>/data/product.json
  - Output: <project_root>/data/product_video_results.json
  - Videos: <project_root>/data/downloaded_videos

Run:
  python amazon_product_video_checker.py
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import html as html_lib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        if (parent / "requirements.txt").exists():
            return parent
    return here.parent


PROJECT_ROOT = _project_root()
DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_INPUT_FILE = str(DATA_DIR / "product.json")
DEFAULT_OUTPUT_FILE = str(DATA_DIR / "product_video_results.json")
DEFAULT_DOWNLOAD_DIR = str(DATA_DIR / "downloaded_videos")

DEFAULT_TIMEOUT_SECONDS = 25
DEFAULT_FFMPEG_TIMEOUT_SECONDS = 600
DEFAULT_WORKERS = max(4, min(24, (os.cpu_count() or 4) * 4))
DEFAULT_MAX_VIDEOS_PER_PRODUCT = 1
DEFAULT_DOWNLOAD_VIDEOS = True

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

ASIN_PATTERN = re.compile(r"/(?:dp|gp/product)/([A-Z0-9]{10})", re.IGNORECASE)
VIDEO_COUNT_PATTERN = re.compile(
    r'id=["\']videoCount["\'][^>]*>\s*([0-9]+)\s+VIDEOS?',
    re.IGNORECASE,
)
VIDEO_THUMB_PATTERN = re.compile(r"\bvideoThumbnail\b", re.IGNORECASE)
VIDEO_ACTION_PATTERN = re.compile(r"openVideoImmersiveView|chromeful-video", re.IGNORECASE)
VIDEO_JSON_URL_PATTERN = re.compile(r'"videoURL"\s*:\s*"([^"]+)"', re.IGNORECASE)
MEDIA_URL_PATTERN = re.compile(
    r"https?:[\\/]+[^\"'\s<>]+?\.(?:m3u8|mp4)(?:[^\"'\s<>]*)",
    re.IGNORECASE,
)


def extract_asin(url: str) -> str:
    match = ASIN_PATTERN.search(url or "")
    if not match:
        return ""
    return match.group(1).upper()


def clean_text(fragment: str) -> str:
    if not fragment:
        return ""
    text = re.sub(r"<script\b[^>]*>.*?</script>", " ", fragment, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html_lib.unescape(text)
    text = text.replace("\u200e", " ").replace("\u200f", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def unique_keep_order(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def load_product_urls(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8-sig")
    urls: list[str] = []
    payload: Any = None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, str):
                value = item.strip()
                if value:
                    urls.append(value)
            elif isinstance(item, dict):
                value = str(item.get("product_url", "")).strip() or str(item.get("url", "")).strip()
                if value:
                    urls.append(value)
    elif isinstance(payload, dict):
        if isinstance(payload.get("results"), list):
            for row in payload["results"]:
                if not isinstance(row, dict):
                    continue
                value = str(row.get("url", "")).strip() or str(row.get("product_url", "")).strip()
                if value:
                    urls.append(value)
        elif isinstance(payload.get("urls"), list):
            for value in payload["urls"]:
                if isinstance(value, str) and value.strip():
                    urls.append(value.strip())
    else:
        for line in raw.splitlines():
            value = line.strip()
            if value:
                urls.append(value)

    return unique_keep_order(urls)


def fetch_html(url: str, timeout_seconds: int) -> tuple[int, str]:
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Language": "en-IN,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Upgrade-Insecure-Requests": "1",
        },
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        status_code = int(response.getcode() or 200)
        html = response.read().decode("utf-8", errors="ignore")
        return status_code, html


def extract_alt_images_block(html: str) -> str:
    marker = 'id="altImages"'
    idx = html.find(marker)
    if idx < 0:
        return html
    return html[idx : idx + 300_000]


def extract_product_title(html: str) -> str:
    match = re.search(
        r'<span[^>]*id=["\']productTitle["\'][^>]*>(.*?)</span>',
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return clean_text(match.group(1))

    match = re.search(r"<title>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    if match:
        title = clean_text(match.group(1))
        title = re.sub(r"\s*:\s*Amazon\.in.*$", "", title, flags=re.IGNORECASE)
        return title
    return ""


def extract_product_description_points(html: str) -> list[str]:
    points: list[str] = []
    lower_html = html.lower()
    idx = lower_html.find('id="feature-bullets"')
    if idx < 0:
        idx = lower_html.find("id='feature-bullets'")

    if idx >= 0:
        block = html[idx : idx + 150_000]
        li_matches = re.findall(r"<li[^>]*>(.*?)</li>", block, flags=re.IGNORECASE | re.DOTALL)
        for fragment in li_matches:
            point = clean_text(fragment)
            if not point:
                continue
            if point.lower().startswith("make sure this fits"):
                continue
            points.append(point)

    if not points:
        meta_match = re.search(
            r'<meta[^>]*name=["\']description["\'][^>]*content=["\'](.*?)["\']',
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if meta_match:
            meta_desc = clean_text(meta_match.group(1))
            if meta_desc:
                points.append(meta_desc)

    return unique_keep_order(points)


def extract_product_description(html: str) -> str:
    points = extract_product_description_points(html)
    if points:
        return " | ".join(points)
    return ""


def normalize_media_url(raw_url: str) -> str:
    url = html_lib.unescape(raw_url or "").strip().strip('"\'')
    url = (
        url.replace("\\/", "/")
        .replace("\\u002F", "/")
        .replace("\\u0026", "&")
        .replace("\\u003D", "=")
        .replace("\\u003d", "=")
        .replace("\\u003F", "?")
        .replace("\\u003f", "?")
    )
    url = url.rstrip("\\")
    url = re.sub(r"[)\],]+$", "", url)
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return ""


def extract_video_urls(html: str) -> list[str]:
    urls: list[str] = []
    decoded = html_lib.unescape(html)

    for source in (decoded, html):
        for match in VIDEO_JSON_URL_PATTERN.findall(source):
            normalized = normalize_media_url(match)
            if normalized:
                urls.append(normalized)
        for match in MEDIA_URL_PATTERN.findall(source):
            normalized = normalize_media_url(match)
            if normalized:
                urls.append(normalized)

    filtered: list[str] = []
    for url in urls:
        lowered = url.lower()
        if ".m3u8" in lowered or ".mp4" in lowered:
            filtered.append(url)
    return unique_keep_order(filtered)


def detect_video_presence(html: str) -> tuple[bool, int, list[str]]:
    block = extract_alt_images_block(html)
    reasons: list[str] = []

    has_video_thumbnail = bool(VIDEO_THUMB_PATTERN.search(block))
    has_video_action = bool(VIDEO_ACTION_PATTERN.search(block))

    if has_video_thumbnail:
        reasons.append("videoThumbnail-class")
    if has_video_action:
        reasons.append("openVideoImmersiveView/chromeful-video")

    video_count = 0
    count_match = VIDEO_COUNT_PATTERN.search(block)
    if count_match:
        video_count = int(count_match.group(1))
        reasons.append("videoCount-label")
    elif has_video_thumbnail or has_video_action:
        video_count = max(1, len(re.findall(r"\bvideoThumbnail\b", block, flags=re.IGNORECASE)))

    media_urls = extract_video_urls(block)
    if media_urls:
        reasons.append("media-url-in-page")
        if video_count == 0:
            video_count = len(media_urls)

    has_video = has_video_thumbnail or has_video_action or video_count > 0 or bool(media_urls)
    return has_video, video_count, unique_keep_order(reasons)


def detect_bot_challenge(html: str) -> bool:
    lowered = html.lower()
    return (
        "captcha" in lowered
        or "enter the characters you see below" in lowered
        or "type the characters you see in this image" in lowered
        or "api-services-support@amazon.com" in lowered
    )


def download_video_with_ffmpeg(
    ffmpeg_path: str,
    media_url: str,
    output_file: Path,
    ffmpeg_timeout_seconds: int,
) -> tuple[bool, str]:
    if output_file.exists() and output_file.stat().st_size > 0:
        return True, "exists"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_path,
        "-y",
        "-nostdin",
        "-loglevel",
        "error",
        "-user_agent",
        USER_AGENT,
        "-i",
        media_url,
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(output_file),
    ]

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(60, int(ffmpeg_timeout_seconds)),
            check=False,
        )
    except subprocess.TimeoutExpired:
        if output_file.exists():
            output_file.unlink(missing_ok=True)
        return False, "ffmpeg timeout"

    if completed.returncode == 0 and output_file.exists() and output_file.stat().st_size > 0:
        return True, ""

    if output_file.exists():
        output_file.unlink(missing_ok=True)
    stderr = (completed.stderr or "").strip()
    if not stderr:
        stderr = f"ffmpeg exited with code {completed.returncode}"
    return False, stderr[:500]


def check_product(
    url: str,
    timeout_seconds: int,
    download_videos: bool,
    download_dir: Path,
    max_videos_per_product: int,
    ffmpeg_path: str,
    ffmpeg_timeout_seconds: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    asin = extract_asin(url)
    result: dict[str, Any] = {
        "url": url,
        "asin": asin,
        "title": "",
        "description": "",
        "has_video": False,
        "video_count": 0,
        "video_urls": [],
        "downloaded_videos": [],
        "download_errors": [],
        "http_status": None,
        "blocked": False,
        "error": "",
        "signals": [],
    }

    try:
        status_code, html = fetch_html(url, timeout_seconds)
        result["http_status"] = status_code

        if detect_bot_challenge(html):
            result["blocked"] = True
            result["error"] = "Bot challenge / captcha page returned"
            return result

        result["title"] = extract_product_title(html)
        result["description"] = extract_product_description(html)

        has_video, video_count, signals = detect_video_presence(html)
        video_urls = extract_video_urls(html)

        if video_urls and not has_video:
            has_video = True
        if has_video and video_count == 0:
            video_count = len(video_urls) if video_urls else 1

        result["has_video"] = has_video
        result["video_count"] = video_count
        result["video_urls"] = video_urls
        result["signals"] = signals

        if (
            download_videos
            and has_video
            and video_urls
            and max_videos_per_product > 0
            and ffmpeg_path
        ):
            product_key = asin if asin else "unknown"
            product_folder = download_dir / product_key
            for idx, media_url in enumerate(video_urls[:max_videos_per_product], start=1):
                output_file = product_folder / f"{product_key}_video_{idx:02d}.mp4"
                ok, message = download_video_with_ffmpeg(
                    ffmpeg_path=ffmpeg_path,
                    media_url=media_url,
                    output_file=output_file,
                    ffmpeg_timeout_seconds=ffmpeg_timeout_seconds,
                )
                if ok:
                    result["downloaded_videos"].append(str(output_file))
                else:
                    result["download_errors"].append(
                        {
                            "video_url": media_url,
                            "error": message,
                        }
                    )

        return result

    except HTTPError as exc:
        result["http_status"] = int(exc.code or 0)
        result["error"] = f"HTTPError: {exc.code}"
        return result
    except URLError as exc:
        result["error"] = f"URLError: {exc.reason}"
        return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check product video, scrape title/description, and download videos."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_FILE,
        help=(
            "Input path: JSON array of product URLs, JSON objects with product_url/url, "
            "or previous report (product_video_results.json)."
        ),
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILE,
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--download-dir",
        default=DEFAULT_DOWNLOAD_DIR,
        help="Folder where downloaded videos will be saved.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="HTTP timeout per product request.",
    )
    parser.add_argument(
        "--ffmpeg-timeout-seconds",
        type=int,
        default=DEFAULT_FFMPEG_TIMEOUT_SECONDS,
        help="Timeout per video download process.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Concurrent worker count.",
    )
    parser.add_argument(
        "--max-videos-per-product",
        type=int,
        default=DEFAULT_MAX_VIDEOS_PER_PRODUCT,
        help="Max videos to download per product (when video exists).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit: process first N URLs only (0 = all).",
    )
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        help="Optional direct product URL (can be passed multiple times).",
    )
    parser.add_argument(
        "--download-videos",
        dest="download_videos",
        action="store_true",
        help="Enable video download (default).",
    )
    parser.add_argument(
        "--no-download-videos",
        dest="download_videos",
        action="store_false",
        help="Disable video download; only scrape metadata.",
    )
    parser.set_defaults(download_videos=DEFAULT_DOWNLOAD_VIDEOS)
    return parser.parse_args()


def main() -> None:
    started = time.perf_counter()
    args = parse_cli_args()

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.url:
        urls = [u.strip() for u in args.url if u.strip()]
    else:
        input_path = Path(args.input).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        urls = load_product_urls(input_path)

    if not urls:
        raise ValueError("No product URLs found to check.")

    if args.limit > 0:
        urls = urls[: int(args.limit)]

    workers = max(1, int(args.workers))
    timeout_seconds = max(5, int(args.timeout_seconds))
    max_videos_per_product = max(0, int(args.max_videos_per_product))
    ffmpeg_timeout_seconds = max(60, int(args.ffmpeg_timeout_seconds))

    ffmpeg_path = shutil.which("ffmpeg") or ""
    download_videos = bool(args.download_videos)
    download_dir = Path(args.download_dir).expanduser().resolve()
    if download_videos:
        if not ffmpeg_path:
            raise RuntimeError("ffmpeg is required for video download but was not found in PATH.")
        download_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input URLs: {len(urls)}")
    print(f"Workers: {workers}")
    print(f"Timeout: {timeout_seconds}s")
    print(f"Download videos: {'yes' if download_videos else 'no'}")
    if download_videos:
        print(f"Download dir: {download_dir}")
        print(f"Max videos/product: {max_videos_per_product}")
    print(f"Output: {output_path}")

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                check_product,
                url,
                timeout_seconds,
                download_videos,
                download_dir,
                max_videos_per_product,
                ffmpeg_path,
                ffmpeg_timeout_seconds,
            ): url
            for url in urls
        }
        completed = 0
        for future in as_completed(futures):
            results.append(future.result())
            completed += 1
            if completed % 10 == 0 or completed == len(urls):
                print(f"Checked {completed}/{len(urls)}")

    order = {url: idx for idx, url in enumerate(urls)}
    results.sort(key=lambda row: order.get(str(row.get("url", "")), 10**9))

    summary = {
        "total": len(results),
        "with_video": sum(1 for row in results if row.get("has_video")),
        "without_video": sum(1 for row in results if not row.get("has_video")),
        "blocked": sum(1 for row in results if row.get("blocked")),
        "page_errors": sum(1 for row in results if row.get("error")),
        "downloaded_products": sum(1 for row in results if row.get("downloaded_videos")),
        "downloaded_videos": sum(len(row.get("downloaded_videos", [])) for row in results),
        "download_errors": sum(len(row.get("download_errors", [])) for row in results),
    }

    payload = {
        "summary": summary,
        "results": results,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - started
    print(f"Done. With video: {summary['with_video']} / {summary['total']}")
    print(
        "Downloaded videos: "
        f"{summary['downloaded_videos']} | download errors: {summary['download_errors']}"
    )
    print(
        "Blocked: "
        f"{summary['blocked']} | page errors: {summary['page_errors']}"
    )
    print(f"Execution time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
