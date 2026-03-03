#!/usr/bin/env python3
r"""
Upload first product video to Gemini, analyze visuals, then create Malayalam promo voice using Gemini TTS.

Default input:
  <project_root>/data/product_video_results.json

Default output folder:
  <project_root>/data/gemini_outputs

Required:
  - GEMINI_API_KEY or GEMINI_API_KEYS env var (or --api-key / --api-keys-file)
  - pip install google-genai
"""

from __future__ import annotations

import argparse
import base64
from difflib import SequenceMatcher
import json
import os
import re
import shutil
import subprocess
import time
import wave
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        if (parent / "requirements.txt").exists():
            return parent
    return here.parent


PROJECT_ROOT = _project_root()
DATA_DIR = PROJECT_ROOT / "data"
SECRETS_DIR = PROJECT_ROOT / "secrets"

DEFAULT_RESULTS_JSON = DATA_DIR / "product_video_results.json"
DEFAULT_OUTPUT_DIR = DATA_DIR / "gemini_outputs"
DEFAULT_KEYS_FILE = SECRETS_DIR / "gemini_keys.txt"
DEFAULT_ANALYSIS_MODEL = "gemini-2.5-flash"
DEFAULT_TTS_MODEL = "gemini-2.5-flash-preview-tts"
DEFAULT_VOICE = "Kore"
DEFAULT_CTA_TEXT = ""
DEFAULT_VIRAL_STYLE = (
    "curiosity hook + before-after contrast + problem-to-quick-solution + "
    "relatable everyday language + simple proof"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze first product video with Gemini and generate Malayalam promo voice via Gemini TTS."
    )
    parser.add_argument(
        "--results-json",
        default=str(DEFAULT_RESULTS_JSON),
        help="Path to product_video_results.json",
    )
    parser.add_argument(
        "--video-path",
        default="",
        help="Optional local video path. If empty, script uses first product's first downloaded video path.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where output files are written.",
    )
    parser.add_argument(
        "--api-key",
        action="append",
        default=[],
        help=(
            "Gemini API key. Repeat this option to pass multiple keys. "
            "Fallback order follows the order provided."
        ),
    )
    parser.add_argument(
        "--api-keys-file",
        default="",
        help=(
            "Optional text file with API keys (one per line, or comma-separated). "
            "If omitted, script tries GEMINI_KEYS_FILE env var, then default "
            "<project_root>/secrets/gemini_keys.txt."
        ),
    )
    parser.add_argument(
        "--analysis-model",
        default=DEFAULT_ANALYSIS_MODEL,
        help="Gemini model for video analysis/script generation.",
    )
    parser.add_argument(
        "--tts-model",
        default=DEFAULT_TTS_MODEL,
        help="Gemini TTS model.",
    )
    parser.add_argument(
        "--voice-name",
        default=DEFAULT_VOICE,
        help="Gemini prebuilt voice name: Kore",
    )
    parser.add_argument(
        "--viral-style",
        default=DEFAULT_VIRAL_STYLE,
        help="Creative direction for viral framing in the promo script.",

    )
    parser.add_argument(
        "--hook-line",
        default="",
        help=(
            "Optional opening hook line override. "
            "If empty, Gemini auto-generates a hook line."
        ),
    )
    parser.add_argument(
        "--cta-text",
        default=DEFAULT_CTA_TEXT,
        help="Optional CTA sentence for the final narration line. Leave empty to disable CTA text.",
    )
    parser.add_argument(
        "--voice-timing-mode",
        choices=["natural_no_stretch", "global_stretch"],
        default="natural_no_stretch",
        help=(
            "How to match audio to video length. "
            "'natural_no_stretch' keeps natural speech speed by default and only applies "
            "limited speed correction when audio is too long. "
            "'global_stretch' applies one global speed fit to target duration."
        ),
    )
    parser.add_argument(
        "--audio-overrun-seconds",
        type=float,
        default=2.0,
        help="Preferred extra seconds over video duration for generated narration.",
    )
    parser.add_argument(
        "--min-audio-overrun-seconds",
        type=float,
        default=0.8,
        help="Minimum allowed extra seconds over video duration.",
    )
    parser.add_argument(
        "--max-audio-overrun-seconds",
        type=float,
        default=2.8,
        help="Maximum allowed extra seconds over video duration.",
    )
    parser.add_argument(
        "--max-global-speedup-factor",
        type=float,
        default=1.25,
        help=(
            "When voice_timing_mode=global_stretch, cap speed-up factor. "
            "1.0 means never speed up voice."
        ),
    )
    parser.add_argument(
        "--max-natural-speedup-factor",
        type=float,
        default=1.20,
        help=(
            "When voice_timing_mode=natural_no_stretch and audio is too long, cap corrective speed-up."
        ),
    )
    parser.add_argument(
        "--tts-speed-factor",
        type=float,
        default=1.08,
        help="Baseline narration speed-up after TTS generation (1.0 keeps original speed).",
    )
    parser.add_argument(
        "--scene-count",
        type=int,
        default=3,
        help="Approximate number of scene narration segments to generate.",
    )
    parser.add_argument(
        "--min-scene-seconds",
        type=float,
        default=1.2,
        help="Minimum duration per narration scene after normalization.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=2,
        help="Polling interval while waiting for uploaded video processing.",
    )
    parser.add_argument(
        "--file-timeout-seconds",
        type=int,
        default=600,
        help="Timeout waiting for Gemini file processing to become ACTIVE.",
    )
    parser.add_argument(
        "--max-key-cycles",
        type=int,
        default=1,
        help="How many full rotations through all keys to attempt when rate-limited.",
    )
    parser.add_argument(
        "--rate-limit-wait-seconds",
        type=int,
        default=3,
        help="Base retry wait (seconds) for retryable API errors.",
    )
    parser.add_argument(
        "--api-call-max-retries",
        type=int,
        default=4,
        help="Max retries per API call when rate-limited.",
    )
    parser.add_argument(
        "--max-retry-wait-seconds",
        type=float,
        default=12.0,
        help="If Gemini asks to wait longer than this, skip retry and rotate to next key.",
    )
    parser.add_argument(
        "--tts-request-delay-seconds",
        type=float,
        default=0.0,
        help="Legacy setting. Not used in one-shot TTS mode.",
    )
    parser.add_argument(
        "--pre-request-sleep-seconds",
        type=float,
        default=0.0,
        help="Optional extra sleep before retry attempts (0 disables proactive sleep).",
    )
    parser.add_argument(
        "--key-rotate-sleep-seconds",
        type=float,
        default=1.5,
        help="Sleep before switching to the next API key after a retryable error.",
    )
    parser.add_argument(
        "--keep-debug-files",
        action="store_true",
        help=(
            "Keep analysis/debug artifacts and intermediate wav files. "
            "By default only final script and final voice wav are kept."
        ),
    )
    return parser.parse_args()


def load_first_product(results_path: Path) -> dict[str, Any]:
    payload = json.loads(results_path.read_text(encoding="utf-8-sig"))
    results = payload.get("results", [])
    if not isinstance(results, list) or not results:
        raise ValueError("No products found in results JSON.")

    first = results[0]
    if not isinstance(first, dict):
        raise ValueError("First product entry is not an object.")
    return first


def load_product_by_asin(results_path: Path, asin: str) -> dict[str, Any] | None:
    target = str(asin or "").strip().upper()
    if not target:
        return None

    payload = json.loads(results_path.read_text(encoding="utf-8-sig"))
    results = payload.get("results", [])
    if not isinstance(results, list):
        return None

    for item in results:
        if not isinstance(item, dict):
            continue
        value = str(item.get("asin", "")).strip().upper()
        if value == target:
            return item
    return None


def infer_asin_from_video_path(video_path: Path) -> str:
    raw = f"{video_path.parent.name} {video_path.stem}".upper()
    match = re.search(r"\b([A-Z0-9]{10})\b", raw)
    return match.group(1).upper() if match else ""


def resolve_video_path(first_product: dict[str, Any], cli_video_path: str) -> Path:
    if cli_video_path.strip():
        path = Path(cli_video_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"--video-path not found: {path}")
        return path

    downloaded = first_product.get("downloaded_videos", [])
    if not isinstance(downloaded, list) or not downloaded:
        raise ValueError(
            "First product has no downloaded_videos. Pass --video-path with a local file."
        )

    path = Path(str(downloaded[0])).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Downloaded video path from results JSON was not found: {path}"
        )
    return path


def split_key_tokens(raw: str) -> list[str]:
    tokens = re.split(r"[\s,;]+", raw or "")
    return [t.strip() for t in tokens if t and t.strip()]


def unique_keep_order(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def load_api_keys(args: argparse.Namespace) -> list[str]:
    keys: list[str] = []

    for value in args.api_key or []:
        keys.extend(split_key_tokens(str(value)))

    env_keys_file = os.getenv("GEMINI_KEYS_FILE", "").strip()
    keys_file_candidates: list[Path] = []

    if str(args.api_keys_file).strip():
        keys_file_candidates.append(Path(str(args.api_keys_file)).expanduser().resolve())
    elif env_keys_file:
        keys_file_candidates.append(Path(env_keys_file).expanduser().resolve())
    else:
        keys_file_candidates.append(DEFAULT_KEYS_FILE.expanduser().resolve())

    for keys_file in keys_file_candidates:
        if not keys_file.exists():
            if str(args.api_keys_file).strip() or env_keys_file:
                raise FileNotFoundError(f"API keys file not found: {keys_file}")
            continue
        keys.extend(split_key_tokens(keys_file.read_text(encoding="utf-8-sig")))

    keys.extend(split_key_tokens(os.getenv("GEMINI_API_KEYS", "")))
    keys.extend(split_key_tokens(os.getenv("GEMINI_API_KEY", "")))

    keys = unique_keep_order(keys)
    if not keys:
        raise RuntimeError(
            "Missing API keys. Provide --api-key (repeatable), --api-keys-file, "
            "GEMINI_KEYS_FILE, GEMINI_API_KEYS, GEMINI_API_KEY, or create "
            f"default file: {DEFAULT_KEYS_FILE}"
        )
    return keys


def mask_api_key(api_key: str) -> str:
    key = (api_key or "").strip()
    if len(key) <= 8:
        return "*" * max(1, len(key))
    return f"{key[:4]}...{key[-4:]}"


def is_rate_limit_error(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    status = str(getattr(exc, "status", "") or "").upper()
    message = str(getattr(exc, "message", "") or "").upper()
    text = f"{type(exc).__name__}: {exc}".upper()

    if code == 429:
        return True

    signals = [status, message, text]
    tokens = (
        "RESOURCE_EXHAUSTED",
        "RATE_LIMIT",
        "TOO MANY REQUESTS",
        "QUOTA",
        "429",
    )
    return any(token in signal for signal in signals for token in tokens)


def is_retriable_key_error(exc: Exception) -> bool:
    if is_rate_limit_error(exc):
        return True
    if is_transient_api_error(exc):
        return True
    text = f"{type(exc).__name__}: {exc}".upper()
    return "TTS RETURNED NO AUDIO" in text or "NO AUDIO BYTES RETURNED BY TTS MODEL" in text


def is_daily_quota_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".upper()
    return (
        "GENERATEREQUESTSPERDAY" in text
        or "PERDAY" in text
        or "QUOTAID': 'GENERATEREQUESTSPERDAY" in text
    )


def is_transient_api_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".upper()
    tokens = (
        "REMOTEPROTOCOLERROR",
        "SERVER DISCONNECTED",
        "READTIMEOUT",
        "CONNECTTIMEOUT",
        "TIMEOUT",
        "CONNECTIONRESETERROR",
        "CONNECTION ABORTED",
        "NETWORK",
        "UNAVAILABLE",
        "503",
        "500",
        "502",
        "504",
    )
    return any(token in text for token in tokens)


def extract_retry_delay_seconds(exc: Exception, default_seconds: float = 3.0) -> float:
    text = f"{exc}"
    patterns = [
        r"retry in ([0-9]+(?:\.[0-9]+)?)s",
        r"retryDelay['\"]?\s*:\s*'([0-9]+(?:\.[0-9]+)?)s'",
        r"retryDelay['\"]?\s*:\s*\"([0-9]+(?:\.[0-9]+)?)s\"",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                return max(0.5, float(match.group(1)))
            except ValueError:
                continue
    return max(0.5, float(default_seconds))


def sleep_with_log(seconds: float, reason: str) -> None:
    delay = max(0.0, float(seconds))
    if delay <= 0:
        return
    print(f"Sleeping {delay:.1f}s: {reason}")
    time.sleep(delay)


def has_link_in_bio_signal(text: str) -> bool:
    lowered = str(text or "").lower()
    signals = (
        "link in bio",
        "check the link in bio",
        "bio link",
        "ലിങ്ക് ഇൻ ബയോ",
        "ബയോയിലെ ലിങ്ക്",
        "ബയോ യിലെ ലിങ്ക്",
        "bioയിലെ ലിങ്ക്",
        "bio യിലെ ലിങ്ക്",
    )
    return any(token in lowered for token in signals)


def remove_link_in_bio_sentences(text: str) -> str:
    body = str(text or "").strip()
    if not body:
        return body

    parts = [p.strip() for p in re.split(r"(?<=[\.\!\?।])\s+", body) if p.strip()]
    filtered = [p for p in parts if not has_link_in_bio_signal(p)]
    if not filtered:
        return ""
    return " ".join(filtered).strip()


def ensure_cta_in_text(text: str, cta_text: str) -> str:
    body = str(text or "").strip()
    cta = str(cta_text or "").strip()
    if not cta:
        return body

    body_lower = body.lower()
    cta_lower = cta.lower()
    if cta_lower in body_lower:
        return body

    if has_link_in_bio_signal(body):
        base = remove_link_in_bio_sentences(body)
        if base and base[-1] not in ".!?":
            base = base + "."
        return f"{base} {cta}".strip()

    if body and body[-1] not in ".!?":
        body = body + "."
    return f"{body} {cta}".strip()


def ensure_cta_in_scene_segments(scene_segments: list[dict[str, Any]], cta_text: str) -> None:
    if not scene_segments:
        return
    last = scene_segments[-1]
    line = str(last.get("narration_malayalam", "")).strip()
    last["narration_malayalam"] = ensure_cta_in_text(line, cta_text)


def has_hook_signal(text: str) -> bool:
    lowered = str(text or "").lower()
    signals = (
        "stop scrolling",
        "nobody talks",
        "nobody tells",
        "i wish i knew",
        "changed everything",
        "mistake",
        "before",
        "after",
        "truth",
    )
    return any(token in lowered for token in signals)


def normalize_text_for_match(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    cleaned = re.sub(r"[\"'`“”‘’]", "", cleaned)
    return cleaned.strip(" .,!?:;|")


def is_repetitive_sentence_pair(current_norm: str, previous_norm: str) -> bool:
    a = str(current_norm or "").strip()
    b = str(previous_norm or "").strip()
    if not a or not b:
        return False
    if a == b:
        return True

    a_words = a.split()
    b_words = b.split()
    short_words, long_words = (a_words, b_words) if len(a_words) <= len(b_words) else (b_words, a_words)
    short_text = " ".join(short_words)
    long_text = " ".join(long_words)

    # Treat prefix-extension phrasing as repetition, e.g. repeated opener with extra tail words.
    if len(short_words) >= 4 and long_text.startswith(short_text):
        coverage = len(short_words) / max(1, len(long_words))
        if coverage >= 0.45:
            return True

    # Catch repeated opener phrasing even when one or two words are changed.
    common_prefix = 0
    for left, right in zip(a_words, b_words):
        if left != right:
            break
        common_prefix += 1
    if common_prefix >= 4:
        prefix_ratio = common_prefix / max(1, min(len(a_words), len(b_words)))
        if prefix_ratio >= 0.55:
            return True

    if min(len(a_words), len(b_words)) < 3:
        return False

    seq_ratio = SequenceMatcher(None, a, b).ratio()
    if seq_ratio >= 0.90:
        return True

    a_set = set(a_words)
    b_set = set(b_words)
    overlap = len(a_set & b_set) / max(1, min(len(a_set), len(b_set)))
    if overlap >= 0.88:
        return True
    return seq_ratio >= 0.80 and overlap >= 0.68


def remove_repetitive_sentences(text: str, recent_window: int = 10) -> tuple[str, int]:
    body = re.sub(r"\s+", " ", str(text or "")).strip()
    if not body:
        return "", 0

    sentences = split_spoken_sentences(body)
    if not sentences:
        return body, 0

    kept_raw: list[str] = []
    kept_norm: list[str] = []
    removed = 0
    window = max(1, int(recent_window))

    for sentence in sentences:
        raw = re.sub(r"\s+", " ", str(sentence or "")).strip()
        if not raw:
            continue
        norm = normalize_text_for_match(raw)
        if not norm:
            continue

        dropped = False
        for prev_norm in kept_norm[-window:]:
            if is_repetitive_sentence_pair(norm, prev_norm):
                removed += 1
                dropped = True
                break
        if dropped:
            continue

        kept_raw.append(raw)
        kept_norm.append(norm)

    if not kept_raw:
        return body, 0
    return " ".join(kept_raw).strip(), removed


def ensure_hook_in_text(text: str, hook_line: str) -> str:
    body = str(text or "").strip()
    hook = str(hook_line or "").strip()
    if not hook:
        return body

    if not body:
        return hook

    hook_norm = normalize_text_for_match(hook)
    body_norm = normalize_text_for_match(body)
    if hook_norm and hook_norm in body_norm:
        return body

    first_part = split_spoken_sentences(body)[0].strip() if body else ""
    first_norm = normalize_text_for_match(first_part)
    if first_norm == hook_norm or is_repetitive_sentence_pair(first_norm, hook_norm):
        return body

    # If the first sentence already starts with the same opener phrase, don't prepend hook again.
    hook_words = hook_norm.split()
    first_words = first_norm.split()
    common_prefix = 0
    for left, right in zip(hook_words, first_words):
        if left != right:
            break
        common_prefix += 1
    if common_prefix >= 3:
        return body

    if first_part and has_hook_signal(first_part):
        return body

    hook_prefix = hook if hook[-1] in ".!?" else f"{hook}."
    return f"{hook_prefix} {body}".strip()


def ensure_hook_in_scene_segments(scene_segments: list[dict[str, Any]], hook_line: str) -> None:
    if not scene_segments:
        return
    first = scene_segments[0]
    line = str(first.get("narration_malayalam", "")).strip()
    first["narration_malayalam"] = ensure_hook_in_text(line, hook_line)


def generate_hook_line_from_gemini(
    *,
    client: Any,
    args: argparse.Namespace,
    product: dict[str, Any],
    analysis_text: str,
) -> str:
    asin = str(product.get("asin", "")).strip()
    title = str(product.get("title", "")).strip()
    description = str(product.get("description", "")).strip()
    url = str(product.get("url", "")).strip()

    response = call_api_with_rate_limit_retry(
        operation_name="hook.generate_content",
        func=lambda: client.models.generate_content(
            model=args.analysis_model,
            contents=(
                "Generate one Malayalam hook line for a social reel. Return ONE line only.\n"
                "Rules:\n"
                "- Must be scroll-stopping in first 2 seconds.\n"
                "- Must fit this product's visuals and benefits.\n"
                "- Use simple spoken Malayalam.\n"
                "- Do not include CTA.\n\n"
                f"Viral style direction: {args.viral_style}\n"
                f"ASIN: {asin}\n"
                f"Title: {title}\n"
                f"Description: {description}\n"
                f"URL: {url}\n\n"
                f"Visual analysis:\n{analysis_text}"
            ),
        ),
        max_attempts=args.api_call_max_retries,
        retry_wait_seconds=args.rate_limit_wait_seconds,
        max_retry_wait_seconds=args.max_retry_wait_seconds,
        pre_request_sleep_seconds=args.pre_request_sleep_seconds,
    )
    raw = get_response_text(response).strip()
    if not raw:
        return ""

    lines = [ln.strip().strip('"').strip("'") for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return ""
    return lines[0]


def get_response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text

    chunks: list[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                chunks.append(part_text.strip())
    return "\n".join(chunks).strip()


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def find_first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escaping = False
    for i, ch in enumerate(text[start:], start=start):
        if in_string:
            if escaping:
                escaping = False
                continue
            if ch == "\\":
                escaping = True
                continue
            if ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return ""


def parse_json_from_text(text: str) -> dict[str, Any] | None:
    candidates = [text.strip(), strip_code_fence(text)]
    extracted = find_first_json_object(text)
    if extracted:
        candidates.append(extracted)

    for candidate in candidates:
        if not candidate:
            continue
        try:
            loaded = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            return loaded
    return None


def wait_until_file_active(
    client: Any,
    uploaded_file: Any,
    poll_seconds: int,
    timeout_seconds: int,
    rate_limit_wait_seconds: float = 3.0,
    max_retry_wait_seconds: float = 12.0,
) -> Any:
    start = time.time()
    current = uploaded_file

    while True:
        state = getattr(current, "state", None)
        state_name = getattr(state, "name", "") if state is not None else ""
        state_name = str(state_name or "").upper()

        if state_name == "ACTIVE":
            return current
        if state_name in {"FAILED", "ERROR", "CANCELLED"}:
            raise RuntimeError(f"Uploaded file entered terminal state: {state_name}")

        if time.time() - start > max(30, int(timeout_seconds)):
            raise TimeoutError("Timed out waiting for Gemini file processing.")

        time.sleep(max(1, int(poll_seconds)))
        try:
            current = client.files.get(name=current.name)
        except Exception as exc:
            if not (is_rate_limit_error(exc) or is_transient_api_error(exc)):
                raise
            if is_rate_limit_error(exc) and is_daily_quota_error(exc):
                raise
            if is_rate_limit_error(exc):
                wait_for = max(
                    float(rate_limit_wait_seconds),
                    extract_retry_delay_seconds(exc, float(rate_limit_wait_seconds)),
                )
                if wait_for > float(max_retry_wait_seconds):
                    raise
            else:
                wait_for = max(1.0, float(rate_limit_wait_seconds))
            print(f"files.get retryable error while polling. Waiting {wait_for:.1f}s...")
            time.sleep(wait_for)


def compute_word_targets_for_duration(target_audio_duration_seconds: float) -> tuple[int, int, int]:
    duration = max(5.0, float(target_audio_duration_seconds))
    # Target normal conversational speed (not slow narration) to stay near video+small overrun.
    target_words_hint = int(max(95, min(280, round(duration * 2.45))))
    target_words_min = int(max(85, min(260, round(target_words_hint * 0.92))))
    target_words_max = int(max(target_words_min + 18, min(320, round(target_words_hint * 1.08))))
    return target_words_min, target_words_max, target_words_hint


def build_analysis_prompt(
    product: dict[str, Any],
    video_duration_seconds: float,
    target_audio_duration_seconds: float,
    scene_count: int,
    viral_style: str,
    cta_text: str,
    hook_line: str,
) -> str:
    url = str(product.get("url", "")).strip()
    title = str(product.get("title", "")).strip()
    description = str(product.get("description", "")).strip()
    asin = str(product.get("asin", "")).strip()
    cta = str(cta_text or "").strip()
    hook_preference = str(hook_line or "").strip() or "Auto-generate from product data and visuals."
    target_words_min, target_words_max, target_words_hint = compute_word_targets_for_duration(
        target_audio_duration_seconds
    )
    cta_summary_line = (
        f"- Optional CTA to include in final line: {cta}"
        if cta
        else "- CTA disabled: do not include CTA text in narration."
    )
    structure_suffix = " -> CTA" if cta else ""
    cta_rules = (
        "- Last scene narration must include the CTA.\n"
        "- short_cta_malayalam should contain the CTA line.\n"
        if cta
        else (
            "- Do not include CTA phrases like 'link in bio' in promo_script_malayalam.\n"
            "- short_cta_malayalam must be an empty string.\n"
        )
    )

    return (
        "You are a product video creative strategist.\n"
        "Analyze the uploaded product video visuals and build a colloquial Malayalam promotion script.\n\n"
        "Product metadata:\n"
        f"- ASIN: {asin}\n"
        f"- URL: {url}\n"
        f"- Title: {title}\n"
        f"- Description: {description}\n\n"
        f"- Video duration (seconds): {video_duration_seconds:.3f}\n"
        f"- Target spoken duration (seconds): {target_audio_duration_seconds:.3f}\n"
        f"- Target script word range: {target_words_min}-{target_words_max} words\n"
        f"- Preferred script length: around {target_words_hint} words\n"
        f"- Requested scene count: {max(1, int(scene_count))}\n\n"
        f"- Viral style direction: {viral_style}\n"
        f"- Preferred opening hook style: {hook_preference}\n"
        f"{cta_summary_line}\n\n"
        "Return ONLY valid JSON with this exact schema:\n"
        "{\n"
        '  "visual_summary_english": "string",\n'
        '  "visual_highlights": ["string", "string"],\n'
        '  "hook_line_malayalam": "short scroll-stopping opening line",\n'
        '  "target_audience_malayalam": "string",\n'
        '  "key_benefits_malayalam": ["string", "string"],\n'
        '  "promo_script_malayalam": "spoken Malayalam script in Unicode Malayalam, within target word range",\n'
        '  "short_cta_malayalam": "string",\n'
        '  "scene_segments": [\n'
        "    {\n"
        '      "start_sec": 0.0,\n'
        '      "end_sec": 0.0,\n'
        '      "visual_focus_english": "what is visible in this scene",\n'
        '      "narration_malayalam": "Malayalam line matching this exact scene"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- promo_script_malayalam must be natural Malayalam speech for ad narration.\n"
        "- First line must be a strong hook in first 2 seconds.\n"
        f"- Structure should be: Hook -> relatable problem -> quick benefit/solution -> proof{structure_suffix}.\n"
        "- Choose exactly ONE primary viral format: curiosity hook, before-after transformation, problem-to-quick-solution, relatable humor, myth-busting, POV, or comparison.\n"
        "- Optionally blend ONE secondary format only if it helps clarity. Never stack more than 2 formats.\n"
        "- Use viral triggers intentionally: curiosity, emotion, relatability, usefulness, surprise, simplicity.\n"
        "- Keep wording simple, emotional, and relatable for social reels.\n"
        "- Script must fit the target word range and stay near the preferred length.\n"
        "- Spoken pace must be normal conversational speed; do NOT write slow-paced narration.\n"
        "- Avoid ellipses (...), repeated filler words, repeated sentence ideas, stage directions, and dramatic pause text.\n"
        "- Do not repeat the same hook line, claim, or benefit phrasing more than once.\n"
        "- Hook line must appear exactly once in the full script.\n"
        "- Never ask the same question twice, even with slightly different wording.\n"
        "- Do not start two sentences with the same first 3+ words.\n"
        "- Every sentence must add new information (no paraphrase repeats).\n"
        "- Self-check before final answer: rewrite until duplicate or near-duplicate sentence count is zero.\n"
        "- Hook should feel scroll-stopping in first 2 seconds and be concise.\n"
        f"- Use this flow: Hook (0-2s) -> relatable problem -> quick value/solution -> visual proof/result{structure_suffix}.\n"
        "- Keep promo_script_malayalam as one continuous paragraph (not multiple lines).\n"
        "- CRITICAL timing rule: narration must finish about 1-3 seconds after the video ends.\n"
        "- Never return a short script that would finish before video end.\n"
        "- Mention clear benefits seen or implied by visuals.\n"
        f"{cta_rules}"
        "- scene_segments must follow actual visual order in the video.\n"
        "- start_sec/end_sec must be numeric and cover the full video from 0 to duration.\n"
        "- narration_malayalam must specifically match each scene's visual action/product usage.\n"
        "- Do not include markdown, code fences, or extra keys."
    )


def extract_audio_bytes(response: Any) -> bytes:
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            inline_data = getattr(part, "inline_data", None)
            if inline_data is None:
                continue
            raw = getattr(inline_data, "data", None)
            if isinstance(raw, (bytes, bytearray)):
                return bytes(raw)
            if isinstance(raw, str) and raw.strip():
                return base64.b64decode(raw)
    raise RuntimeError("No audio bytes returned by TTS model.")


def write_wav(path: Path, pcm_data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(pcm_data)


def run_command(cmd: list[str]) -> str:
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        detail = stderr or stdout or f"exit code {completed.returncode}"
        raise RuntimeError(f"Command failed: {' '.join(cmd)} | {detail[:700]}")
    return (completed.stdout or "").strip()


def get_media_duration_seconds(ffprobe_path: str, media_path: Path) -> float:
    output = run_command(
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
    if not output:
        raise RuntimeError(f"Could not read media duration: {media_path}")
    try:
        duration = float(output.strip().splitlines()[-1].strip())
    except ValueError as exc:
        raise RuntimeError(f"Invalid duration from ffprobe for {media_path}: {output}") from exc
    if duration <= 0:
        raise RuntimeError(f"Non-positive media duration for {media_path}: {duration}")
    return duration


def get_wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        frame_rate = wf.getframerate()
    if frame_rate <= 0:
        raise RuntimeError(f"Invalid wav framerate in {path}")
    return float(frames) / float(frame_rate)


def split_spoken_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned:
        return []

    parts = [p.strip() for p in re.split(r"(?<=[\.\!\?।])\s+", cleaned) if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in re.split(r"\s*[,:;]\s*", cleaned) if p.strip()]
    return parts or [cleaned]


def build_fallback_scene_segments(
    promo_script: str,
    video_duration_seconds: float,
    scene_count: int,
) -> list[dict[str, Any]]:
    lines = split_spoken_sentences(promo_script)
    if not lines:
        lines = ["ഈ ഉൽപ്പന്നം ഇപ്പോൾ തന്നെ പരീക്ഷിക്കൂ."]

    wanted = max(2, int(scene_count))
    if len(lines) > wanted:
        group_size = max(1, (len(lines) + wanted - 1) // wanted)
        grouped: list[str] = []
        for i in range(0, len(lines), group_size):
            grouped.append(" ".join(lines[i : i + group_size]).strip())
        lines = grouped

    if len(lines) < 2 and video_duration_seconds >= 6:
        lines = [lines[0].strip(), "ഇത് ഇന്ന് തന്നെ ഓർഡർ ചെയ്യൂ.".strip()]

    n = max(1, len(lines))
    segment_duration = video_duration_seconds / n
    segments: list[dict[str, Any]] = []
    start = 0.0
    for i, line in enumerate(lines, start=1):
        end = video_duration_seconds if i == n else start + segment_duration
        segments.append(
            {
                "scene_index": i,
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "duration_sec": round(end - start, 3),
                "visual_focus_english": "",
                "narration_malayalam": line.strip(),
            }
        )
        start = end
    return segments


def to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_scene_segments(
    raw_segments: Any,
    video_duration_seconds: float,
    promo_script: str,
    scene_count: int,
    min_scene_seconds: float,
) -> list[dict[str, Any]]:
    if not isinstance(raw_segments, list):
        return build_fallback_scene_segments(promo_script, video_duration_seconds, scene_count)

    cleaned: list[dict[str, Any]] = []
    for item in raw_segments:
        if not isinstance(item, dict):
            continue
        line = (
            str(item.get("narration_malayalam", "")).strip()
            or str(item.get("script_line_malayalam", "")).strip()
            or str(item.get("line", "")).strip()
            or str(item.get("text", "")).strip()
        )
        if not line:
            continue
        cleaned.append(
            {
                "start_sec": to_float(item.get("start_sec")),
                "end_sec": to_float(item.get("end_sec")),
                "visual_focus_english": str(item.get("visual_focus_english", "")).strip(),
                "narration_malayalam": line,
            }
        )

    if not cleaned:
        return build_fallback_scene_segments(promo_script, video_duration_seconds, scene_count)

    if any(item["start_sec"] is not None for item in cleaned):
        cleaned.sort(
            key=lambda row: row["start_sec"] if row["start_sec"] is not None else 10**9
        )

    n = len(cleaned)
    min_sec = max(0.4, float(min_scene_seconds))
    if n * min_sec > video_duration_seconds:
        min_sec = max(0.1, video_duration_seconds / n)

    weights: list[float] = []
    for row in cleaned:
        start = row["start_sec"]
        end = row["end_sec"]
        model_duration = 0.0
        if start is not None and end is not None and end > start:
            model_duration = end - start
        line_words = max(1, len(str(row["narration_malayalam"]).split()))
        speech_guess = max(0.8, line_words * 0.35)
        weights.append(max(model_duration, speech_guess))

    total_weight = sum(weights) or float(n)
    variable_budget = max(0.0, video_duration_seconds - (n * min_sec))

    segments: list[dict[str, Any]] = []
    cursor = 0.0
    for i, row in enumerate(cleaned, start=1):
        weight_share = weights[i - 1] / total_weight
        duration = min_sec + (variable_budget * weight_share)
        if i == n:
            end = video_duration_seconds
        else:
            end = min(video_duration_seconds, cursor + duration)
        segments.append(
            {
                "scene_index": i,
                "start_sec": round(cursor, 3),
                "end_sec": round(end, 3),
                "duration_sec": round(end - cursor, 3),
                "visual_focus_english": row["visual_focus_english"],
                "narration_malayalam": row["narration_malayalam"],
            }
        )
        cursor = end

    if segments:
        segments[-1]["end_sec"] = round(video_duration_seconds, 3)
        last_start = float(segments[-1]["start_sec"])
        segments[-1]["duration_sec"] = round(video_duration_seconds - last_start, 3)

    return segments


def compress_scene_segments(
    segments: list[dict[str, Any]],
    max_segments: int,
    video_duration_seconds: float,
) -> list[dict[str, Any]]:
    wanted = max(1, int(max_segments))
    if len(segments) <= wanted:
        return segments

    group_size = max(1, (len(segments) + wanted - 1) // wanted)
    grouped: list[dict[str, Any]] = []
    cursor = 0.0

    for i in range(0, len(segments), group_size):
        chunk = segments[i : i + group_size]
        if not chunk:
            continue

        duration = sum(float(item.get("duration_sec", 0.0) or 0.0) for item in chunk)
        if duration <= 0:
            start_guess = float(chunk[0].get("start_sec", 0.0) or 0.0)
            end_guess = float(chunk[-1].get("end_sec", start_guess) or start_guess)
            duration = max(0.1, end_guess - start_guess)

        visual_parts = [
            str(item.get("visual_focus_english", "")).strip()
            for item in chunk
            if str(item.get("visual_focus_english", "")).strip()
        ]
        line_parts = [
            str(item.get("narration_malayalam", "")).strip()
            for item in chunk
            if str(item.get("narration_malayalam", "")).strip()
        ]
        line = " ".join(line_parts).strip() or "ഈ ഉൽപ്പന്നം ഇപ്പോൾ തന്നെ പരീക്ഷിക്കൂ."

        end = min(video_duration_seconds, cursor + duration)
        grouped.append(
            {
                "scene_index": len(grouped) + 1,
                "start_sec": round(cursor, 3),
                "end_sec": round(end, 3),
                "duration_sec": round(end - cursor, 3),
                "visual_focus_english": " | ".join(visual_parts),
                "narration_malayalam": line,
            }
        )
        cursor = end

    if not grouped:
        return segments[:wanted]

    if len(grouped) > wanted:
        grouped = grouped[:wanted]

    grouped[-1]["end_sec"] = round(video_duration_seconds, 3)
    last_start = float(grouped[-1]["start_sec"])
    grouped[-1]["duration_sec"] = round(video_duration_seconds - last_start, 3)
    return grouped


def build_atempo_chain(speed_factor: float) -> str:
    factor = float(speed_factor)
    if factor <= 0:
        raise ValueError(f"Invalid speed factor: {factor}")

    parts: list[float] = []
    while factor < 0.5:
        parts.append(0.5)
        factor /= 0.5
    while factor > 2.0:
        parts.append(2.0)
        factor /= 2.0
    parts.append(factor)
    return ",".join(f"atempo={p:.6f}" for p in parts)


def fit_audio_to_duration_ffmpeg(
    ffmpeg_path: str,
    input_wav: Path,
    target_duration_seconds: float,
    output_wav: Path,
    max_speedup_factor: float = 1.0,
) -> None:
    target = float(target_duration_seconds)
    if target <= 0:
        raise RuntimeError(f"Invalid target duration: {target}")

    input_duration = get_wav_duration_seconds(input_wav)
    if input_duration <= 0:
        raise RuntimeError(f"Invalid source wav duration: {input_wav}")

    speed_factor_raw = max(0.01, input_duration / target)
    speedup_cap = max(1.0, float(max_speedup_factor))
    speed_factor = min(speed_factor_raw, speedup_cap)
    atempo_chain = build_atempo_chain(speed_factor)
    afilter = (
        f"{atempo_chain},"
        f"apad=whole_dur={target:.6f}"
    )

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            ffmpeg_path,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(input_wav),
            "-af",
            afilter,
            "-ar",
            "24000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output_wav),
        ]
    )


def apply_audio_speed_ffmpeg(
    ffmpeg_path: str,
    input_wav: Path,
    speed_factor: float,
    output_wav: Path,
) -> None:
    factor = float(speed_factor)
    if factor <= 0:
        raise RuntimeError(f"Invalid speed factor: {factor}")

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    atempo_chain = build_atempo_chain(factor)
    run_command(
        [
            ffmpeg_path,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(input_wav),
            "-af",
            atempo_chain,
            "-ar",
            "24000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output_wav),
        ]
    )


def trim_or_pad_audio_ffmpeg(
    ffmpeg_path: str,
    input_wav: Path,
    target_duration_seconds: float,
    output_wav: Path,
) -> None:
    target = float(target_duration_seconds)
    if target <= 0:
        raise RuntimeError(f"Invalid target duration: {target}")

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            ffmpeg_path,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(input_wav),
            "-af",
            f"apad=whole_dur={target:.6f}",
            "-ar",
            "24000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output_wav),
        ]
    )


def concat_wav_files_ffmpeg(ffmpeg_path: str, input_wavs: list[Path], output_wav: Path) -> None:
    if not input_wavs:
        raise RuntimeError("No scene wav files to concatenate.")

    concat_list = output_wav.parent / "_scene_concat_list.txt"
    lines: list[str] = []
    for wav_path in input_wavs:
        full = str(wav_path.resolve()).replace("\\", "/")
        full = full.replace("'", "'\\''")
        lines.append(f"file '{full}'")
    concat_list.write_text("\n".join(lines), encoding="utf-8")

    try:
        run_command(
            [
                ffmpeg_path,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list),
                "-ar",
                "24000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(output_wav),
            ]
        )
    finally:
        concat_list.unlink(missing_ok=True)


def synthesize_tts_audio_bytes(
    *,
    client: Any,
    types_module: Any,
    tts_model: str,
    voice_name: str,
    text: str,
    max_attempts: int = 4,
    retry_wait_seconds: float = 3.0,
    max_retry_wait_seconds: float = 12.0,
    pre_request_sleep_seconds: float = 0.0,
) -> bytes:
    attempts = max(1, int(max_attempts))
    last_error = ""
    for attempt in range(1, attempts + 1):
        if attempt > 1 and float(pre_request_sleep_seconds) > 0:
            sleep_with_log(
                pre_request_sleep_seconds,
                f"before tts.generate_content retry {attempt}/{attempts}",
            )
        try:
            response = client.models.generate_content(
                model=tts_model,
                contents=text,
                config=types_module.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types_module.SpeechConfig(
                        voice_config=types_module.VoiceConfig(
                            prebuilt_voice_config=types_module.PrebuiltVoiceConfig(
                                voice_name=voice_name
                            )
                        )
                    ),
                ),
            )
        except Exception as exc:
            if not (is_rate_limit_error(exc) or is_transient_api_error(exc)):
                raise
            if is_rate_limit_error(exc) and is_daily_quota_error(exc):
                raise
            last_error = str(exc)
            if attempt >= attempts:
                raise
            if is_rate_limit_error(exc):
                wait_for = max(retry_wait_seconds, extract_retry_delay_seconds(exc, retry_wait_seconds))
                if wait_for > float(max_retry_wait_seconds):
                    raise
            else:
                wait_for = max(1.0, float(retry_wait_seconds))
            print(f"TTS retryable error. Waiting {wait_for:.1f}s before retry {attempt + 1}/{attempts}...")
            time.sleep(wait_for)
            continue

        try:
            return extract_audio_bytes(response)
        except Exception as exc:
            last_error = str(exc)
            if attempt >= attempts:
                details = get_response_text(response)
                snippet = details[:200] if details else ""
                raise RuntimeError(
                    f"TTS returned no audio after {attempts} attempts. {last_error} {snippet}".strip()
                ) from exc
            time.sleep(0.25)

    raise RuntimeError(f"TTS failed: {last_error}")


def call_api_with_rate_limit_retry(
    *,
    operation_name: str,
    func: Any,
    max_attempts: int,
    retry_wait_seconds: float = 3.0,
    max_retry_wait_seconds: float = 12.0,
    pre_request_sleep_seconds: float = 0.0,
) -> Any:
    attempts = max(1, int(max_attempts))
    for attempt in range(1, attempts + 1):
        if attempt > 1 and float(pre_request_sleep_seconds) > 0:
            sleep_with_log(
                pre_request_sleep_seconds,
                f"before {operation_name} retry {attempt}/{attempts}",
            )
        try:
            return func()
        except Exception as exc:
            if not (is_rate_limit_error(exc) or is_transient_api_error(exc)):
                raise
            if is_rate_limit_error(exc) and is_daily_quota_error(exc):
                raise
            if attempt >= attempts:
                raise
            if is_rate_limit_error(exc):
                wait_for = max(
                    retry_wait_seconds,
                    extract_retry_delay_seconds(exc, retry_wait_seconds),
                )
                if wait_for > float(max_retry_wait_seconds):
                    raise
            else:
                wait_for = max(1.0, float(retry_wait_seconds))
            print(
                f"{operation_name} retryable error. Waiting {wait_for:.1f}s before retry {attempt + 1}/{attempts}..."
            )
            time.sleep(wait_for)
    raise RuntimeError(f"{operation_name} failed after retries.")


def run_pipeline_with_client(
    *,
    client: Any,
    types_module: Any,
    args: argparse.Namespace,
    product: dict[str, Any],
    video_path: Path,
    video_duration_seconds: float,
    ffmpeg_path: str,
    output_dir: Path,
    asin: str,
    key_index: int,
    key_masked: str,
) -> dict[str, str]:
    min_overrun_seconds = max(0.0, float(args.min_audio_overrun_seconds))
    max_overrun_seconds = max(min_overrun_seconds, float(args.max_audio_overrun_seconds))
    preferred_overrun_seconds = max(0.0, float(args.audio_overrun_seconds))
    target_overrun_seconds = min(
        max_overrun_seconds,
        max(min_overrun_seconds, preferred_overrun_seconds),
    )
    target_audio_duration_seconds = float(video_duration_seconds) + target_overrun_seconds
    min_audio_duration_seconds = float(video_duration_seconds) + min_overrun_seconds
    max_audio_duration_seconds = float(video_duration_seconds) + max_overrun_seconds
    cta_text = str(args.cta_text or "").strip()
    cta_required_line = (
        f"Mandatory CTA line: {cta_text}\n"
        if cta_text
        else "Do not include CTA lines, including 'link in bio' statements.\n"
    )
    final_structure_line = (
        "- Final CTA in closing line\n"
        if cta_text
        else "- No CTA line; end naturally with product value\n"
    )
    tighten_keep_line = (
        "Keep hook, benefits, and CTA, but remove repetitive or low-value lines.\n"
        if cta_text
        else "Keep hook and benefits, remove repetitive or low-value lines, and do not add CTA.\n"
    )
    print("Uploading video to Gemini Files API...")
    uploaded = call_api_with_rate_limit_retry(
        operation_name="files.upload",
        func=lambda: client.files.upload(file=str(video_path)),
        max_attempts=args.api_call_max_retries,
        retry_wait_seconds=args.rate_limit_wait_seconds,
        max_retry_wait_seconds=args.max_retry_wait_seconds,
        pre_request_sleep_seconds=args.pre_request_sleep_seconds,
    )
    uploaded = wait_until_file_active(
        client=client,
        uploaded_file=uploaded,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.file_timeout_seconds,
        rate_limit_wait_seconds=args.rate_limit_wait_seconds,
        max_retry_wait_seconds=args.max_retry_wait_seconds,
    )
    print(f"Video uploaded and ACTIVE: {uploaded.name}")

    analysis_prompt = build_analysis_prompt(
        product=product,
        video_duration_seconds=video_duration_seconds,
        target_audio_duration_seconds=target_audio_duration_seconds,
        scene_count=args.scene_count,
        viral_style=args.viral_style,
        cta_text=cta_text,
        hook_line=args.hook_line,
    )
    analysis_response = call_api_with_rate_limit_retry(
        operation_name="analysis.generate_content",
        func=lambda: client.models.generate_content(
            model=args.analysis_model,
            contents=[uploaded, analysis_prompt],
        ),
        max_attempts=args.api_call_max_retries,
        retry_wait_seconds=args.rate_limit_wait_seconds,
        max_retry_wait_seconds=args.max_retry_wait_seconds,
        pre_request_sleep_seconds=args.pre_request_sleep_seconds,
    )
    analysis_text = get_response_text(analysis_response).strip()
    if not analysis_text:
        raise RuntimeError("Analysis model returned empty text.")

    analysis_json = parse_json_from_text(analysis_text)
    if analysis_json is None:
        analysis_json = {
            "visual_summary_english": "",
            "visual_highlights": [],
            "target_audience_malayalam": "",
            "key_benefits_malayalam": [],
            "promo_script_malayalam": "",
            "short_cta_malayalam": "",
            "scene_segments": [],
            "raw_analysis_text": analysis_text,
        }

    raw_scene_segments = analysis_json.get("scene_segments", [])
    promo_script = str(analysis_json.get("promo_script_malayalam", "")).strip()
    user_hook_override = str(args.hook_line or "").strip()
    hook_hint_for_generation = (
        user_hook_override or "Auto-generate based on product metadata and visuals."
    )
    if not promo_script and isinstance(raw_scene_segments, list):
        promo_script = " ".join(
            str(item.get("narration_malayalam", "")).strip()
            for item in raw_scene_segments
            if isinstance(item, dict)
        ).strip()

    target_words_min, target_words_max, target_words_hint = compute_word_targets_for_duration(
        target_audio_duration_seconds
    )
    if not promo_script:
        script_response = call_api_with_rate_limit_retry(
            operation_name="script.generate_content",
            func=lambda: client.models.generate_content(
                model=args.analysis_model,
                contents=(
                    f"Write a viral product promotion voiceover script in Colloqual Malayalam around {target_words_hint} words "
                    f"(strictly between {target_words_min} and {target_words_max} words; never below {target_words_min}), "
                    "based on this visual analysis and metadata. Return plain Malayalam text only.\n\n"
                    "Required structure:\n"
                    "- Opening hook in first line\n"
                    "- Relatable pain/problem\n"
                    "- Quick benefit/solution from product\n"
                    "- Trust/proof statement\n"
                    f"{final_structure_line}\n"
                    "Viral format rules:\n"
                    "- Pick EXACTLY ONE primary format from: curiosity hook, before-vs-after, problem-to-quick-solution, relatable humor, myth-busting truth reveal, POV/real-life moment, comparison.\n"
                    "- Optionally add ONE secondary format only if it improves clarity.\n"
                    "- Never stack more than two formats.\n\n"
                    "Viral triggers to include naturally: curiosity, emotion, relatability, usefulness, surprise, simplicity.\n\n"
                    f"Viral style direction: {args.viral_style}\n"
                    f"Hook line to adapt: {hook_hint_for_generation}\n"
                    f"{cta_required_line}\n"
                    f"Video duration: {video_duration_seconds:.1f} seconds.\n"
                    f"Target spoken duration: about {target_audio_duration_seconds:.1f} seconds.\n\n"
                    "Required flow:\n"
                    "- Hook (0-2 sec)\n"
                    "- Relatable problem\n"
                    "- Quick value/insight\n"
                    "- Visual proof/result\n"
                    f"{final_structure_line}\n\n"
                    "Pacing rules:\n"
                    "- Use normal conversational Malayalam speed (do NOT write slow-paced narration).\n"
                    "- Keep output as one paragraph, not line-by-line script.\n"
                    "- Avoid ellipses (...), repeated fillers, and dramatic pause wording.\n"
                    "- Do not repeat the same sentence, hook wording, or benefit wording.\n"
                    "- Hook line must appear exactly once in the full script.\n"
                    "- Never ask the same question twice, even with small wording changes.\n"
                    "- Do not begin two sentences with the same first 3+ words.\n"
                    "- Every sentence must add new information.\n"
                    "- Self-check before returning: duplicate/near-duplicate sentence count must be zero.\n"
                    "- Timing objective is critical: narration must end about 1-3 seconds after video end.\n"
                    "- Keep sentence punctuation light so TTS does not add long pauses.\n\n"
                    f"Metadata:\nASIN: {asin}\nTitle: {product.get('title', '')}\n"
                    f"Description: {product.get('description', '')}\nURL: {product.get('url', '')}\n\n"
                    f"Visual Analysis:\n{analysis_text}"
                ),
            ),
            max_attempts=args.api_call_max_retries,
            retry_wait_seconds=args.rate_limit_wait_seconds,
            max_retry_wait_seconds=args.max_retry_wait_seconds,
            pre_request_sleep_seconds=args.pre_request_sleep_seconds,
        )
        promo_script = get_response_text(script_response).strip()
        analysis_json["promo_script_malayalam"] = promo_script

    script_words = len(re.findall(r"\S+", promo_script))
    if promo_script and script_words < target_words_min:
        expand_response = call_api_with_rate_limit_retry(
            operation_name="script.expand_for_duration",
            func=lambda: client.models.generate_content(
                model=args.analysis_model,
                contents=(
                    "Expand this Malayalam voiceover script so spoken duration becomes slightly longer than video.\n"
                    f"Current word count: {script_words}\n"
                    f"Required word count range: {target_words_min}-{target_words_max}\n"
                    f"Video duration: {video_duration_seconds:.1f} seconds.\n"
                    f"Target spoken duration: about {target_audio_duration_seconds:.1f} seconds.\n"
                    f"{cta_required_line}"
                    "Timing objective is critical: narration must end about 1-3 seconds after video end.\n"
                    "Use normal conversational speed, not slow narration.\n"
                    "Avoid ellipses and dramatic pause wording.\n"
                    "Remove duplicate/repeated sentence ideas while expanding.\n"
                    "Ensure hook line appears exactly once in the full script.\n"
                    "Never ask the same question twice or restate the same opener.\n"
                    "Do not begin two sentences with the same first 3+ words.\n"
                    "Every sentence must add new information.\n"
                    "Self-check before returning: duplicate/near-duplicate sentence count must be zero.\n"
                    "Keep meaning, style, and claims consistent. Keep first-line hook strong.\n"
                    "Return only expanded Malayalam script text.\n\n"
                    f"Current script:\n{promo_script}"
                ),
            ),
            max_attempts=args.api_call_max_retries,
            retry_wait_seconds=args.rate_limit_wait_seconds,
            max_retry_wait_seconds=args.max_retry_wait_seconds,
            pre_request_sleep_seconds=args.pre_request_sleep_seconds,
        )
        expanded_script = get_response_text(expand_response).strip()
        if expanded_script:
            promo_script = expanded_script
            analysis_json["promo_script_malayalam"] = promo_script
            script_words = len(re.findall(r"\S+", promo_script))

    if promo_script and script_words > target_words_max:
        shorten_response = call_api_with_rate_limit_retry(
            operation_name="script.shorten_for_duration",
            func=lambda: client.models.generate_content(
                model=args.analysis_model,
                contents=(
                    "Tighten this Malayalam voiceover script so it stays slightly longer than video, not much longer.\n"
                    f"Current word count: {script_words}\n"
                    f"Required word count range: {target_words_min}-{target_words_max}\n"
                    f"Video duration: {video_duration_seconds:.1f} seconds.\n"
                    f"Target spoken duration: about {target_audio_duration_seconds:.1f} seconds.\n"
                    f"{cta_required_line}"
                    "Use normal conversational speed. Do NOT use slow-paced narration.\n"
                    f"{tighten_keep_line}"
                    "Avoid ellipses and dramatic pause wording.\n"
                    "Remove duplicate/repeated sentence ideas while tightening.\n"
                    "Ensure hook line appears exactly once in the full script.\n"
                    "Never ask the same question twice or restate the same opener.\n"
                    "Do not begin two sentences with the same first 3+ words.\n"
                    "Every sentence must add new information.\n"
                    "Self-check before returning: duplicate/near-duplicate sentence count must be zero.\n"
                    "Return one-paragraph Malayalam text only.\n\n"
                    f"Current script:\n{promo_script}"
                ),
            ),
            max_attempts=args.api_call_max_retries,
            retry_wait_seconds=args.rate_limit_wait_seconds,
            max_retry_wait_seconds=args.max_retry_wait_seconds,
            pre_request_sleep_seconds=args.pre_request_sleep_seconds,
        )
        shortened_script = get_response_text(shorten_response).strip()
        if shortened_script:
            promo_script = shortened_script
            analysis_json["promo_script_malayalam"] = promo_script
            script_words = len(re.findall(r"\S+", promo_script))

    hook_line_candidate = str(analysis_json.get("hook_line_malayalam", "")).strip()
    hook_line_source = "gemini_analysis"
    if user_hook_override:
        hook_line_candidate = user_hook_override
        hook_line_source = "user_override"
    elif not hook_line_candidate:
        generated_hook = generate_hook_line_from_gemini(
            client=client,
            args=args,
            product=product,
            analysis_text=analysis_text,
        )
        if generated_hook:
            hook_line_candidate = generated_hook
            hook_line_source = "gemini_generated"

    if not hook_line_candidate:
        sentences = split_spoken_sentences(promo_script)
        hook_line_candidate = sentences[0] if sentences else ""
        hook_line_source = "promo_first_sentence"

    promo_script, repetitive_removed = remove_repetitive_sentences(promo_script, recent_window=12)
    if repetitive_removed > 0:
        print(f"Repetition cleanup removed {repetitive_removed} repeated sentence(s).")

    promo_script = ensure_hook_in_text(promo_script, hook_line_candidate)
    if cta_text:
        promo_script = ensure_cta_in_text(promo_script, cta_text)
    else:
        promo_script = remove_link_in_bio_sentences(promo_script)
    promo_script, repetitive_removed_after_hook = remove_repetitive_sentences(
        promo_script, recent_window=14
    )
    if repetitive_removed_after_hook > 0:
        print(
            "Post-hook repetition cleanup removed "
            f"{repetitive_removed_after_hook} repeated sentence(s)."
        )
    analysis_json["hook_line_malayalam"] = hook_line_candidate
    analysis_json["hook_line_source"] = hook_line_source
    analysis_json["promo_script_malayalam"] = promo_script
    final_script_words = len(re.findall(r"\S+", promo_script))
    print(
        f"Script word count (final): {final_script_words} "
        f"(target {target_words_min}-{target_words_max})"
    )
    if cta_text:
        if not str(analysis_json.get("short_cta_malayalam", "")).strip():
            analysis_json["short_cta_malayalam"] = cta_text
    else:
        analysis_json["short_cta_malayalam"] = ""

    if not promo_script:
        raise RuntimeError("Could not produce Malayalam promo script.")

    print(f"Generating narration audio (mode: {args.voice_timing_mode})...")
    full_raw_audio_path = output_dir / f"{asin}_promo_ml_raw.wav"
    sped_raw_audio_path = output_dir / f"{asin}_promo_ml_raw_sped.wav"
    scene_concat_audio_path = output_dir / f"{asin}_promo_ml_scene_concat.wav"
    audio_path = output_dir / f"{asin}_promo_ml.wav"
    full_raw_audio = synthesize_tts_audio_bytes(
        client=client,
        types_module=types_module,
        tts_model=args.tts_model,
        voice_name=args.voice_name,
        text=promo_script,
        max_attempts=args.api_call_max_retries,
        retry_wait_seconds=args.rate_limit_wait_seconds,
        max_retry_wait_seconds=args.max_retry_wait_seconds,
        pre_request_sleep_seconds=args.pre_request_sleep_seconds,
    )
    write_wav(full_raw_audio_path, full_raw_audio)
    tts_speed_factor = max(0.1, float(args.tts_speed_factor))
    effective_raw_audio_path = full_raw_audio_path
    if abs(tts_speed_factor - 1.0) > 0.0001:
        apply_audio_speed_ffmpeg(
            ffmpeg_path=ffmpeg_path,
            input_wav=full_raw_audio_path,
            speed_factor=tts_speed_factor,
            output_wav=sped_raw_audio_path,
        )
        effective_raw_audio_path = sped_raw_audio_path

    raw_audio_duration = get_wav_duration_seconds(effective_raw_audio_path)
    speech_coverage = raw_audio_duration / max(0.001, video_duration_seconds)
    if raw_audio_duration > max_audio_duration_seconds:
        timing_adjustment_reason = "above_max_overrun"
    elif raw_audio_duration < min_audio_duration_seconds:
        timing_adjustment_reason = "below_min_overrun"
    elif raw_audio_duration >= target_audio_duration_seconds:
        timing_adjustment_reason = "within_range_above_preferred"
    elif raw_audio_duration > video_duration_seconds:
        timing_adjustment_reason = "within_range_below_preferred"
    else:
        timing_adjustment_reason = "at_or_below_video_duration"
    print(
        "Narration duration "
        f"{raw_audio_duration:.2f}s for video {video_duration_seconds:.2f}s "
        f"(target {target_audio_duration_seconds:.2f}s, "
        f"allowed range {min_audio_duration_seconds:.2f}-{max_audio_duration_seconds:.2f}s, "
        f"coverage {speech_coverage:.2f}x)"
    )

    scene_segments = normalize_scene_segments(
        raw_segments=raw_scene_segments,
        video_duration_seconds=video_duration_seconds,
        promo_script=promo_script,
        scene_count=args.scene_count,
        min_scene_seconds=args.min_scene_seconds,
    )
    scene_segments = compress_scene_segments(
        segments=scene_segments,
        max_segments=max(1, int(args.scene_count)),
        video_duration_seconds=video_duration_seconds,
    )
    if not scene_segments:
        raise RuntimeError("Could not produce scene-aligned narration segments.")
    ensure_hook_in_scene_segments(scene_segments, hook_line_candidate)
    if cta_text:
        ensure_cta_in_scene_segments(scene_segments, cta_text)
    else:
        for segment in scene_segments:
            if not isinstance(segment, dict):
                continue
            line = str(segment.get("narration_malayalam", "")).strip()
            segment["narration_malayalam"] = remove_link_in_bio_sentences(line)

    if raw_audio_duration > max_audio_duration_seconds and args.voice_timing_mode != "global_stretch":
        fit_audio_to_duration_ffmpeg(
            ffmpeg_path=ffmpeg_path,
            input_wav=effective_raw_audio_path,
            target_duration_seconds=max_audio_duration_seconds,
            output_wav=scene_concat_audio_path,
            max_speedup_factor=float(args.max_natural_speedup_factor),
        )
        shutil.copy2(scene_concat_audio_path, audio_path)
        timing_adjustment_reason = "above_max_overrun_sped_to_max"
    elif raw_audio_duration < min_audio_duration_seconds and args.voice_timing_mode != "global_stretch":
        trim_or_pad_audio_ffmpeg(
            ffmpeg_path=ffmpeg_path,
            input_wav=effective_raw_audio_path,
            target_duration_seconds=min_audio_duration_seconds,
            output_wav=scene_concat_audio_path,
        )
        shutil.copy2(scene_concat_audio_path, audio_path)
        timing_adjustment_reason = "below_min_overrun_padded_to_min"
    elif args.voice_timing_mode == "global_stretch":
        fit_audio_to_duration_ffmpeg(
            ffmpeg_path=ffmpeg_path,
            input_wav=effective_raw_audio_path,
            target_duration_seconds=target_audio_duration_seconds,
            output_wav=scene_concat_audio_path,
            max_speedup_factor=float(args.max_global_speedup_factor),
        )
        shutil.copy2(scene_concat_audio_path, audio_path)
    else:
        scene_concat_audio_path = effective_raw_audio_path
        shutil.copy2(effective_raw_audio_path, audio_path)

    final_audio_duration = get_wav_duration_seconds(audio_path)

    analysis_json["scene_segments"] = scene_segments
    analysis_json["video_duration_seconds"] = round(video_duration_seconds, 3)
    analysis_json["target_audio_duration_seconds"] = round(target_audio_duration_seconds, 3)
    analysis_json["min_audio_duration_seconds"] = round(min_audio_duration_seconds, 3)
    analysis_json["max_audio_duration_seconds"] = round(max_audio_duration_seconds, 3)
    analysis_json["preferred_audio_overrun_seconds"] = round(target_overrun_seconds, 3)
    analysis_json["min_audio_overrun_seconds"] = round(min_overrun_seconds, 3)
    analysis_json["max_audio_overrun_seconds"] = round(max_overrun_seconds, 3)
    analysis_json["raw_audio_duration_seconds"] = round(raw_audio_duration, 3)
    analysis_json["speech_coverage_ratio"] = round(speech_coverage, 3)
    analysis_json["timing_adjustment_reason"] = timing_adjustment_reason
    analysis_json["final_audio_duration_seconds"] = round(final_audio_duration, 3)
    analysis_json["voice_timing_mode"] = args.voice_timing_mode
    analysis_json["max_global_speedup_factor"] = round(float(args.max_global_speedup_factor), 3)
    analysis_json["max_natural_speedup_factor"] = round(float(args.max_natural_speedup_factor), 3)
    analysis_json["tts_speed_factor"] = round(float(tts_speed_factor), 3)

    script_path = output_dir / f"{asin}_promo_ml.txt"
    analysis_json_path = output_dir / f"{asin}_analysis.json"
    raw_analysis_path = output_dir / f"{asin}_analysis_raw.txt"
    scene_segments_path = output_dir / f"{asin}_scene_segments.json"
    manifest_path = output_dir / f"{asin}_run_manifest.json"

    script_path.write_text(promo_script, encoding="utf-8")

    manifest: dict[str, Any] | None = None
    if args.keep_debug_files:
        analysis_json_path.write_text(
            json.dumps(analysis_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raw_analysis_path.write_text(analysis_text, encoding="utf-8")
        scene_segments_path.write_text(
            json.dumps(scene_segments, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        manifest = {
            "asin": asin,
            "product_url": str(product.get("url", "")),
            "video_path": str(video_path),
            "video_duration_seconds": round(video_duration_seconds, 3),
            "target_audio_duration_seconds": round(target_audio_duration_seconds, 3),
            "min_audio_duration_seconds": round(min_audio_duration_seconds, 3),
            "max_audio_duration_seconds": round(max_audio_duration_seconds, 3),
            "preferred_audio_overrun_seconds": round(target_overrun_seconds, 3),
            "min_audio_overrun_seconds": round(min_overrun_seconds, 3),
            "max_audio_overrun_seconds": round(max_overrun_seconds, 3),
            "raw_audio_duration_seconds": round(raw_audio_duration, 3),
            "speech_coverage_ratio": round(speech_coverage, 3),
            "final_audio_duration_seconds": round(final_audio_duration, 3),
            "uploaded_file_name": str(getattr(uploaded, "name", "")),
            "analysis_model": args.analysis_model,
            "tts_model": args.tts_model,
            "voice_name": args.voice_name,
            "viral_style": args.viral_style,
            "cta_text": cta_text,
            "hook_line_override": user_hook_override,
            "hook_line_used": hook_line_candidate,
            "hook_line_source": hook_line_source,
            "voice_timing_mode": args.voice_timing_mode,
            "max_global_speedup_factor": round(float(args.max_global_speedup_factor), 3),
            "max_natural_speedup_factor": round(float(args.max_natural_speedup_factor), 3),
            "tts_speed_factor": round(float(tts_speed_factor), 3),
            "timing_adjustment_reason": timing_adjustment_reason,
            "scene_count_requested": int(args.scene_count),
            "scene_count_generated": len(scene_segments),
            "api_key_index_used": key_index,
            "api_key_masked_used": key_masked,
            "files": {
                "analysis_json": str(analysis_json_path),
                "analysis_raw_text": str(raw_analysis_path),
                "promo_script_malayalam_text": str(script_path),
                "scene_segments_json": str(scene_segments_path),
                "promo_voice_raw_wav": str(full_raw_audio_path),
                "promo_voice_raw_sped_wav": (
                    str(sped_raw_audio_path) if effective_raw_audio_path == sped_raw_audio_path else ""
                ),
                "promo_voice_scene_concat_wav": str(scene_concat_audio_path),
                "promo_voice_wav": str(audio_path),
            },
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        for temp_path in (full_raw_audio_path, sped_raw_audio_path, scene_concat_audio_path):
            if temp_path == audio_path:
                continue
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

    return {
        "analysis_json": str(analysis_json_path) if args.keep_debug_files else "",
        "script": str(script_path),
        "audio": str(audio_path),
        "manifest": str(manifest_path) if args.keep_debug_files else "",
    }


def main() -> None:
    args = parse_args()
    if float(args.min_audio_overrun_seconds) < 0:
        raise ValueError("--min-audio-overrun-seconds must be >= 0")
    if float(args.max_audio_overrun_seconds) < float(args.min_audio_overrun_seconds):
        raise ValueError("--max-audio-overrun-seconds must be >= --min-audio-overrun-seconds")
    if float(args.max_global_speedup_factor) < 1.0:
        raise ValueError("--max-global-speedup-factor must be >= 1.0")
    if float(args.max_natural_speedup_factor) < 1.0:
        raise ValueError("--max-natural-speedup-factor must be >= 1.0")
    if float(args.tts_speed_factor) <= 0:
        raise ValueError("--tts-speed-factor must be > 0")

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'google-genai'. Install with: pip install google-genai"
        ) from exc

    api_keys = load_api_keys(args)

    results_path = Path(args.results_json).expanduser().resolve()
    if not results_path.exists():
        raise FileNotFoundError(f"Results JSON not found: {results_path}")

    product = load_first_product(results_path)
    video_path = resolve_video_path(product, args.video_path)
    if str(args.video_path or "").strip():
        path_asin = infer_asin_from_video_path(video_path)
        if path_asin:
            matched = load_product_by_asin(results_path, path_asin)
            if matched is not None:
                product = matched
            else:
                product = {
                    "asin": path_asin,
                    "url": "",
                    "title": "",
                    "description": "",
                }
    asin = str(product.get("asin", "")).strip() or "unknown"
    ffmpeg_path = shutil.which("ffmpeg") or ""
    ffprobe_path = shutil.which("ffprobe") or ""
    if not ffmpeg_path or not ffprobe_path:
        raise RuntimeError(
            "ffmpeg and ffprobe are required in PATH to produce scene-aligned exact-length audio."
        )
    video_duration_seconds = get_media_duration_seconds(ffprobe_path, video_path)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).expanduser().resolve() / f"{asin}_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results JSON: {results_path}")
    print(f"Video file: {video_path}")
    print(f"Output dir: {output_dir}")
    print(f"Analysis model: {args.analysis_model}")
    print(f"TTS model: {args.tts_model}")
    print(f"Voice: {args.voice_name}")
    print(f"Viral style: {args.viral_style}")
    print(
        f"Hook line override: "
        f"{args.hook_line if str(args.hook_line).strip() else '<auto: Gemini generated>'}"
    )
    print(f"CTA text: {str(args.cta_text).strip() or '<none>'}")
    print(f"Voice timing mode: {args.voice_timing_mode}")
    print(f"Audio overrun target: +{float(args.audio_overrun_seconds):.2f}s")
    print(
        "Allowed audio overrun range: "
        f"+{float(args.min_audio_overrun_seconds):.2f}s to +{float(args.max_audio_overrun_seconds):.2f}s"
    )
    print(f"Max global speed-up factor: {float(args.max_global_speedup_factor):.3f}x")
    print(f"Max natural speed-up factor: {float(args.max_natural_speedup_factor):.3f}x")
    print(f"Base TTS speed factor: {float(args.tts_speed_factor):.3f}x")
    print(f"Video duration: {video_duration_seconds:.3f}s")
    print(f"Scene count target: {max(1, int(args.scene_count))}")
    print(f"Pre-request sleep: {float(args.pre_request_sleep_seconds):.1f}s")
    print(f"API keys loaded: {len(api_keys)}")

    max_cycles = max(1, int(args.max_key_cycles))
    wait_seconds = max(0.0, float(args.rate_limit_wait_seconds))

    attempts = 0
    last_retryable_error = ""

    for cycle in range(1, max_cycles + 1):
        if max_cycles > 1:
            print(f"Key cycle {cycle}/{max_cycles}")

        for key_idx, api_key in enumerate(api_keys, start=1):
            attempts += 1
            key_masked = mask_api_key(api_key)
            print(
                f"Attempt {attempts}: using API key {key_idx}/{len(api_keys)} ({key_masked})"
            )
            client = genai.Client(api_key=api_key)

            try:
                outputs = run_pipeline_with_client(
                    client=client,
                    types_module=types,
                    args=args,
                    product=product,
                    video_path=video_path,
                    video_duration_seconds=video_duration_seconds,
                    ffmpeg_path=ffmpeg_path,
                    output_dir=output_dir,
                    asin=asin,
                    key_index=key_idx,
                    key_masked=key_masked,
                )
                print("Done.")
                print(f"Malayalam script: {outputs['script']}")
                print(f"Malayalam voice WAV: {outputs['audio']}")
                if outputs.get("analysis_json"):
                    print(f"Analysis JSON: {outputs['analysis_json']}")
                if outputs.get("manifest"):
                    print(f"Manifest: {outputs['manifest']}")
                return
            except Exception as exc:
                if not is_retriable_key_error(exc):
                    raise

                last_retryable_error = str(exc)
                print(f"Retryable key error on {key_idx}/{len(api_keys)} ({key_masked}): {exc}")
                wait_for = max(wait_seconds, float(args.key_rotate_sleep_seconds))
                if is_rate_limit_error(exc):
                    retry_hint = extract_retry_delay_seconds(exc, wait_for)
                    if retry_hint > wait_for:
                        print(
                            f"Rate-limit suggested waiting {retry_hint:.1f}s; "
                            f"rotating key after {wait_for:.1f}s instead."
                        )
                if wait_for > 0:
                    sleep_with_log(wait_for, "before rotating to next API key")

    raise RuntimeError(
        "All provided API keys failed with retryable errors and retries were exhausted. "
        f"Attempts: {attempts}. Last error: {last_retryable_error}"
    )


if __name__ == "__main__":
    main()
