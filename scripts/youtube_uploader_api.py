#!/usr/bin/env python3
"""
Upload a video to YouTube using the official YouTube Data API (OAuth).

Setup:
1. Create OAuth client credentials (Desktop app) in Google Cloud Console.
2. Save the downloaded JSON as client_secrets.json in this folder.
3. Run this script. The first run opens a browser for Google consent.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        if (parent / "requirements.txt").exists():
            return parent
    return here.parent


PROJECT_ROOT = _project_root()
DEFAULT_CLIENT_SECRETS = str(PROJECT_ROOT / "secrets" / "client_secret.json")
DEFAULT_VIDEO_PATH = str(PROJECT_ROOT / "data" / "reel" / "reel.mp4")
DEFAULT_RESULTS_JSON = str(PROJECT_ROOT / "data" / "runtime" / "current_product_video_results.json")
DEFAULT_FALLBACK_RESULTS_JSON = str(PROJECT_ROOT / "data" / "product_video_results.json")
DEFAULT_API_KEYS_FILE = str(PROJECT_ROOT / "secrets" / "gemini_keys.txt")
DEFAULT_METADATA_MODEL = "gemini-2.5-flash"
DEFAULT_TITLE = ""
DEFAULT_DESCRIPTION = ""
DEFAULT_PRIVACY = "private"
ASIN_RE = re.compile(r"\b([A-Z0-9]{10})\b", re.IGNORECASE)
EN_STOPWORDS = {
    "a", "an", "the", "and", "or", "for", "to", "of", "in", "on", "at", "by", "with", "from", "into",
    "your", "you", "this", "that", "these", "those", "is", "are", "was", "were", "be", "been", "being",
    "it", "its", "as", "about", "how", "what", "why", "when", "where", "who", "which", "will", "can",
    "could", "should", "would", "may", "might", "than", "then", "also", "very", "more", "most", "best",
    "new", "now", "today", "video", "short", "reel", "review", "product", "amazon", "india", "buy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a video to YouTube via API.")
    parser.add_argument(
        "--video",
        default=DEFAULT_VIDEO_PATH,
        help="Path to the video file.",
    )
    parser.add_argument(
        "--title",
        default=DEFAULT_TITLE,
        help="Video title.",
    )
    parser.add_argument(
        "--description",
        default=DEFAULT_DESCRIPTION,
        help="Video description.",
    )
    parser.add_argument(
        "--privacy",
        choices=["private", "unlisted", "public"],
        default=DEFAULT_PRIVACY,
        help="Video privacy status.",
    )
    parser.add_argument(
        "--category-id",
        default="22",
        help="YouTube category id (default: 22 = People & Blogs).",
    )
    parser.add_argument(
        "--client-secrets",
        default=DEFAULT_CLIENT_SECRETS,
        help="Path to OAuth client secrets JSON file.",
    )
    parser.add_argument(
        "--token-file",
        default="youtube_token.json",
        help="Path to store OAuth token JSON.",
    )
    parser.add_argument(
        "--metadata-mode",
        choices=["auto", "off", "force"],
        default="force",
        help=(
            "'auto' generates title/description with Gemini when default placeholders are used; "
            "'force' always regenerates; 'off' disables Gemini metadata generation."
        ),
    )
    parser.add_argument(
        "--results-json",
        default=DEFAULT_RESULTS_JSON,
        help=(
            "Results JSON from amazon_product_video_checker.py used to fetch product title/description "
            "for dynamic metadata."
        ),
    )
    parser.add_argument(
        "--fallback-results-json",
        default=DEFAULT_FALLBACK_RESULTS_JSON,
        help="Fallback results JSON path if --results-json is unavailable.",
    )
    parser.add_argument(
        "--asin",
        default="",
        help="Optional ASIN override for selecting product metadata row.",
    )
    parser.add_argument(
        "--api-keys-file",
        default=DEFAULT_API_KEYS_FILE,
        help="Gemini API keys file (one key per line or comma-separated).",
    )
    parser.add_argument(
        "--metadata-model",
        default=DEFAULT_METADATA_MODEL,
        help="Gemini model for generating YouTube title/description metadata.",
    )
    return parser.parse_args()


def split_key_tokens(raw: str) -> list[str]:
    tokens = re.split(r"[\s,;]+", raw or "")
    return [t.strip() for t in tokens if t and t.strip()]


def unique_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def load_gemini_keys(keys_file: Path) -> list[str]:
    keys: list[str] = []
    if keys_file.exists():
        keys.extend(split_key_tokens(keys_file.read_text(encoding="utf-8-sig")))
    keys.extend(split_key_tokens(os.getenv("GEMINI_API_KEYS", "")))
    keys.extend(split_key_tokens(os.getenv("GEMINI_API_KEY", "")))
    return unique_keep_order(keys)


def extract_asin_from_text(text: str) -> str:
    match = ASIN_RE.search(str(text or "").upper())
    return match.group(1).upper() if match else ""


def infer_asin(video_path: Path, override_asin: str) -> str:
    manual = extract_asin_from_text(override_asin)
    if manual:
        return manual
    probe = f"{video_path.stem} {video_path.parent.name}"
    return extract_asin_from_text(probe)


def load_results_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def find_product_row(results_payload: dict[str, Any], asin: str) -> dict[str, Any] | None:
    rows = results_payload.get("results", [])
    if not isinstance(rows, list) or not rows:
        return None
    target = str(asin or "").strip().upper()
    if target:
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("asin", "")).strip().upper() == target:
                return row
    for row in rows:
        if isinstance(row, dict):
            return row
    return None


def sanitize_scraped_description(text: str) -> str:
    body = re.sub(r"\s+", " ", str(text or "")).strip()
    if not body:
        return ""
    for token in (
        "window.$Nav",
        "Sponsored window.$Nav",
        "aax-eu-zaz.amazon.in",
        "amazonmobile://intent?",
        "\"renderingContext\"",
        "window.P &&",
        "$Nav.when(",
        "Sign in New customer?",
        "Your Lists Create a Wish List",
        "accountListContent",
        "data-csa-c-",
    ):
        idx = body.find(token)
        if idx > 0:
            body = body[:idx].strip()
    if len(body) > 2200:
        body = body[:2200].rstrip() + "..."
    return body


def ensure_malayalam_suffix_title(title: str) -> str:
    suffix = "(MALAYALAM)"
    value = re.sub(r"\s+", " ", str(title or "")).strip()
    if not value:
        value = "Product Review"
    value = re.sub(r"\(\s*malayalam\s*\)\s*$", "", value, flags=re.IGNORECASE).strip()
    reserve = len(suffix) + 1
    if len(value) > 100 - reserve:
        value = value[: 100 - reserve].rstrip(" -|,:;")
    out = f"{value} {suffix}".strip()
    return out[:100].rstrip()


def derive_keywords_from_product_text(product_title: str, product_desc: str, limit: int = 12) -> list[str]:
    text = f"{product_title or ''} {product_desc or ''}"
    chunks = re.split(r"[,\n\r\|\.;:()\-]+", text)
    phrases: list[str] = []
    for chunk in chunks:
        phrase = re.sub(r"\s+", " ", str(chunk or "")).strip()
        if not phrase:
            continue
        if not re.search(r"[A-Za-z]", phrase):
            continue
        word_count = len([w for w in phrase.split() if w])
        if 2 <= word_count <= 5 and len(phrase) <= 42:
            phrases.append(phrase)

    singles = re.findall(r"[A-Za-z][A-Za-z0-9+\-]{2,}", text)
    phrases.extend(singles)

    out: list[str] = []
    seen: set[str] = set()
    for token in phrases:
        cleaned = re.sub(r"[^A-Za-z0-9 #&+\-]", "", token).strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in EN_STOPWORDS:
            continue
        if len(cleaned) <= 2:
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        out.append(cleaned)
        if len(out) >= limit:
            break
    return out


def _split_sentences(text: str) -> list[str]:
    body = re.sub(r"\s+", " ", str(text or "")).strip()
    if not body:
        return []
    parts = [p.strip() for p in re.split(r"(?<=[\.\!\?])\s+", body) if p.strip()]
    return parts or [body]


def build_fallback_description(product_title: str, product_desc: str, keywords: list[str]) -> str:
    title = re.sub(r"\s+", " ", product_title or "").strip()
    desc = re.sub(r"\s+", " ", product_desc or "").strip()
    points = _split_sentences(desc)
    short_points = points[:4]
    body_parts = [
        f"{title} explained in this short video with practical highlights and real-use context.",
    ]
    if short_points:
        body_parts.append("Key features: " + " ".join(short_points[:2]).strip())
    if len(short_points) > 2:
        body_parts.append("Why it matters: " + " ".join(short_points[2:4]).strip())
    body_parts.append("Watch till the end for a quick buying perspective in Malayalam.")
    return normalize_description_no_links("\n\n".join(body_parts), keywords=keywords)


def normalize_description_no_links(text: str, keywords: list[str] | None = None) -> str:
    body = str(text or "").replace("\r", "\n")
    body = re.sub(r"[^\x00-\x7F]+", " ", body)
    body = re.sub(r"https?://\S+", "", body, flags=re.IGNORECASE)
    body = re.sub(r"www\.\S+", "", body, flags=re.IGNORECASE)
    body = re.sub(r"(?im)^\s*(product\s*url|affiliate\s*tag)\s*:.*$", "", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    body = re.sub(r"[ \t]+", " ", body).strip()

    paragraphs = [re.sub(r"\s+", " ", line).strip(" -•\t") for line in body.split("\n") if line.strip()]
    if len(paragraphs) <= 1:
        sentences = _split_sentences(" ".join(paragraphs))
        if len(sentences) >= 5:
            paragraphs = [
                " ".join(sentences[:2]).strip(),
                " ".join(sentences[2:4]).strip(),
                " ".join(sentences[4:]).strip(),
            ]
        elif sentences:
            paragraphs = [" ".join(sentences[:2]).strip(), " ".join(sentences[2:]).strip()]
    paragraphs = [p for p in paragraphs if p]
    if paragraphs:
        body = "\n\n".join(paragraphs[:4]).strip()

    clean_keywords: list[str] = []
    for kw in keywords or []:
        token = re.sub(r"[^A-Za-z0-9 #&\\-]", "", str(kw or "")).strip()
        if token and token.lower() not in {k.lower() for k in clean_keywords}:
            clean_keywords.append(token)
    if clean_keywords:
        top = clean_keywords[:12]
        tags = [f"#{re.sub(r'[^A-Za-z0-9]', '', kw)[:30]}" for kw in top]
        tags = [t for t in tags if len(t) > 1]
        keywords_line = "Keywords: " + ", ".join(top)
        hashtags_line = " ".join(tags[:12])
        tail = "\n\n".join([keywords_line, hashtags_line]).strip()
        if tail:
            body = f"{body}\n\n{tail}".strip()

    return body[:5000].rstrip()


def get_response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    chunks: list[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                chunks.append(part_text.strip())
    return "\n".join(chunks).strip()


def extract_first_json_object(text: str) -> str:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3 and lines[-1].strip().startswith("```"):
            raw = "\n".join(lines[1:-1]).strip()
    start = raw.find("{")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escaped = False
    for idx, ch in enumerate(raw[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == "\"":
                in_string = False
            continue
        if ch == "\"":
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start : idx + 1]
    return ""


def maybe_generate_dynamic_metadata(
    *,
    args: argparse.Namespace,
    video_path: Path,
    title: str,
    description: str,
) -> tuple[str, str]:
    mode = str(args.metadata_mode or "auto").strip().lower()
    if mode == "off":
        return title, description
    if mode == "auto" and not (title == DEFAULT_TITLE or description == DEFAULT_DESCRIPTION):
        return title, description

    keys = load_gemini_keys(Path(args.api_keys_file).expanduser().resolve())
    if not keys:
        if mode == "force":
            raise RuntimeError("Dynamic metadata failed: no Gemini API keys found.")
        print("Dynamic metadata skipped: no Gemini API keys found.", file=sys.stderr)
        return title, description

    try:
        from google import genai
    except Exception as exc:
        if mode == "force":
            raise RuntimeError(f"Dynamic metadata failed: google-genai import error: {exc}") from exc
        print(f"Dynamic metadata skipped: google-genai import failed ({exc})", file=sys.stderr)
        return title, description

    primary_results = Path(args.results_json).expanduser().resolve()
    fallback_results = Path(args.fallback_results_json).expanduser().resolve()
    payload = load_results_payload(primary_results) or load_results_payload(fallback_results)
    if payload is None:
        if mode == "force":
            raise RuntimeError("Dynamic metadata failed: results JSON not found/invalid.")
        print("Dynamic metadata skipped: results JSON not found/invalid.", file=sys.stderr)
        return title, description

    asin = infer_asin(video_path, args.asin)
    row = find_product_row(payload, asin)
    if row is None:
        if mode == "force":
            raise RuntimeError("Dynamic metadata failed: no matching product row.")
        print("Dynamic metadata skipped: no matching product row.", file=sys.stderr)
        return title, description

    product_title = re.sub(r"\s+", " ", str(row.get("title", "")).strip())
    product_desc = sanitize_scraped_description(str(row.get("description", "")))
    fallback_keywords = derive_keywords_from_product_text(product_title, product_desc)

    prompt = (
        "Generate high-CTR, SEO-focused YouTube metadata for a product video.\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        '  "title": "string",\n'
        '  "description": "string",\n'
        '  "keywords": ["keyword1", "keyword2"]\n'
        "}\n\n"
        "Rules:\n"
        "- Title max 100 chars and must end with '(MALAYALAM)'.\n"
        "- Keep title truthful and product-specific; avoid fake claims.\n"
        "- Description MUST be in clear English only.\n"
        "- Description should be keyword-rich, readable, and structured in short paragraphs.\n"
        "- Include practical value points and high-intent search terms for this product category.\n"
        "- Do NOT include URLs, product links, affiliate links, or any external links.\n"
        "- Add relevant hashtags at the end.\n\n"
        f"ASIN: {asin}\n"
        f"Product Title: {product_title}\n"
        f"Product Description: {product_desc}\n"
    )

    for api_key in keys:
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=args.metadata_model,
                contents=prompt,
            )
            text = get_response_text(response)
            blob = extract_first_json_object(text) or text
            parsed = json.loads(blob)
            if not isinstance(parsed, dict):
                continue
            new_title = ensure_malayalam_suffix_title(str(parsed.get("title", "")).strip() or title)
            keyword_list = parsed.get("keywords", [])
            keywords = [str(x).strip() for x in keyword_list if str(x).strip()] if isinstance(keyword_list, list) else []
            if not keywords:
                keywords = fallback_keywords
            new_description = normalize_description_no_links(
                str(parsed.get("description", "")).strip(),
                keywords=keywords,
            )
            if not new_description:
                new_description = build_fallback_description(product_title, product_desc, keywords)
            if not new_description:
                continue
            return new_title, new_description
        except Exception as exc:
            print(f"Gemini metadata generation failed on one key: {exc}", file=sys.stderr)
            continue

    if mode == "force":
        raise RuntimeError("Dynamic metadata generation failed for all Gemini keys.")
    print("Dynamic metadata generation failed for all keys. Using fallback title/description.", file=sys.stderr)
    return title, description


def get_credentials(client_secrets: Path, token_file: Path) -> Credentials:
    creds = None
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    elif not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets), SCOPES)
        creds = flow.run_local_server(port=0)

    token_file.write_text(creds.to_json(), encoding="utf-8")
    return creds


def upload_video(
    video_path: Path,
    title: str,
    description: str,
    privacy: str,
    category_id: str,
    creds: Credentials,
):
    youtube = build("youtube", "v3", credentials=creds)

    request = youtube.videos().insert(
        part="snippet,status",
        body={
            "snippet": {
                "title": title,
                "description": description,
                "categoryId": category_id,
            },
            "status": {
                "privacyStatus": privacy,
            },
        },
        media_body=MediaFileUpload(str(video_path), chunksize=-1, resumable=True),
    )

    print("Uploading...")
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"Progress: {int(status.progress() * 100)}%")

    print(f"Upload complete. Video ID: {response['id']}")
    print(f"URL: https://www.youtube.com/watch?v={response['id']}")


def main():
    args = parse_args()
    if len(sys.argv) == 1:
        print("Running with default upload settings (Run Python File mode).")
        print(f"Video: {args.video}")
        print(f"Title: {args.title}")
        print(f"Privacy: {args.privacy}")

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists() or not video_path.is_file():
        print(f"Video file does not exist: {video_path}", file=sys.stderr)
        sys.exit(1)

    client_secrets = Path(args.client_secrets).expanduser().resolve()
    if not client_secrets.exists() or not client_secrets.is_file():
        print(
            f"Missing OAuth client secrets file: {client_secrets}\n"
            "Create/download it in Google Cloud Console and pass --client-secrets <path>.",
            file=sys.stderr,
        )
        sys.exit(1)

    token_file = Path(args.token_file).expanduser().resolve()
    if not token_file.parent.exists():
        os.makedirs(token_file.parent, exist_ok=True)

    final_title, final_description = maybe_generate_dynamic_metadata(
        args=args,
        video_path=video_path,
        title=args.title,
        description=args.description,
    )

    try:
        creds = get_credentials(client_secrets, token_file)
        upload_video(
            video_path=video_path,
            title=final_title,
            description=final_description,
            privacy=args.privacy,
            category_id=args.category_id,
            creds=creds,
        )
    except HttpError as exc:
        print(f"YouTube API error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
