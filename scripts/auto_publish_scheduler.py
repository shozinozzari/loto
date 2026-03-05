#!/usr/bin/env python3
"""
Run the full reel pipeline on a fixed daily schedule.

Schedule defaults:
  11:30, 17:30, 19:30 (Asia/Kolkata)

Behavior:
  - Sleeps until 30 minutes before next slot.
  - Prepares one publishable reel from the next product URL(s).
  - Uploads at slot time.
  - Continues through all products in a branch.
  - Moves to next branch when current branch is exhausted.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time, timedelta
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv


ASIN_RE = re.compile(r"/(?:dp|gp/product)/([A-Z0-9]{10})", re.IGNORECASE)


class PipelineError(RuntimeError):
    pass


@dataclass
class SchedulerState:
    branch_index: int = 0
    product_index: int = 0
    current_branch_url: str = ""
    current_branch_name: str = ""
    branch_products: list[str] = field(default_factory=list)
    uploaded_urls: list[str] = field(default_factory=list)
    skipped_urls: list[str] = field(default_factory=list)
    upload_error_by_url: dict[str, str] = field(default_factory=dict)
    last_upload_time: str = ""
    last_uploaded_url: str = ""
    last_completed_slot: str = ""

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "SchedulerState":
        state = cls()
        for key in state.__dataclass_fields__.keys():
            if key in payload:
                setattr(state, key, payload[key])
        return state

    def to_json(self) -> dict[str, Any]:
        return {
            "branch_index": int(self.branch_index),
            "product_index": int(self.product_index),
            "current_branch_url": str(self.current_branch_url),
            "current_branch_name": str(self.current_branch_name),
            "branch_products": list(self.branch_products),
            "uploaded_urls": list(self.uploaded_urls),
            "skipped_urls": list(self.skipped_urls),
            "upload_error_by_url": dict(self.upload_error_by_url),
            "last_upload_time": str(self.last_upload_time),
            "last_uploaded_url": str(self.last_uploaded_url),
            "last_completed_slot": str(self.last_completed_slot),
        }


@dataclass
class PreparedReel:
    product_url: str
    asin: str
    product_title: str
    product_description: str
    final_video_path: Path
    cleanup_paths: list[Path] = field(default_factory=list)


def parse_slots(raw: str) -> list[dt_time]:
    slots: list[dt_time] = []
    seen: set[tuple[int, int]] = set()
    for token in [x.strip() for x in str(raw).split(",") if x.strip()]:
        parts = token.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid slot format: {token!r}. Use HH:MM (24h).")
        hour = int(parts[0])
        minute = int(parts[1])
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError(f"Invalid slot time: {token!r}.")
        key = (hour, minute)
        if key in seen:
            continue
        seen.add(key)
        slots.append(dt_time(hour=hour, minute=minute))
    if not slots:
        raise ValueError("At least one publish slot is required.")
    slots.sort(key=lambda t: (t.hour, t.minute))
    return slots


def extract_asin(url: str) -> str:
    match = ASIN_RE.search(url or "")
    return match.group(1).upper() if match else ""


def now_tz(tz: ZoneInfo) -> datetime:
    return datetime.now(tz)


def next_slot_after(current: datetime, slots: list[dt_time]) -> datetime:
    candidates: list[datetime] = []
    for slot in slots:
        candidate = current.replace(
            hour=slot.hour,
            minute=slot.minute,
            second=0,
            microsecond=0,
        )
        if candidate > current:
            candidates.append(candidate)
    if candidates:
        return min(candidates)
    tomorrow = current + timedelta(days=1)
    first = slots[0]
    return tomorrow.replace(
        hour=first.hour,
        minute=first.minute,
        second=0,
        microsecond=0,
    )


class AutoPublisher:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.project_root = Path(args.project_root).expanduser().resolve()
        self.scripts_dir = self.project_root / "scripts"
        self.runtime_dir = Path(args.runtime_dir).expanduser().resolve()
        self.runtime_dir.mkdir(parents=True, exist_ok=True)

        self.tz = ZoneInfo(args.timezone)
        self.slots = parse_slots(args.schedule_slots)

        self.state_path = self.runtime_dir / "scheduler_state.json"
        self.branches_path = self.runtime_dir / "branches.json"
        self.current_products_path = self.runtime_dir / "current_branch_products.json"
        self.current_result_path = self.runtime_dir / "current_product_video_results.json"
        self.download_dir = self.runtime_dir / "downloaded_videos"
        self.gemini_outputs_dir = self.runtime_dir / "gemini_outputs"
        self.reels_dir = self.runtime_dir / "reels"
        self.qr_output = self.runtime_dir / "amazon_product_qr.png"
        self.reels_dir.mkdir(parents=True, exist_ok=True)
        self.gemini_outputs_dir.mkdir(parents=True, exist_ok=True)
        self.download_dir.mkdir(parents=True, exist_ok=True)

        self.upload_log_path = self.runtime_dir / "upload_log.jsonl"
        self.state = self._load_state()
        self.branches: list[dict[str, Any]] = []
        self._gemini_keys_cache: list[str] | None = None

    def log(self, message: str) -> None:
        stamp = now_tz(self.tz).strftime("%Y-%m-%d %H:%M:%S %Z")
        print(f"[{stamp}] {message}", flush=True)

    def _load_state(self) -> SchedulerState:
        if not self.state_path.exists():
            return SchedulerState()
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return SchedulerState()
        return SchedulerState.from_json(payload)

    def _save_state(self) -> None:
        self.state_path.write_text(
            json.dumps(self.state.to_json(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _run_cmd(self, cmd: list[str]) -> str:
        """Run a subprocess command and return its stdout on success."""
        self.log("Running: " + " ".join(cmd))
        completed = subprocess.run(
            cmd,
            cwd=str(self.project_root),
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode == 0:
            stdout = (completed.stdout or "").strip()
            if stdout:
                self.log(stdout.splitlines()[-1])
            return stdout

        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        detail = stderr or stdout or f"exit code {completed.returncode}"
        raise PipelineError(detail[-1000:])

    def _script_path(self, script_name: str) -> str:
        script = (self.scripts_dir / script_name).resolve()
        if not script.exists():
            raise PipelineError(f"Script not found: {script}")
        return str(script)

    @staticmethod
    def _split_key_tokens(raw: str) -> list[str]:
        tokens = re.split(r"[\s,;]+", raw or "")
        return [t.strip() for t in tokens if t and t.strip()]

    @staticmethod
    def _unique_keep_order(values: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            out.append(value)
        return out

    def _load_gemini_keys(self) -> list[str]:
        if self._gemini_keys_cache is not None:
            return self._gemini_keys_cache

        keys: list[str] = []
        # Prefer env vars (set by .env) first
        keys.extend(self._split_key_tokens(os.getenv("GEMINI_API_KEYS", "")))
        keys.extend(self._split_key_tokens(os.getenv("GEMINI_API_KEY", "")))

        # Fall back to keys file on disk
        if not keys:
            api_keys_file = str(self.args.api_keys_file or "").strip()
            if api_keys_file:
                key_file_path = Path(api_keys_file).expanduser().resolve()
                if key_file_path.exists():
                    keys.extend(self._split_key_tokens(key_file_path.read_text(encoding="utf-8-sig")))

        keys = self._unique_keep_order(keys)
        self._gemini_keys_cache = keys
        return keys

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()

        parts: list[str] = []
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) or []:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str) and part_text.strip():
                    parts.append(part_text.strip())
        return "\n".join(parts).strip()

    @staticmethod
    def _extract_first_json_object(text: str) -> str:
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

    def _sanitize_product_description(self, text: str) -> str:
        body = re.sub(r"\s+", " ", str(text or "")).strip()
        if not body:
            return ""

        # Drop noisy embedded script/ad tails commonly scraped from Amazon pages.
        kill_tokens = (
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
        )
        for token in kill_tokens:
            pos = body.find(token)
            if pos > 0:
                body = body[:pos].strip()

        if len(body) > 2200:
            body = body[:2200].rstrip() + "..."
        return body

    @staticmethod
    def _normalize_description_no_links(text: str, keywords: list[str] | None = None) -> str:
        body = str(text or "").replace("\r", "\n")
        body = re.sub(r"https?://\S+", "", body, flags=re.IGNORECASE)
        body = re.sub(r"www\.\S+", "", body, flags=re.IGNORECASE)
        body = re.sub(r"(?im)^\s*(product\s*url|affiliate\s*tag)\s*:.*$", "", body)
        body = re.sub(r"\n{3,}", "\n\n", body)
        body = re.sub(r"[ \t]+", " ", body).strip()
        clean_keywords: list[str] = []
        for kw in keywords or []:
            token = re.sub(r"[^A-Za-z0-9 #&\\-]", "", str(kw or "")).strip()
            if token and token.lower() not in {k.lower() for k in clean_keywords}:
                clean_keywords.append(token)
        if clean_keywords:
            top = clean_keywords[:12]
            tags = [f"#{re.sub(r'[^A-Za-z0-9]', '', kw)[:30]}" for kw in top]
            tags = [t for t in tags if len(t) > 1]
            body = (
                f"{body}\n\n"
                f"Keywords: {', '.join(top)}\n\n"
                f"{' '.join(tags[:12])}"
            ).strip()
        return body[:5000].rstrip()

    def _ensure_malayalam_suffix_title(self, title: str) -> str:
        suffix = "(MALAYALAM)"
        value = re.sub(r"\s+", " ", str(title or "")).strip()
        if not value:
            value = "Product Review"

        # Remove existing variants and enforce exact suffix once.
        value = re.sub(r"\(\s*malayalam\s*\)\s*$", "", value, flags=re.IGNORECASE).strip()
        base = value
        max_total = 100
        reserve = len(suffix) + 1
        if len(base) > max_total - reserve:
            base = base[: max_total - reserve].rstrip(" -|,:;")
        final = f"{base} {suffix}".strip()
        if len(final) > max_total:
            allowed = max(1, max_total - reserve)
            final = f"{base[:allowed].rstrip()} {suffix}".strip()
        return final

    def _gemini_generate_youtube_metadata(self, prepared: PreparedReel) -> tuple[str, str] | None:
        keys = self._load_gemini_keys()
        if not keys:
            self.log("No Gemini key available for dynamic YouTube metadata; using static fallback.")
            return None

        try:
            from google import genai
        except Exception as exc:
            self.log(f"google-genai import failed for metadata generation: {exc}")
            return None

        product_title = re.sub(r"\s+", " ", prepared.product_title).strip()
        product_desc = self._sanitize_product_description(prepared.product_description)
        prefix = str(self.args.youtube_description_prefix or "").strip()

        prompt = (
            "Generate high-CTR, SEO-optimized YouTube metadata for a product short video.\n"
            "Goal: maximize discoverability using relevant keywords naturally.\n"
            "Do not make unrealistic guarantees (no 'No.1 guaranteed' claims).\n"
            "Title must be truthful and product-specific.\n"
            "Return ONLY valid JSON with schema:\n"
            "{\n"
            '  "title": "string",\n'
            '  "description": "string",\n'
            '  "keywords": ["kw1", "kw2"]\n'
            "}\n\n"
            "Hard rules:\n"
            "- Title max 100 chars.\n"
            "- Title must end exactly with '(MALAYALAM)'.\n"
            "- Put primary search keyword early in title.\n"
            "- Description MUST be in English only.\n"
            "- Description: 2-4 short readable paragraphs.\n"
            "- Include practical product benefit keywords and high-intent YouTube search terms.\n"
            "- Do NOT include any URL, product link, affiliate link, or external link.\n"
            "- Avoid fake claims, clickbait lies, and policy-unsafe language.\n"
            "- Include 8-15 relevant hashtags at end of description.\n\n"
            f"ASIN: {prepared.asin}\n"
            f"Product Title: {product_title}\n"
            f"Product Description: {product_desc}\n"
            f"Description Prefix (must include near start): {prefix}\n"
        )

        max_retries = max(1, int(self.args.youtube_metadata_max_retries))
        for key_index, api_key in enumerate(keys, start=1):
            try:
                client = genai.Client(api_key=api_key)
            except Exception as exc:
                self.log(f"Gemini client init failed for key {key_index}: {exc}")
                continue

            for attempt in range(1, max_retries + 1):
                try:
                    response = client.models.generate_content(
                        model=self.args.youtube_metadata_model,
                        contents=prompt,
                    )
                    text = self._extract_response_text(response)
                    json_blob = self._extract_first_json_object(text) or text
                    payload = json.loads(json_blob)
                    if not isinstance(payload, dict):
                        raise ValueError("metadata response is not an object")

                    title = self._ensure_malayalam_suffix_title(str(payload.get("title", "")).strip())
                    keyword_list = payload.get("keywords", [])
                    keywords = [str(x).strip() for x in keyword_list if str(x).strip()] if isinstance(keyword_list, list) else []
                    description = self._normalize_description_no_links(
                        str(payload.get("description", "")).strip(),
                        keywords=keywords,
                    )
                    if not description:
                        raise ValueError("empty generated description")

                    if prefix and prefix.lower() not in description.lower():
                        description = f"{prefix}\n\n{description}".strip()

                    description = description[:5000].rstrip()
                    return title, description
                except Exception as exc:
                    if attempt >= max_retries:
                        self.log(
                            f"Gemini metadata generation failed for key {key_index} "
                            f"after {attempt} attempts: {exc}"
                        )
                    else:
                        self.log(
                            f"Gemini metadata retry key {key_index} attempt {attempt}/{max_retries}: {exc}"
                        )
                        time.sleep(1.0)
        return None

    def _sleep_until(self, target: datetime) -> None:
        while True:
            now = now_tz(self.tz)
            remaining = (target - now).total_seconds()
            if remaining <= 0:
                return
            chunk = min(float(self.args.sleep_check_interval_seconds), remaining)
            self.log(f"Sleeping {chunk:.0f}s (until {target.strftime('%Y-%m-%d %H:%M:%S %Z')})")
            time.sleep(max(1.0, chunk))

    def _ensure_branches(self) -> None:
        need_crawl = True
        if self.branches_path.exists():
            try:
                existing = json.loads(self.branches_path.read_text(encoding="utf-8"))
                need_crawl = not isinstance(existing, list) or not existing
            except json.JSONDecodeError:
                need_crawl = True

        if need_crawl:
            self.log("Branch list missing/empty. Crawling branches now...")
            cmd = [
                self.args.python_bin,
                self._script_path("amazon_spider.py"),
                "--department-name",
                self.args.department_name,
                "--output",
                str(self.branches_path),
                "--speed",
                self.args.branch_crawl_speed,
                "--max-pages",
                str(self.args.branch_max_pages),
            ]
            if self.args.follow_other_departments:
                cmd.append("--follow-other-departments")
            self._run_cmd(cmd)

        raw = json.loads(self.branches_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise PipelineError("branches.json is not a JSON list.")

        filtered: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            url = str(item.get("branch_url", "")).strip()
            if not url or url in seen:
                continue
            if not self.args.include_up_links and bool(item.get("is_up_link")):
                continue
            seen.add(url)
            filtered.append(item)

        if not filtered:
            raise PipelineError("No branch URLs available after filtering.")

        self.branches = filtered
        self.log(f"Loaded {len(self.branches)} branches.")

    def _recover_skipped_with_ready_reels(self) -> None:
        if not self.state.skipped_urls:
            return

        recovered: list[str] = []
        remaining: list[str] = []
        rewind_to: int | None = None
        for url in self.state.skipped_urls:
            asin = extract_asin(url)
            if not asin:
                remaining.append(url)
                continue
            ready = any(self.reels_dir.glob(f"{asin}_*_square_reel_complete.mp4"))
            if ready:
                recovered.append(url)
                try:
                    idx = self.state.branch_products.index(url)
                    rewind_to = idx if rewind_to is None else min(rewind_to, idx)
                except ValueError:
                    pass
            else:
                remaining.append(url)

        if recovered:
            self.state.skipped_urls = remaining
            if rewind_to is not None:
                self.state.product_index = min(self.state.product_index, rewind_to)
            self._save_state()
            self.log(
                f"Recovered {len(recovered)} skipped URL(s) because ready reels exist. "
                "They will be retried."
            )

    def _reset_branch_progress(self, new_index: int) -> None:
        self.state.branch_index = int(new_index)
        self.state.product_index = 0
        self.state.current_branch_url = ""
        self.state.current_branch_name = ""
        self.state.branch_products = []
        self._save_state()

    def _mark_branch_finished(self) -> None:
        self._reset_branch_progress(self.state.branch_index + 1)

    def _load_products_for_current_branch(self) -> None:
        if self.state.branch_index >= len(self.branches):
            if self.args.stop_when_all_branches_done:
                raise StopIteration("All branches completed.")
            self.log("All branches completed. Recrawling branch list and restarting from first branch.")
            self._ensure_branches()
            self._reset_branch_progress(0)

        branch = self.branches[self.state.branch_index]
        branch_url = str(branch.get("branch_url", "")).strip()
        branch_name = str(branch.get("branch_name", "")).strip() or f"branch_{self.state.branch_index}"
        self.log(
            f"Loading products for branch index {self.state.branch_index}: {branch_name} ({branch_url})"
        )

        cmd = [
            self.args.python_bin,
            self._script_path("amazon_products_spider.py"),
            "--branch-url",
            branch_url,
            "--output",
            str(self.current_products_path),
            "--max-pages",
            str(self.args.product_max_pages),
        ]
        self._run_cmd(cmd)

        payload = json.loads(self.current_products_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise PipelineError("Branch product list is not a JSON list.")
        products = [str(url).strip() for url in payload if isinstance(url, str) and str(url).strip()]

        # Hard guarantee: never process same product URL twice in a branch list.
        deduped_products: list[str] = []
        seen_products: set[str] = set()
        for url in products:
            if url in seen_products:
                continue
            seen_products.add(url)
            deduped_products.append(url)
        products = deduped_products

        completed_urls = set(self.state.uploaded_urls) | set(self.state.skipped_urls)
        products = [url for url in products if url not in completed_urls]

        self.state.current_branch_url = branch_url
        self.state.current_branch_name = branch_name
        self.state.branch_products = products
        self.state.product_index = 0
        self._save_state()
        self.log(f"Loaded {len(products)} pending product URLs for current branch.")

    def _next_product_url(self) -> str:
        while True:
            if not self.state.branch_products or self.state.product_index >= len(self.state.branch_products):
                self._load_products_for_current_branch()

            if not self.state.branch_products:
                self.log("Current branch has no pending products. Moving to next branch.")
                self._mark_branch_finished()
                continue

            if self.state.product_index >= len(self.state.branch_products):
                self.log("Reached end of product list for branch. Moving to next branch.")
                self._mark_branch_finished()
                continue

            url = self.state.branch_products[self.state.product_index]
            self.state.product_index += 1
            self._save_state()
            return url

    def _latest_gemini_audio_for_asin(self, asin: str, before_dirs: set[str]) -> Path | None:
        candidates = []
        for p in self.gemini_outputs_dir.glob(f"{asin}_*"):
            if not p.is_dir():
                continue
            if str(p.resolve()) in before_dirs:
                continue
            wav = p / f"{asin}_promo_ml.wav"
            if wav.exists():
                candidates.append(wav)
        if not candidates:
            for p in self.gemini_outputs_dir.glob(f"{asin}_*"):
                wav = p / f"{asin}_promo_ml.wav"
                if wav.exists():
                    candidates.append(wav)
        if not candidates:
            return None
        candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return candidates[0]

    def _is_within_runtime(self, path: Path) -> bool:
        try:
            resolved = path.expanduser().resolve()
            base = self.runtime_dir.resolve()
            return str(resolved).startswith(str(base))
        except Exception:
            return False

    def _cleanup_uploaded_artifacts(self, prepared: PreparedReel) -> None:
        targets = []
        seen: set[str] = set()
        for path in prepared.cleanup_paths:
            key = str(path.expanduser().resolve())
            if key in seen:
                continue
            seen.add(key)
            targets.append(path)

        for path in targets:
            try:
                target = path.expanduser().resolve()
            except Exception:
                continue

            if not self._is_within_runtime(target):
                self.log(f"Skip cleanup outside runtime: {target}")
                continue

            if not target.exists():
                continue

            try:
                if target.is_dir():
                    shutil.rmtree(target, ignore_errors=True)
                    self.log(f"Deleted directory: {target}")
                else:
                    target.unlink(missing_ok=True)
                    self.log(f"Deleted file: {target}")
            except Exception as exc:
                self.log(f"Cleanup warning for {target}: {exc}")

        # Prune empty runtime folders created per product.
        for root in (self.download_dir, self.gemini_outputs_dir, self.reels_dir):
            try:
                for child in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                    if child.is_dir():
                        try:
                            child.rmdir()
                        except OSError:
                            pass
            except Exception:
                pass

    def _prepare_reel_for_product(self, product_url: str, slot_time: datetime) -> PreparedReel | None:
        asin = extract_asin(product_url) or "unknown"
        self.log(f"Preparing product {product_url} (ASIN={asin})")

        asin_download_dir = self.download_dir / asin
        cached_results_path = asin_download_dir / f"{asin}_results.json"

        # ── Step 1: Video download (skip if cached) ────────────────────
        downloaded_video: Path | None = None
        report: dict[str, Any] | None = None

        if cached_results_path.exists():
            try:
                report = json.loads(cached_results_path.read_text(encoding="utf-8"))
                rows = report.get("results", [])
                if rows and isinstance(rows[0], dict):
                    vids = rows[0].get("downloaded_videos", [])
                    if vids:
                        candidate = Path(str(vids[0])).expanduser().resolve()
                        if candidate.exists():
                            downloaded_video = candidate
                            self.log(f"Reusing cached video: {downloaded_video.name}")
            except Exception:
                report = None

        if downloaded_video is None:
            cmd_video = [
                self.args.python_bin,
                self._script_path("amazon_product_video_checker.py"),
                "--url",
                product_url,
                "--output",
                str(self.current_result_path),
                "--download-dir",
                str(self.download_dir),
                "--workers",
                "1",
                "--max-videos-per-product",
                "1",
                "--timeout-seconds",
                str(self.args.http_timeout_seconds),
                "--ffmpeg-timeout-seconds",
                str(self.args.ffmpeg_timeout_seconds),
            ]
            self._run_cmd(cmd_video)

            report = json.loads(self.current_result_path.read_text(encoding="utf-8"))
            results = report.get("results", [])
            if not isinstance(results, list) or not results:
                self.log("No product result row found; skipping URL.")
                return None
            row = results[0] if isinstance(results[0], dict) else {}
            has_video = bool(row.get("has_video"))
            downloaded = row.get("downloaded_videos", [])
            if not has_video or not isinstance(downloaded, list) or not downloaded:
                self.log("Product has no downloadable video. Skipping.")
                return None
            downloaded_video = Path(str(downloaded[0])).expanduser().resolve()
            if not downloaded_video.exists():
                self.log(f"Downloaded video path missing: {downloaded_video}. Skipping.")
                return None

            # Cache results JSON alongside the video for crash-resume.
            asin_download_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.current_result_path, cached_results_path)
            self.log(f"Execution time: {cached_results_path.name} cached.")

        # Extract product metadata row from whichever report we have.
        row = {}
        if report:
            results = report.get("results", [])
            if isinstance(results, list) and results and isinstance(results[0], dict):
                row = results[0]
        results_json_for_scripts = str(
            cached_results_path if cached_results_path.exists() else self.current_result_path
        )

        # ── Step 2: QR code (always regenerate — fast) ─────────────────
        cmd_qr = [
            self.args.python_bin,
            self._script_path("product_url_to_qr.py"),
            "--url",
            product_url,
            "--affiliate-tag",
            self.args.affiliate_tag,
            "--output",
            str(self.qr_output),
        ]
        self._run_cmd(cmd_qr)

        # ── Step 3: Gemini audio (skip if cached) ──────────────────────
        promo_audio = self._latest_gemini_audio_for_asin(asin, set())
        gemini_output_dir: Path | None = None

        if promo_audio is not None and promo_audio.exists():
            self.log(f"Reusing cached Gemini audio: {promo_audio.name}")
            gemini_output_dir = promo_audio.parent
        else:
            before_dirs = {
                str(p.resolve())
                for p in self.gemini_outputs_dir.glob(f"{asin}_*")
                if p.is_dir()
            }
            cmd_gemini = [
                self.args.python_bin,
                self._script_path("gemini_video_promoter_ml.py"),
                "--results-json",
                results_json_for_scripts,
                "--video-path",
                str(downloaded_video),
                "--output-dir",
                str(self.gemini_outputs_dir),
                "--voice-name",
                self.args.gemini_voice_name,
                "--analysis-model",
                self.args.gemini_analysis_model,
                "--tts-model",
                self.args.gemini_tts_model,
                "--scene-count",
                str(self.args.scene_count),
                "--voice-timing-mode",
                self.args.voice_timing_mode,
            ]
            if self.args.cta_text.strip():
                cmd_gemini.extend(["--cta-text", self.args.cta_text.strip()])
            if self.args.api_keys_file.strip():
                cmd_gemini.extend(["--api-keys-file", self.args.api_keys_file.strip()])
            self._run_cmd(cmd_gemini)

            promo_audio = self._latest_gemini_audio_for_asin(asin, before_dirs)
            if promo_audio is None or not promo_audio.exists():
                self.log("Could not locate generated promo audio. Skipping.")
                return None
            gemini_output_dir = promo_audio.parent

        # ── Step 4: Square reel (skip if cached) ───────────────────────
        existing_complete = sorted(
            self.reels_dir.glob(f"{asin}_*_square_reel_complete.mp4"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if existing_complete:
            complete_output = existing_complete[0]
            square_output = Path(
                str(complete_output).replace("_complete.mp4", ".mp4")
            )
            self.log(f"Reusing existing reel: {complete_output.name}")
        else:
            slot_tag = slot_time.strftime("%Y%m%d_%H%M")
            square_output = self.reels_dir / f"{asin}_{slot_tag}_square_reel.mp4"
            complete_output = self.reels_dir / f"{asin}_{slot_tag}_square_reel_complete.mp4"

            cmd_reel = [
                self.args.python_bin,
                self._script_path("make_square_reel.py"),
                "--video",
                str(downloaded_video),
                "--audio",
                str(promo_audio),
                "--output",
                str(square_output),
                "--complete-output",
                str(complete_output),
                "--cta-video",
                self.args.cta_video,
                "--qr-image",
                str(self.qr_output),
            ]
            if self.args.no_final_music:
                cmd_reel.append("--no-final-music")
            else:
                if self.args.final_music.strip():
                    cmd_reel.extend(["--final-music", self.args.final_music.strip()])
                elif self.args.use_default_final_music:
                    cmd_reel.append("--use-default-final-music")
            self._run_cmd(cmd_reel)

            if not complete_output.exists():
                self.log(f"Expected final reel missing: {complete_output}")
                return None

        return PreparedReel(
            product_url=product_url,
            asin=asin,
            product_title=str(row.get("title", "")).strip(),
            product_description=str(row.get("description", "")).strip(),
            final_video_path=complete_output,
            cleanup_paths=[
                downloaded_video,
                cached_results_path,
                gemini_output_dir,
                square_output,
                complete_output,
                self.qr_output,
                self.current_result_path,
            ],
        )

    def _build_youtube_title(self, prepared: PreparedReel) -> str:
        base = prepared.product_title.strip() or f"Amazon Product {prepared.asin}"
        prefix = self.args.youtube_title_prefix.strip()
        if prefix:
            title = f"{prefix} | {base}"
        else:
            title = base
        title = re.sub(r"\s+", " ", title).strip()
        return self._ensure_malayalam_suffix_title(title)

    def _build_youtube_description(self, prepared: PreparedReel) -> str:
        lines = []
        if self.args.youtube_description_prefix.strip():
            lines.append(self.args.youtube_description_prefix.strip())
        if prepared.product_description:
            clean = self._sanitize_product_description(prepared.product_description)[:1500].strip()
            lines.append(clean)
        description = "\n\n".join([x for x in lines if x]).strip()
        return self._normalize_description_no_links(description)

    def _record_upload(self, product_url: str, youtube_url: str = "") -> None:
        """Append to durable upload log (crash-safe — written before cleanup)."""
        entry = {
            "url": product_url,
            "youtube_url": youtube_url,
            "time": now_tz(self.tz).isoformat(),
        }
        try:
            with open(self.upload_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exc:
            self.log(f"Warning: could not write upload log: {exc}")

    def _merge_upload_log(self) -> None:
        """On startup, merge any uploads recorded in the durable log into state."""
        if not self.upload_log_path.exists():
            return
        try:
            logged_urls: set[str] = set()
            for line in self.upload_log_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    url = entry.get("url", "")
                    if url:
                        logged_urls.add(url)
                except Exception:
                    continue
            added = 0
            for url in logged_urls:
                if url not in self.state.uploaded_urls:
                    self.state.uploaded_urls.append(url)
                    added += 1
            if added:
                self._save_state()
                self.log(f"Merged {added} URLs from upload log into state.")
        except Exception as exc:
            self.log(f"Warning: could not merge upload log: {exc}")

    def _upload_reel(self, prepared: PreparedReel) -> str:
        """Upload reel and return the YouTube video URL."""
        title = self._build_youtube_title(prepared)
        description = self._build_youtube_description(prepared)
        if self.args.youtube_metadata_mode == "gemini":
            generated = self._gemini_generate_youtube_metadata(prepared)
            if generated is not None:
                title, description = generated
                self.log(f"Using Gemini-generated YouTube metadata for ASIN {prepared.asin}")
            else:
                self.log("Falling back to static YouTube metadata template.")
        cmd = [
            self.args.python_bin,
            self._script_path("youtube_uploader_api.py"),
            "--video",
            str(prepared.final_video_path),
            "--title",
            title,
            "--description",
            description,
            "--privacy",
            self.args.youtube_privacy,
            "--category-id",
            self.args.youtube_category_id,
            "--client-secrets",
            self.args.client_secrets,
            "--token-file",
            self.args.token_file,
            "--metadata-mode",
            "off",
        ]
        output = self._run_cmd(cmd)

        # Extract YouTube URL from uploader stdout.
        youtube_url = ""
        for line in (output or "").splitlines():
            match = re.search(r"https://www\.youtube\.com/watch\?v=\S+", line)
            if match:
                youtube_url = match.group(0)
                break
        if youtube_url:
            self.log(f"URL: {youtube_url}")
        return youtube_url

    def _prepare_for_slot(self, slot_time: datetime) -> PreparedReel | None:
        deadline = slot_time - timedelta(minutes=max(1, self.args.min_minutes_before_upload))
        while True:
            now = now_tz(self.tz)
            if now >= deadline:
                self.log("Preparation deadline reached for this slot.")
                return None
            try:
                product_url = self._next_product_url()
            except StopIteration:
                self.log("All branches complete and stop flag enabled.")
                return None

            try:
                prepared = self._prepare_reel_for_product(product_url, slot_time)
            except Exception as exc:
                err_str = str(exc)
                self.log(f"Product processing failed, skipping URL: {product_url} | {err_str}")
                prepared = None
                # Fatal: all Gemini keys are revoked/leaked — stop wasting time.
                if "leaked" in err_str.lower() or (
                    "PERMISSION_DENIED" in err_str and "API key" in err_str
                ):
                    self.log(
                        "FATAL: All Gemini API keys appear to be revoked/leaked. "
                        "Generate new keys at https://aistudio.google.com/apikey "
                        "and update .env with new keys. Stopping."
                    )
                    return None

            if prepared is not None:
                return prepared

            if product_url not in self.state.skipped_urls:
                self.state.skipped_urls.append(product_url)
                self._save_state()

    def run_forever(self) -> None:
        self._ensure_branches()
        if self.args.reset_skipped_urls:
            removed = len(self.state.skipped_urls)
            self.state.skipped_urls = []
            self._save_state()
            self.log(f"Reset skipped URLs list ({removed} entries removed).")
        elif self.args.recover_skipped_uploads:
            self._recover_skipped_with_ready_reels()

        # Merge any uploads from a previous crashed run.
        self._merge_upload_log()

        if self.args.test_mode:
            self.log(
                "Scheduler started in TEST MODE: timing/waiting disabled. "
                "Runs will execute immediately."
            )
            runs = max(1, int(self.args.test_max_runs))
            for run_idx in range(1, runs + 1):
                self.log(f"Test run {run_idx}/{runs} starting...")
                slot_time = now_tz(self.tz) + timedelta(days=3650)
                prepared = self._prepare_for_slot(slot_time)
                if prepared is None:
                    self.log("No publishable reel ready in test mode. Stopping test runs.")
                    break
                try:
                    youtube_url = self._upload_reel(prepared)
                except Exception as exc:
                    self.log(f"Upload failed for {prepared.product_url}: {exc}")
                    self.state.upload_error_by_url[prepared.product_url] = str(exc)
                    # Retry same product on next cycle instead of skipping permanently.
                    self.state.product_index = max(0, self.state.product_index - 1)
                    self._save_state()
                    continue

                # Record upload immediately (durable log — survives crashes).
                self._record_upload(prepared.product_url, youtube_url)

                self.state.upload_error_by_url.pop(prepared.product_url, None)
                if prepared.product_url not in self.state.uploaded_urls:
                    self.state.uploaded_urls.append(prepared.product_url)
                self.state.last_upload_time = now_tz(self.tz).isoformat()
                self.state.last_uploaded_url = prepared.product_url
                self.state.last_completed_slot = "test_mode"
                self._save_state()
                self.log(f"Uploaded successfully (test mode): {prepared.product_url}")
                self._cleanup_uploaded_artifacts(prepared)
            self.log("Test mode complete.")
            return

        self.log(
            "Scheduler started. Slots: "
            + ", ".join(t.strftime("%H:%M") for t in self.slots)
            + f" | Wake before: {self.args.wake_before_minutes} minutes"
        )

        while True:
            now = now_tz(self.tz)
            slot_time = next_slot_after(now, self.slots)
            wake_time = slot_time - timedelta(minutes=self.args.wake_before_minutes)

            if now < wake_time:
                self.log(
                    f"Sleeping now. Next wake-up: {wake_time.strftime('%Y-%m-%d %H:%M:%S %Z')} "
                    f"for slot {slot_time.strftime('%Y-%m-%d %H:%M')}"
                )
                self._sleep_until(wake_time)

            self.log(f"Wake phase started for slot {slot_time.strftime('%Y-%m-%d %H:%M')}")
            prepared = self._prepare_for_slot(slot_time)

            now = now_tz(self.tz)
            if now < slot_time:
                self.log(f"Waiting for exact publish slot at {slot_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                self._sleep_until(slot_time)

            if prepared is None:
                self.log("No publishable reel ready for this slot. Skipping upload.")
                continue

            try:
                youtube_url = self._upload_reel(prepared)
            except Exception as exc:
                self.log(f"Upload failed for {prepared.product_url}: {exc}")
                self.state.upload_error_by_url[prepared.product_url] = str(exc)
                # Keep URL eligible for retry by rewinding index.
                self.state.product_index = max(0, self.state.product_index - 1)
                self._save_state()
                continue

            # Record upload immediately (durable log — survives crashes).
            self._record_upload(prepared.product_url, youtube_url)

            self.state.upload_error_by_url.pop(prepared.product_url, None)
            if prepared.product_url not in self.state.uploaded_urls:
                self.state.uploaded_urls.append(prepared.product_url)
            self.state.last_upload_time = now_tz(self.tz).isoformat()
            self.state.last_uploaded_url = prepared.product_url
            self.state.last_completed_slot = slot_time.isoformat()
            self._save_state()
            self.log(f"Uploaded successfully: {prepared.product_url}")
            self._cleanup_uploaded_artifacts(prepared)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()
    project_root = here.parent
    for parent in [here.parent] + list(here.parents):
        if (parent / "requirements.txt").exists():
            project_root = parent
            break
    parser = argparse.ArgumentParser(
        description="Fixed-time daily auto publisher for the YT Auto pipeline."
    )
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable for child scripts.")
    parser.add_argument("--project-root", default=str(project_root), help="Project root path.")
    parser.add_argument("--runtime-dir", default=str(project_root / "data" / "runtime"), help="Runtime working directory.")
    parser.add_argument("--timezone", default="Asia/Kolkata", help="IANA timezone name.")
    parser.add_argument(
        "--schedule-slots",
        default="11:30,17:30,19:30",
        help="Comma-separated daily publish slots in HH:MM 24h format.",
    )
    parser.add_argument(
        "--wake-before-minutes",
        type=int,
        default=30,
        help="Wake this many minutes before each slot.",
    )
    parser.add_argument(
        "--min-minutes-before-upload",
        type=int,
        default=1,
        help="Stop preparing new product this many minutes before slot.",
    )
    parser.add_argument(
        "--sleep-check-interval-seconds",
        type=int,
        default=120,
        help="Sleep loop check interval.",
    )
    parser.add_argument(
        "--test-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        # default=True,
        help="Enable immediate test runs (no slot timing or waiting).",
    )
    parser.add_argument(
        "--test-max-runs",
        type=int,
        default=1,
        help="How many immediate runs to execute when --test-mode is enabled.",
    )

    parser.add_argument("--department-name", default="Home & Kitchen", help="Seed department name for branch crawl.")
    parser.add_argument("--follow-other-departments", action="store_true", help="Follow external departments in branch crawl.")
    parser.add_argument("--branch-crawl-speed", choices=["safe", "fast", "ultra"], default="fast")
    parser.add_argument("--branch-max-pages", type=int, default=0, help="Max pages for branch crawler (0=no limit).")
    parser.add_argument("--product-max-pages", type=int, default=0, help="Max pages for product crawl (0=no limit).")
    parser.add_argument("--include-up-links", action="store_true", help="Include branch items marked as up-links.")
    parser.add_argument("--stop-when-all-branches-done", action="store_true", help="Exit when all branches are completed.")
    parser.add_argument(
        "--recover-skipped-uploads",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-unskip URLs when corresponding complete reels are already present in runtime/reels.",
    )
    parser.add_argument(
        "--reset-skipped-urls",
        action="store_true",
        help="Clear skipped_urls at startup (useful after fixing pipeline issues).",
    )

    parser.add_argument("--affiliate-tag", default="shozi01-21", help="Affiliate tag for QR URL.")
    parser.add_argument("--api-keys-file", default="", help="Gemini keys file path.")
    parser.add_argument("--gemini-voice-name", default="Kore", help="Gemini TTS voice name.")
    parser.add_argument("--gemini-analysis-model", default="gemini-2.5-flash", help="Gemini analysis model.")
    parser.add_argument("--gemini-tts-model", default="gemini-2.5-flash-preview-tts", help="Gemini TTS model.")
    parser.add_argument("--scene-count", type=int, default=3, help="Scene count for Gemini script generation.")
    parser.add_argument(
        "--voice-timing-mode",
        choices=["natural_no_stretch", "global_stretch"],
        default="natural_no_stretch",
        help="Narration timing mode for Gemini script.",
    )
    parser.add_argument("--cta-text", default="", help="Optional CTA text for Malayalam narration.")

    parser.add_argument("--http-timeout-seconds", type=int, default=25, help="HTTP timeout for product checker.")
    parser.add_argument("--ffmpeg-timeout-seconds", type=int, default=600, help="ffmpeg timeout for downloads.")

    parser.add_argument("--cta-video", default=str(project_root / "assets" / "video" / "CTA YT.mp4"), help="CTA video path.")
    parser.add_argument("--final-music", default="", help="Optional final music file path.")
    parser.add_argument("--use-default-final-music", action="store_true", help="Use make_square_reel default final music.")
    parser.add_argument("--no-final-music", action="store_true", help="Disable final music in complete reel.")

    parser.add_argument("--client-secrets", default=str(project_root / "secrets" / "client_secret.json"), help="YouTube OAuth client secrets JSON.")
    parser.add_argument("--token-file", default=str(project_root / "secrets" / "youtube_token.json"), help="YouTube token JSON.")
    parser.add_argument("--youtube-privacy", choices=["private", "unlisted", "public"], default="public")
    parser.add_argument("--youtube-category-id", default="22", help="YouTube category ID.")
    parser.add_argument("--youtube-title-prefix", default="Daily Amazon Reel", help="Prefix for YouTube title.")
    parser.add_argument("--youtube-description-prefix", default="Malayalam product reel.", help="Prefix for YouTube description.")
    parser.add_argument(
        "--youtube-metadata-mode",
        choices=["gemini", "static"],
        default="gemini",
        help="Generate upload title/description with Gemini or use static template fallback.",
    )
    parser.add_argument(
        "--youtube-metadata-model",
        default="gemini-2.5-flash",
        help="Gemini model for dynamic YouTube metadata generation.",
    )
    parser.add_argument(
        "--youtube-metadata-max-retries",
        type=int,
        default=2,
        help="Max retries per Gemini key for generating YouTube metadata.",
    )
    # Load .env from project root
    load_dotenv(project_root / ".env")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    publisher = AutoPublisher(args)
    publisher.run_forever()


if __name__ == "__main__":
    main()
