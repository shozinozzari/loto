# YT Auto - Complete Project Analysis

## 1. Project purpose

This project automates an Amazon-to-YouTube short-video workflow:

1. Crawl Amazon India best-seller category branches.
2. Crawl product URLs from a chosen branch.
3. Check each product page for embedded videos and download product videos.
4. Generate a product affiliate QR code.
5. Use Gemini to analyze a product video and generate Malayalam promo narration audio.
6. Build a square reel (`1080x1080`) by combining product video + generated narration.
7. Append a CTA clip with QR overlay.
8. Optionally add background music.
9. Upload final reel to YouTube via OAuth API.

The repo is workflow-first (many standalone scripts), not a single packaged app.

## 2. Pipeline map (actual script chain)

Typical execution order:

1. `amazon_spider.py` -> `branches.json`
2. `amazon_products_spider.py` -> `product.json`
3. `amazon_product_video_checker.py` -> `product_video_results.json` + `downloaded_videos/...`
4. `product_url_to_qr.py` -> `amazon_product_qr.png`
5. `gemini_video_promoter_ml.py` -> `gemini_outputs/<ASIN_timestamp>/...promo_ml.wav` (+ script txt/json)
6. `make_square_reel.py` -> `reel/<video>_square_reel.mp4` and `..._complete.mp4`
7. `youtube_uploader_api.py` -> uploads final reel

Helper-only scripts exist for individual steps (`download_first_video_from_results.py`, `add_qr_to_cta_video.py`).

## 3. Core scripts (detailed behavior)

### 3.1 `amazon_spider.py` (main branch crawler, advanced)

What it does:

- Crawls Amazon best-seller branch tree (default department: `Home & Kitchen`).
- Normalizes best-seller URLs (strips `/ref=` suffixes, query/fragment).
- Emits unique branch entries with metadata: name, URL, source page, depth, up-link flag, cluster key.
- Supports optional follow outside start department.

Important implementation details:

- Department names are mapped to known Amazon slugs through `DEPARTMENT_NAME_TO_SLUG`.
- If department is unknown, slug is guessed by normalization (`& -> and`, non-alnum -> `-`).
- Includes a seed item (`is_seed=true`) before normal nav traversal.
- Uses round-robin clustered scheduling by branch cluster key to balance deep-tree crawling.
- Dedup uses canonical URL via `BranchDedupePipeline`.

Resource scaling:

- Contains `AdaptiveResourceScalerExtension` that dynamically adjusts Scrapy downloader concurrency using:
  - CPU usage (`psutil`)
  - RAM pressure
  - Optional GPU stats from `nvidia-smi` (`utilization.gpu`, memory used/total)
- Computes min/start/max concurrency from hardware capacity.
- Supports speed profiles: `safe`, `fast`, `ultra`.

Default output:

- `<project_root>/data/branches.json`

### 3.2 `amazon_spider copy.py` (older/simple variant)

What it does:

- Earlier, simpler crawler for the same branch tree problem.
- Has static speed profiles (`safe/fast/ultra`) but no runtime hardware autoscaler.
- Similar round-robin branch scheduling and dedupe behavior.

Main difference from `amazon_spider.py`:

- No dynamic CPU/RAM/GPU feedback loop.
- Uses direct start URL instead of strong department-name-to-slug resolution flow.

### 3.3 `amazon_products_spider.py`

What it does:

- Reads first valid branch URL from `branches.json` (or `--branch-url` override).
- Crawls paginated best-seller listing pages.
- Extracts canonical product URLs (`https://www.amazon.in/dp/<ASIN>`).
- Writes output as JSON array of URLs.

Extraction details:

- Extracts links from card anchors and `/dp/` or `/gp/product/` links.
- Also extracts ASINs from page attributes:
  - `data-asin`
  - `data-client-recs-list` JSON or regex fallback.
- Dedupe pipeline avoids duplicate product URLs.

Pagination behavior:

- First preference: explicit next-page selectors.
- Fallback: pagination-number links.
- Last fallback: synthetic page URL pattern when pagination UI is missing.

Default output:

- `<project_root>/data/product.json`

### 3.4 `amazon_product_video_checker.py`

What it does:

- Loads product URLs (`product.json` by default).
- Requests each product page with custom headers.
- Detects if product has video.
- Scrapes product title and description points.
- Extracts video media URLs (`.m3u8`/`.mp4`).
- Optionally downloads first N videos per product via `ffmpeg`.

Concurrency:

- Uses `ThreadPoolExecutor`.
- Default workers: `max(4, min(24, cpu_count * 4))`.

Detection signals:

- `videoThumbnail` classes
- `openVideoImmersiveView`/`chromeful-video` markers
- `videoCount` label pattern
- direct media URL presence

Bot challenge detection:

- Flags captcha/challenge phrases in HTML.

Download behavior:

- Uses `ffmpeg -c copy` with custom user-agent and timeout.
- Output path pattern:
  - `downloaded_videos/<ASIN>/<ASIN>_video_01.mp4`

Output report:

- `product_video_results.json` with:
  - `summary` totals
  - `results[]` entries including title/description/video URLs/download info/errors/timing.

Observed repo sample summary:

- total: `100`
- with_video: `92`
- without_video: `8`
- blocked: `0`
- downloaded_videos: `92`

### 3.5 `download_first_video_from_results.py`

What it does:

- Utility script to download only first product's first video from `product_video_results.json`.
- Uses `ffmpeg` copy mode.
- Output:
  - `downloaded_videos/<ASIN>/<ASIN>_video_01.mp4`

### 3.6 `product_url_to_qr.py`

What it does:

- Builds affiliate URL by injecting/replacing `tag=` query param.
- Generates QR code PNG using `qrcode` library.

Defaults:

- URL: `https://www.amazon.in/dp/B07WMS7TWB`
- Affiliate tag: `shozi01-21`
- Output: `amazon_product_qr.png`

### 3.7 `add_qr_to_cta_video.py`

What it does:

- Overlays QR image on CTA video for first N seconds (default `2.9s`) using ffmpeg filter graph.
- Supports corner positioning and sizing options.

Important detail:

- `USE_HARDCODED_QR_SIZE = True` by default.
- When true, CLI size flags are ignored and hardcoded size (`470`) is used.

Default files:

- Input video: `CTA YT.mp4`
- QR image: `amazon_product_qr.png`
- Output: `CTA YT_with_qr.mp4`

### 3.8 `gemini_video_promoter_ml.py` (largest script, AI core)

What it does:

- Picks product video (default: first product's first downloaded video in `product_video_results.json`).
- Uploads video to Gemini Files API.
- Waits for uploaded file to become `ACTIVE`.
- Requests visual analysis + Malayalam promo script generation.
- Enforces hook/CTA and anti-repetition cleanup.
- Generates Malayalam TTS audio (`gemini-2.5-flash-preview-tts`, default voice `Kore`).
- Applies timing correction so audio duration targets `video_duration + overrun`.
- Saves final script and WAV; optional debug artifacts.

Input key loading order:

1. `--api-key` (repeatable)
2. `--api-keys-file` (or env `GEMINI_KEYS_FILE`, else default `gemini_keys.txt`)
3. env `GEMINI_API_KEYS`
4. env `GEMINI_API_KEY`

Rate-limit and retry strategy:

- Detects rate-limit/quota/transient/network errors.
- Extracts retry delay from exception text when available.
- Rotates through multiple API keys.
- Supports max key cycles and configurable per-call retry counts.

Prompting and script constraints:

- Prompts force JSON schema with visual summary, highlights, hook, audience, benefits, promo script, CTA, scene segments.
- Strong constraints against repeated lines and duplicated phrasing.
- Enforces spoken-pace and duration targeting.
- If script too short/long by word targets, runs expand/shorten refinement prompts.
- Hook line source priority:
  - user override -> analysis output -> generated hook -> first sentence fallback.

Audio duration control:

- Modes:
  - `natural_no_stretch`: mostly keep natural speech, only corrective speed-up/pad if outside allowed overrun range.
  - `global_stretch`: force-fit audio toward target duration.
- Uses ffmpeg `atempo` chains and pad/trim logic.

Output folder pattern:

- `gemini_outputs/<ASIN>_<YYYYMMDD_HHMMSS>/`

Main output files:

- `<ASIN>_promo_ml.txt` (Malayalam script)
- `<ASIN>_promo_ml.wav` (final voice audio)
- Optional when `--keep-debug-files`:
  - `<ASIN>_analysis.json`
  - `<ASIN>_analysis_raw.txt`
  - `<ASIN>_scene_segments.json`
  - `<ASIN>_run_manifest.json`
  - intermediate raw/sped/concat WAV files

### 3.9 `make_square_reel.py`

What it does:

- Creates square reel (`1:1`) from product video + narration audio.
- Automatically aligns video duration to narration by changing video speed (`setpts`).
- Background mode:
  - blurred enlarged video (`default`)
  - black background
- Optional full finalization:
  - overlays QR on CTA clip in two time windows
  - concatenates base reel + CTA
  - optionally mixes background music into final reel audio

Auto-discovery behavior (when no manual inputs):

- Finds latest `*_video_01.mp4` in `downloaded_videos/`
- Finds matching/latest `*_promo_ml.wav` in `gemini_outputs/`
- Writes output under `reel/`

CTA overlay details:

- Two QR display windows on CTA clip:
  - `0 -> cta-overlay-first-end` (default `2.9`)
  - `cta-overlay-start -> cta-overlay-end` (default start `7.9`, end `<0` means till clip end)

Default assets:

- CTA video: `CTA YT.mp4`
- QR image: `amazon_product_qr.png`
- default music candidate: `reel Music.mp3`

Output naming:

- square reel: `<video_stem>_square_reel.mp4`
- complete reel: `<square_stem>_complete.mp4`

### 3.10 `youtube_uploader_api.py`

What it does:

- Uploads final video to YouTube Data API v3 (`videos.insert`).
- Uses OAuth desktop flow on first run, then stores token file.
- Supports privacy, title, description, category.

Default files:

- client secrets: `client_secret.json`
- token: `youtube_token.json`
- default video: `reel/reel.mp4`

## 4. Secondary/experimental script

### `test - Copy copy 2.py`

- Standalone Gemini TTS test utility.
- Loads `.env` from a completely different project path:
  - `C:\Users\Shozin\Desktop\AUTO LEADS PROJECT\AI_Video_Funnel\backend\.env`
- Uses `GOOGLE_API_KEY` env var and writes `output_<voice>.wav`.
- Not wired into main pipeline.

## 5. Data and artifact files in this repo

Key generated/input artifacts observed:

- `branches.json`:
  - 879 branch records
  - depth distribution: 1:1, 2:20, 3:108, 4:467, 5:263, 6:20
- `all_departments.json`:
  - 31 department slug mappings (Amazon best-seller root)
- `product.json`:
  - 100 canonical product URLs
- `first_branch_product_urls.txt`:
  - line-by-line copy of product URLs
- `product_video_results.json`:
  - full per-product video detection + download report
- `downloaded_videos/<ASIN>/...mp4`:
  - downloaded source product videos
- `gemini_outputs/<ASIN_timestamp>/`:
  - generated Malayalam script and audio (+ optional debug files)
- `reel/`:
  - square and complete reel outputs

Media assets used by composition scripts:

- `CTA YT.mp4`
- `Top Banner.mp4`
- `reel Music.mp3`
- `amazon_product_qr.png`
- additional sample video/audio/image files

## 6. Dependencies and runtime requirements

From `requirements.txt`:

- `playwright`
- `google-api-python-client`
- `google-auth-oauthlib`
- `scrapy`
- `psutil`
- `qrcode[pil]`

Also required at runtime (not fully listed in `requirements.txt`):

- `ffmpeg` and `ffprobe` on PATH (many scripts)
- `google-genai` (explicitly imported in Gemini promoter script)

External services/accounts:

- Amazon website access
- Gemini API key(s)
- Google OAuth client for YouTube upload

## 7. Config/secret files present

Observed files containing credentials/tokens:

- `client_secret.json` (OAuth client ID/secret + redirect URIs)
- `youtube_token.json` (access token + refresh token)
- `gemini_keys.txt` (3 lines in current repo copy)

`.gitignore` excludes key/env patterns but these concrete secret files are currently present in workspace.

## 8. Notable implementation details and quirks

1. Many scripts use hardcoded absolute Windows default paths.
2. `make_square_reel.py` help text has small mismatches vs defaults:
   - `--cta-overlay-start` help says default `7.3`, code default is `7.9`.
   - `--final-music-volume` help says default `1.0`, code default is `0.7`.
3. `youtube_uploader_api.py` header comment says `client_secrets.json`, code uses `client_secret.json`.
4. Product description/video URL regex extraction can capture noisy Amazon embedded JSON/ad payload text (seen in sample results).
5. Repo contains large generated/binary artifacts (`.mp4`, `.mp3`, `.wav`, `.vsix`, `.pyc`) rather than only source code.
6. Unicode Malayalam renders correctly in files, but terminal output may appear mojibake depending on console encoding.

## 9. What this project ultimately delivers

A near-end-to-end content automation stack that turns Amazon best-seller products into short-form promo reels (Malayalam narration + product footage + CTA + QR + optional music) and can publish the final result to YouTube with minimal manual intervention.
