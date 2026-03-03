# Docker Daily Auto-Publish Setup

## 1. What is implemented

The scheduler script `scripts/auto_publish_scheduler.py` now runs the full pipeline automatically with fixed publish slots:

- `11:30 AM`
- `5:30 PM`
- `7:30 PM`

Timezone default: `Asia/Kolkata`

Behavior:

1. Sleeps and wakes `30` minutes before next slot.
2. Prepares a publishable reel from next pending product URL.
3. Uploads at exact slot time.
4. Continues product-by-product for current branch.
5. Moves to next branch when current branch is finished.
6. Persists progress in `data/runtime/scheduler_state.json` so restarts resume correctly.
7. After a successful upload, deletes generated per-product artifacts (downloaded video, Gemini output folder, generated reel files, QR file, temporary product result JSON).

## 2. Required files before starting container

Create a local folder `secrets/` and place:

- `secrets/client_secret.json` (YouTube OAuth desktop credentials)
- `secrets/youtube_token.json` (authorized token file)
- `secrets/gemini_keys.txt` (one Gemini API key per line, or comma-separated)

Also ensure these media files exist:

- `assets/video/CTA YT.mp4`
- `assets/images/amazon_product_qr.png` (will be overwritten each product)
- optional: `assets/audio/reel Music.mp3`

## 3. First-time YouTube token note

`scripts/youtube_uploader_api.py` requires OAuth consent at least once.

Recommended flow:

1. Generate `youtube_token.json` once on a machine with browser access.
2. Copy that token file into `secrets/youtube_token.json`.
3. Deploy container on server.

After that, refresh token flow runs automatically.

## 4. Build and run

```bash
docker compose build
docker compose up -d
```

View logs:

```bash
docker compose logs -f yt-auto
```

Stop:

```bash
docker compose down
```

## 5. Runtime data location

All persistent runtime data is mounted to `./data/runtime`:

- scheduler state
- branches/products intermediate JSON files
- downloaded videos
- Gemini outputs
- rendered reels

## 6. Customize schedule or behavior

Edit `docker-compose.yml` command args, for example:

- `--schedule-slots 11:30,17:30,19:30`
- `--wake-before-minutes 30`
- `--department-name "Home & Kitchen"`
- `--youtube-privacy public`
- `--affiliate-tag yourtag-21`
- `--cta-text "your CTA"`
- `--no-final-music`

Then restart:

```bash
docker compose up -d --build
```

## 7. Direct host run (without Docker)

```bash
python scripts/auto_publish_scheduler.py \
  --timezone Asia/Kolkata \
  --schedule-slots 11:30,17:30,19:30 \
  --wake-before-minutes 30
```
