FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Kolkata

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    tzdata \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY . /app

RUN mkdir -p /data/runtime /data/secrets

CMD ["python", "scripts/auto_publish_scheduler.py", \
     "--project-root", "/app", \
     "--runtime-dir", "/data/runtime", \
     "--client-secrets", "/data/secrets/client_secret.json", \
     "--token-file", "/data/secrets/youtube_token.json", \
     "--api-keys-file", "/data/secrets/gemini_keys.txt", \
     "--timezone", "Asia/Kolkata", \
     "--schedule-slots", "11:30,17:30,19:30", \
     "--wake-before-minutes", "30", \
     "--youtube-privacy", "public"]
