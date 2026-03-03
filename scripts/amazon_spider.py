from __future__ import annotations

import argparse
from collections import defaultdict, deque
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any
from urllib.parse import urlparse, urlunparse

import scrapy
from scrapy import signals
from scrapy.crawler import CrawlerProcess
from scrapy.exceptions import DropItem
from twisted.internet import task

try:
    import psutil
except ImportError:  # pragma: no cover - handled at runtime
    psutil = None


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        if (parent / "requirements.txt").exists():
            return parent
    return here.parent


PROJECT_ROOT = _project_root()
DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_OUTPUT_FILE = str(DATA_DIR / "branches.json")
DEFAULT_MAX_PAGES = 0
DEFAULT_SPEED = "ultra"
DEFAULT_SCALE_INTERVAL = 2.0
ABSOLUTE_MAX_CONCURRENCY = 512
DEFAULT_DEPARTMENT_NAME = "Home & Kitchen"

DEPARTMENT_NAME_TO_SLUG = {
    "amazon launchpad": "boost",
    "amazon renewed": "amazon-renewed",
    "apps & games": "mobile-apps",
    "apps and games": "mobile-apps",
    "baby products": "baby",
    "bags, wallets and luggage": "luggage",
    "beauty": "beauty",
    "books": "books",
    "car & motorbike": "automotive",
    "car and motorbike": "automotive",
    "clothing & accessories": "apparel",
    "clothing and accessories": "apparel",
    "computers & accessories": "computers",
    "computers and accessories": "computers",
    "electronics": "electronics",
    "garden & outdoors": "garden",
    "garden and outdoors": "garden",
    "gift cards": "gift-cards",
    "grocery & gourmet foods": "grocery",
    "grocery and gourmet foods": "grocery",
    "health & personal care": "hpc",
    "health and personal care": "hpc",
    "home & kitchen": "kitchen",
    "home and kitchen": "kitchen",
    "home improvement": "home-improvement",
    "industrial & scientific": "industrial",
    "industrial and scientific": "industrial",
    "jewellery": "jewelry",
    "jewelry": "jewelry",
    "kindle store": "digital-text",
    "movies & tv shows": "dvd",
    "movies and tv shows": "dvd",
    "music": "music",
    "musical instruments": "musical-instruments",
    "office products": "office",
    "pet supplies": "pet-supplies",
    "shoes & handbags": "shoes",
    "shoes and handbags": "shoes",
    "software": "software",
    "sports, fitness & outdoors": "sports",
    "sports, fitness and outdoors": "sports",
    "toys & games": "toys",
    "toys and games": "toys",
    "video games": "videogames",
    "watches": "watches",
}


def _normalize_department_name(name: str) -> str:
    return " ".join(name.strip().lower().split())


def _slugify_guess(name: str) -> str:
    cleaned = _normalize_department_name(name)
    cleaned = cleaned.replace("&", "and")
    slug = re.sub(r"[^a-z0-9]+", "-", cleaned).strip("-")
    return slug or "kitchen"


def _resolve_department_slug(department_name: str) -> tuple[str, bool]:
    normalized = _normalize_department_name(department_name)
    slug = DEPARTMENT_NAME_TO_SLUG.get(normalized)
    if slug:
        return slug, True

    # If user already passed a slug-like value, accept it directly.
    if re.fullmatch(r"[a-z0-9-]+", normalized):
        return normalized, False

    return _slugify_guess(department_name), False


def _build_department_start_url(department_name: str) -> tuple[str, str, bool]:
    slug, known_department = _resolve_department_slug(department_name)
    start_url = f"https://www.amazon.in/gp/bestsellers/{slug}/ref=zg_bs_nav_{slug}_0"
    return start_url, slug, known_department


def _canonical_bestseller_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path
    if "/ref=" in path:
        path = path.split("/ref=", 1)[0]
    if path != "/":
        path = path.rstrip("/")
    if not path:
        path = "/"
    normalized = parsed._replace(
        scheme="https",
        netloc=parsed.netloc.lower(),
        path=path,
        params="",
        query="",
        fragment="",
    )
    return urlunparse(normalized)


DEFAULT_START_URL, DEFAULT_DEPARTMENT_SLUG, _ = _build_department_start_url(
    DEFAULT_DEPARTMENT_NAME
)

SPEED_PROFILES: dict[str, dict] = {
    "safe": {
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 0.2,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 8.0,
        "DOWNLOAD_DELAY": 0.0,
        "RETRY_TIMES": 4,
    },
    "fast": {
        "AUTOTHROTTLE_ENABLED": False,
        "DOWNLOAD_DELAY": 0.0,
        "RETRY_TIMES": 2,
    },
    "ultra": {
        "AUTOTHROTTLE_ENABLED": False,
        "DOWNLOAD_DELAY": 0.0,
        "RETRY_TIMES": 1,
    },
}


def _query_gpu_snapshot() -> dict[str, float] | None:
    """
    Return aggregate GPU utilization and memory stats from nvidia-smi when available.
    """
    try:
        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            return None

        gpu_utils: list[float] = []
        mem_totals: list[float] = []
        mem_used: list[float] = []
        for line in lines:
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 3:
                continue
            gpu_utils.append(float(parts[0]))
            mem_totals.append(float(parts[1]))
            mem_used.append(float(parts[2]))

        if not gpu_utils or not mem_totals:
            return None

        total_mem_mb = sum(mem_totals)
        used_mem_mb = sum(mem_used)
        mem_util = (used_mem_mb / total_mem_mb * 100.0) if total_mem_mb > 0 else 0.0
        return {
            "gpu_count": float(len(gpu_utils)),
            "gpu_util_percent_avg": sum(gpu_utils) / len(gpu_utils),
            "gpu_util_percent_peak": max(gpu_utils),
            "gpu_mem_util_percent": mem_util,
            "gpu_total_mem_gb": total_mem_mb / 1024.0,
            "gpu_used_mem_gb": used_mem_mb / 1024.0,
        }
    except Exception:
        return None


def _cpu_freq_ghz() -> float:
    if not psutil:
        return 0.0
    try:
        freq = psutil.cpu_freq()
        if not freq:
            return 0.0
        current = float(freq.max or freq.current or 0.0)
        return current / 1000.0
    except Exception:
        return 0.0


def _detect_capacity(speed: str, requested_concurrency: int) -> dict[str, Any]:
    logical_cpu_cores = (psutil.cpu_count(logical=True) if psutil else None) or os.cpu_count() or 1
    logical_cpu_cores = max(1, int(logical_cpu_cores))
    physical_cpu_cores = (
        (psutil.cpu_count(logical=False) if psutil else None)
        or max(1, logical_cpu_cores // 2)
    )
    physical_cpu_cores = max(1, int(physical_cpu_cores))
    cpu_freq_ghz = _cpu_freq_ghz()

    if psutil:
        vm = psutil.virtual_memory()
        ram_total_gb = vm.total / (1024.0**3)
        ram_available_gb = vm.available / (1024.0**3)
    else:
        ram_total_gb = 8.0
        ram_available_gb = 4.0

    gpu_snapshot = _query_gpu_snapshot()
    gpu_count = 0.0
    gpu_total_mem_gb = 0.0
    if gpu_snapshot:
        gpu_count = float(gpu_snapshot["gpu_count"])
        gpu_total_mem_gb = float(gpu_snapshot["gpu_total_mem_gb"])

    # Hardware-aware concurrency ceilings tuned to avoid overshooting on laptops
    # while still allowing strong servers to run very high concurrency.
    cpu_ceiling = logical_cpu_cores * 48
    ram_ceiling = int(max(64, ram_total_gb * 24))
    base_ceiling = min(cpu_ceiling, ram_ceiling)

    if logical_cpu_cores <= 2:
        base_ceiling = min(base_ceiling, 96)
    elif logical_cpu_cores <= 4:
        base_ceiling = min(base_ceiling, 160)
    elif logical_cpu_cores <= 8:
        base_ceiling = min(base_ceiling, 256)

    gpu_bonus = int(min(128, (gpu_count * 24) + (gpu_total_mem_gb * 3)))
    base_ceiling = max(32, base_ceiling + gpu_bonus)

    speed_multiplier = {"safe": 0.55, "fast": 0.80, "ultra": 1.00}[speed]
    computed_max = int(
        max(32, min(ABSOLUTE_MAX_CONCURRENCY, base_ceiling * speed_multiplier))
    )
    computed_min = max(8, min(64, int(computed_max * 0.12)))
    startup_concurrency = max(computed_min, min(96, int(computed_max * 0.35)))

    if requested_concurrency > 0:
        computed_max = requested_concurrency
        computed_min = max(1, min(32, requested_concurrency // 4))
        startup_concurrency = requested_concurrency

    per_domain = max(1, min(ABSOLUTE_MAX_CONCURRENCY, int(computed_max * 0.70)))
    capacity_index = (
        (logical_cpu_cores * 1.6)
        + (physical_cpu_cores * 1.2)
        + (ram_total_gb * 0.9)
        + (gpu_total_mem_gb * 0.8)
        + (gpu_count * 6.0)
        + (cpu_freq_ghz * 5.0)
    )

    return {
        "cpu_cores_logical": logical_cpu_cores,
        "cpu_cores_physical": physical_cpu_cores,
        "cpu_freq_ghz": cpu_freq_ghz,
        "ram_total_gb": ram_total_gb,
        "ram_available_gb": ram_available_gb,
        "gpu_snapshot": gpu_snapshot,
        "capacity_index": capacity_index,
        "min_concurrency": computed_min,
        "startup_concurrency": startup_concurrency,
        "max_concurrency": computed_max,
        "per_domain_concurrency": per_domain,
    }


class AdaptiveResourceScalerExtension:
    """
    Dynamically scales Scrapy downloader concurrency based on real-time CPU/RAM/GPU load.
    """

    def __init__(self, crawler):
        self.crawler = crawler
        self.enabled = crawler.settings.getbool("RESOURCE_AUTOSCALE_ENABLED", True)
        self.scale_interval = max(
            0.5, crawler.settings.getfloat("RESOURCE_SCALE_INTERVAL", DEFAULT_SCALE_INTERVAL)
        )
        self.min_concurrency = max(
            1, crawler.settings.getint("RESOURCE_MIN_CONCURRENCY", 4)
        )
        self.max_concurrency = max(
            self.min_concurrency,
            crawler.settings.getint("RESOURCE_MAX_CONCURRENCY", 32),
        )
        self.per_domain_cap = max(
            1,
            crawler.settings.getint("RESOURCE_PER_DOMAIN_CAP", self.max_concurrency // 2),
        )
        self.gpu_poll_interval = max(
            1.0, crawler.settings.getfloat("RESOURCE_GPU_POLL_INTERVAL", 5.0)
        )
        self.up_step_ratio = min(
            1.0, max(0.1, crawler.settings.getfloat("RESOURCE_SCALE_UP_STEP_RATIO", 0.60))
        )
        self.down_step_ratio = min(
            1.0, max(0.1, crawler.settings.getfloat("RESOURCE_SCALE_DOWN_STEP_RATIO", 0.45))
        )
        self.cpu_pressure_weight = crawler.settings.getfloat("RESOURCE_CPU_WEIGHT", 0.50)
        self.ram_pressure_weight = crawler.settings.getfloat("RESOURCE_RAM_WEIGHT", 0.35)
        self.gpu_pressure_weight = crawler.settings.getfloat("RESOURCE_GPU_WEIGHT", 0.15)

        self._loop = task.LoopingCall(self._scale_tick)
        self._cached_gpu: dict[str, float] | None = None
        self._last_gpu_poll = 0.0
        self._last_log = 0.0
        self._log_interval = 6.0
        self._smoothed_pressure = 0.0
        self._last_tick_time = time.monotonic()
        self._last_response_count = 0
        self._last_error_count = 0
        self._last_rps = 0.0

    @classmethod
    def from_crawler(cls, crawler):
        ext = cls(crawler)
        crawler.signals.connect(ext.spider_opened, signal=signals.spider_opened)
        crawler.signals.connect(ext.spider_closed, signal=signals.spider_closed)
        return ext

    def spider_opened(self, spider):
        if psutil:
            # Prime the first cpu_percent reading.
            psutil.cpu_percent(interval=None)

        if not self.enabled:
            spider.logger.info("Resource autoscaler disabled.")
            return

        spider.logger.info(
            "Resource autoscaler enabled (interval=%.1fs, min=%d, max=%d).",
            self.scale_interval,
            self.min_concurrency,
            self.max_concurrency,
        )
        if not self._loop.running:
            self._loop.start(self.scale_interval, now=False)

    def spider_closed(self, spider):
        if self._loop.running:
            self._loop.stop()

    def _collect_metrics(self) -> dict[str, float]:
        cpu_percent = psutil.cpu_percent(interval=None) if psutil else 0.0

        if psutil:
            vm = psutil.virtual_memory()
            ram_used_percent = vm.percent
            ram_available_ratio = vm.available / vm.total if vm.total else 0.0
            ram_available_gb = vm.available / (1024.0**3)
            ram_total_gb = vm.total / (1024.0**3)
        else:
            ram_used_percent = 0.0
            ram_available_ratio = 1.0
            ram_available_gb = 0.0
            ram_total_gb = 0.0

        now = time.monotonic()
        if now - self._last_gpu_poll >= self.gpu_poll_interval:
            self._cached_gpu = _query_gpu_snapshot()
            self._last_gpu_poll = now

        gpu_pressure = 0.0
        gpu_util_percent = 0.0
        gpu_mem_util_percent = 0.0
        gpu_total_mem_gb = 0.0
        gpu_count = 0.0
        if self._cached_gpu:
            gpu_count = self._cached_gpu["gpu_count"]
            gpu_total_mem_gb = self._cached_gpu["gpu_total_mem_gb"]
            gpu_util_percent = self._cached_gpu["gpu_util_percent_avg"]
            gpu_mem_util_percent = self._cached_gpu["gpu_mem_util_percent"]
            gpu_pressure = max(
                gpu_util_percent / 100.0,
                gpu_mem_util_percent / 100.0,
            )

        cpu_pressure = cpu_percent / 100.0
        ram_pressure = ram_used_percent / 100.0
        weighted_pressure = (
            (cpu_pressure * self.cpu_pressure_weight)
            + (ram_pressure * self.ram_pressure_weight)
            + (gpu_pressure * self.gpu_pressure_weight)
        )
        self._smoothed_pressure = (self._smoothed_pressure * 0.65) + (weighted_pressure * 0.35)

        stats = self.crawler.stats
        total_responses = int(stats.get_value("downloader/response_count", 0) or 0)
        total_errors = int(
            (stats.get_value("retry/count", 0) or 0)
            + (stats.get_value("downloader/exception_count", 0) or 0)
        )
        now = time.monotonic()
        window_seconds = max(0.2, now - self._last_tick_time)
        delta_responses = max(0, total_responses - self._last_response_count)
        delta_errors = max(0, total_errors - self._last_error_count)
        responses_per_sec = delta_responses / window_seconds
        error_rate = delta_errors / max(1, delta_responses + delta_errors)

        self._last_tick_time = now
        self._last_response_count = total_responses
        self._last_error_count = total_errors

        return {
            "cpu_percent": cpu_percent,
            "ram_used_percent": ram_used_percent,
            "ram_available_ratio": ram_available_ratio,
            "ram_available_gb": ram_available_gb,
            "ram_total_gb": ram_total_gb,
            "gpu_count": gpu_count,
            "gpu_total_mem_gb": gpu_total_mem_gb,
            "gpu_util_percent": gpu_util_percent,
            "gpu_mem_util_percent": gpu_mem_util_percent,
            "gpu_pressure": gpu_pressure,
            "weighted_pressure": weighted_pressure,
            "smoothed_pressure": self._smoothed_pressure,
            "responses_per_sec": responses_per_sec,
            "error_rate": error_rate,
        }

    def _compute_target(self, downloader, metrics: dict[str, float]) -> int:
        effective_headroom = max(0.03, 1.0 - metrics["smoothed_pressure"])
        dynamic_span = self.max_concurrency - self.min_concurrency
        # Keep aggressive growth but avoid overdriving weak machines.
        target = self.min_concurrency + int(dynamic_span * (effective_headroom**0.95))

        backlog = 0
        transferring = 0
        for slot in downloader.slots.values():
            backlog += len(slot.queue)
            transferring += len(slot.transferring)

        if backlog > 0 and effective_headroom > 0.20 and metrics["error_rate"] < 0.03:
            burst_boost = min(self.max_concurrency // 3, backlog + (transferring // 2))
            target = min(self.max_concurrency, target + burst_boost)

        # Emergency brakes from live pressure and errors.
        if metrics["cpu_percent"] >= 97.0 or metrics["ram_used_percent"] >= 96.0:
            target = max(self.min_concurrency, int(target * 0.55))
        elif metrics["cpu_percent"] >= 93.0 or metrics["ram_used_percent"] >= 92.0:
            target = max(self.min_concurrency, int(target * 0.75))
        elif metrics["error_rate"] >= 0.08:
            target = max(self.min_concurrency, int(target * 0.60))
        elif metrics["error_rate"] >= 0.04:
            target = max(self.min_concurrency, int(target * 0.80))

        # Throughput-aware damping: if throughput falls while pressure is high,
        # back off to recover responsiveness.
        if (
            self._last_rps > 0
            and metrics["responses_per_sec"] < (self._last_rps * 0.75)
            and metrics["smoothed_pressure"] > 0.60
        ):
            target = max(self.min_concurrency, int(target * 0.85))
        self._last_rps = metrics["responses_per_sec"]

        return max(self.min_concurrency, min(self.max_concurrency, target))

    def _apply_concurrency(self, target: int, metrics: dict[str, float]) -> None:
        engine = getattr(self.crawler, "engine", None)
        if engine is None or engine.downloader is None:
            return

        downloader = engine.downloader
        current = int(getattr(downloader, "total_concurrency", target))
        if current <= 0:
            current = target

        if target > current:
            grow_ratio = self.up_step_ratio
            if (
                metrics["cpu_percent"] < 70.0
                and metrics["ram_used_percent"] < 75.0
                and metrics["error_rate"] < 0.02
            ):
                grow_ratio = min(0.95, grow_ratio + 0.20)
            step = max(2, int(current * grow_ratio))
            next_total = min(target, current + step)
        elif target < current:
            shrink_ratio = self.down_step_ratio
            if metrics["error_rate"] >= 0.06 or metrics["cpu_percent"] >= 92.0:
                shrink_ratio = min(0.90, shrink_ratio + 0.25)
            step = max(2, int(current * shrink_ratio))
            next_total = max(target, current - step)
        else:
            return

        next_total = max(self.min_concurrency, min(self.max_concurrency, next_total))
        if next_total == current:
            return

        downloader.total_concurrency = next_total
        per_domain = max(1, min(self.per_domain_cap, int(next_total * 0.80)))
        downloader.domain_concurrency = per_domain
        for slot in downloader.slots.values():
            slot.concurrency = per_domain

        now = time.monotonic()
        if now - self._last_log >= self._log_interval:
            self._last_log = now
            spider = getattr(self.crawler, "spider", None)
            if spider:
                spider.logger.info(
                    (
                        "Autoscale update: total=%d, per-domain=%d, cpu=%.1f%%, "
                        "ram=%.1f%% (%.2f/%.2f GB free), gpu=%.1f%%/%.1f%%, "
                        "gpu-count=%.0f, gpu-mem=%.1fGB, pressure=%.2f, "
                        "rps=%.2f, err=%.2f%%"
                    ),
                    next_total,
                    per_domain,
                    metrics["cpu_percent"],
                    metrics["ram_used_percent"],
                    metrics["ram_available_gb"],
                    metrics["ram_total_gb"],
                    metrics["gpu_util_percent"],
                    metrics["gpu_mem_util_percent"],
                    metrics["gpu_count"],
                    metrics["gpu_total_mem_gb"],
                    metrics["smoothed_pressure"],
                    metrics["responses_per_sec"],
                    metrics["error_rate"] * 100.0,
                )

    def _scale_tick(self) -> None:
        if not self.enabled:
            return
        engine = getattr(self.crawler, "engine", None)
        if engine is None or engine.downloader is None:
            return

        metrics = self._collect_metrics()
        target = self._compute_target(engine.downloader, metrics)
        self._apply_concurrency(target, metrics)


class BranchDedupePipeline:
    def __init__(self):
        self._seen_urls: set[str] = set()

    def process_item(self, item, spider):
        branch_url = str(item.get("branch_url", "")).strip()
        if not branch_url:
            raise DropItem("Missing branch_url")
        canonical = _canonical_bestseller_url(branch_url)
        if canonical in self._seen_urls:
            raise DropItem(f"Duplicate branch_url: {branch_url}")
        self._seen_urls.add(canonical)
        return item


class AmazonBestSellerBranchesSpider(scrapy.Spider):
    name = "amazon_bestseller_branches"
    allowed_domains = ["amazon.in", "www.amazon.in"]
    start_urls = [
        DEFAULT_START_URL,
    ]

    custom_settings = {
        "USER_AGENT": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "DEFAULT_REQUEST_HEADERS": {
            "Accept-Language": "en-IN,en;q=0.9",
            "Upgrade-Insecure-Requests": "1",
        },
    }

    def __init__(
        self,
        start_url: str | None = None,
        department_name: str = DEFAULT_DEPARTMENT_NAME,
        max_pages: int | str = 0,
        follow_other_departments: str = "false",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if start_url:
            self.start_urls = [start_url]

        self.max_pages = int(max_pages)
        self.follow_other_departments = str(follow_other_departments).lower() == "true"
        self.root_department = self._extract_department(self.start_urls[0])

        self._seen_branch_urls: set[str] = set()
        self._scheduled_urls: set[str] = set()
        self._pages_crawled = 0
        self._max_pages_logged = False
        self._seed_emitted = False

        self.seed_department_name = department_name.strip() if department_name else ""
        self.seed_department_slug = self._extract_department(self.start_urls[0]) or ""
        if self.seed_department_slug:
            self.seed_branch_url = (
                f"https://www.amazon.in/gp/bestsellers/{self.seed_department_slug}/"
                f"ref=zg_bs_nav_{self.seed_department_slug}_0"
            )
        else:
            self.seed_branch_url = self.start_urls[0]

    async def start(self):
        for url in self.start_urls:
            normalized = self._normalize_url(url)
            self._scheduled_urls.add(normalized)
            yield scrapy.Request(normalized, callback=self.parse)

    def parse(self, response: scrapy.http.Response):
        if self.max_pages > 0 and self._pages_crawled >= self.max_pages:
            if not self._max_pages_logged:
                self.logger.info("Reached max_pages=%s. Stopping crawl.", self.max_pages)
                self._max_pages_logged = True
            return

        self._pages_crawled += 1
        source_page = self._normalize_url(response.url)

        if not self._seed_emitted:
            self._seed_emitted = True
            seed_normalized = self._normalize_url(self.seed_branch_url)
            self._seen_branch_urls.add(seed_normalized)
            yield {
                "branch_name": self._clean_text(self.seed_department_name)
                or self.seed_department_slug
                or "Start Department",
                "branch_url": self.seed_branch_url,
                "source_page": source_page,
                "depth": 0,
                "is_up_link": False,
                "cluster_key": self._cluster_key(seed_normalized),
                "is_seed": True,
            }

        nav_root = response.css(
            "#zg-left-col div[cel_widget_id^='p13n-zg-nav-tree-all_'], "
            "div[cel_widget_id^='p13n-zg-nav-tree-all_']"
        )
        if not nav_root:
            nav_root = response

        link_nodes = nav_root.css("a[href*='/gp/bestsellers']")
        if not link_nodes:
            self.logger.warning("No branch links found on %s", response.url)

        clustered_follow_queue: dict[str, deque[tuple[str, int]]] = defaultdict(deque)

        for link in link_nodes:
            raw_href = link.attrib.get("href")
            if not raw_href:
                continue

            branch_url = self._normalize_url(response.urljoin(raw_href))
            if self._is_any_department_root(branch_url):
                continue
            if not self._is_internal_branch_url(branch_url):
                continue

            branch_name = self._clean_text(link.xpath("normalize-space(.)").get())
            depth = len(
                link.xpath(
                    "ancestor::ul[contains(@class, 'zg-browse')]"
                )
            )
            is_up_link = bool(link.xpath("ancestor::li[contains(@class, 'zg-browse-up')]"))

            if branch_url not in self._seen_branch_urls:
                self._seen_branch_urls.add(branch_url)
                yield {
                    "branch_name": branch_name,
                    "branch_url": branch_url,
                    "source_page": source_page,
                    "depth": max(depth - 1, 0),
                    "is_up_link": is_up_link,
                    "cluster_key": self._cluster_key(branch_url),
                }

            can_follow_more = self.max_pages <= 0 or self._pages_crawled < self.max_pages
            if can_follow_more and branch_url not in self._scheduled_urls:
                self._scheduled_urls.add(branch_url)
                cluster_key = self._cluster_key(branch_url)
                clustered_follow_queue[cluster_key].append((branch_url, max(depth - 1, 0)))

        # Round-robin clustered scheduling keeps deep trees balanced and uses
        # concurrency better than draining one branch at a time.
        while clustered_follow_queue:
            for cluster_key in list(clustered_follow_queue.keys()):
                queue = clustered_follow_queue[cluster_key]
                if not queue:
                    clustered_follow_queue.pop(cluster_key, None)
                    continue
                next_url, next_depth = queue.popleft()
                yield response.follow(
                    next_url,
                    callback=self.parse,
                    priority=max(1000 - (next_depth * 10), 0),
                )
                if not queue:
                    clustered_follow_queue.pop(cluster_key, None)

    @staticmethod
    def _clean_text(value: str | None) -> str:
        return " ".join((value or "").split())

    @staticmethod
    def _normalize_url(url: str) -> str:
        return _canonical_bestseller_url(url)

    @staticmethod
    def _extract_department(url: str) -> str | None:
        parts = [part for part in urlparse(url).path.split("/") if part]
        try:
            idx = parts.index("bestsellers")
        except ValueError:
            return None
        return parts[idx + 1] if idx + 1 < len(parts) else None

    @staticmethod
    def _cluster_key(url: str) -> str:
        parts = [part for part in urlparse(url).path.split("/") if part]
        if len(parts) >= 4 and parts[0] == "gp" and parts[1] == "bestsellers":
            if parts[3].isdigit():
                return parts[3]
            return parts[2]
        if len(parts) >= 3 and parts[0] == "gp" and parts[1] == "bestsellers":
            return parts[2]
        return "root"

    def _is_internal_branch_url(self, url: str) -> bool:
        parsed = urlparse(url)
        if not parsed.netloc.endswith("amazon.in"):
            return False
        if "/gp/bestsellers" not in parsed.path:
            return False

        if self.follow_other_departments or not self.root_department:
            return True

        # Keep links inside the starting department, but include root "Any Department"
        # navigation links because they appear in the branch tree.
        return (
            parsed.path.startswith("/gp/bestsellers")
            and (
                f"/gp/bestsellers/{self.root_department}" in parsed.path
            )
        )

    @staticmethod
    def _is_any_department_root(url: str) -> bool:
        parsed = urlparse(url)
        return parsed.path == "/gp/bestsellers"


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape Amazon best-seller branch links. "
            "This runs directly with: python amazon_spider.py"
        )
    )
    parser.add_argument(
        "--department-name",
        default=DEFAULT_DEPARTMENT_NAME,
        help="Department name (or slug). Start URL is auto-generated from this value.",
    )
    parser.add_argument(
        "--start-url",
        default="",
        help="Optional manual seed URL. If set, it overrides --department-name.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILE,
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=DEFAULT_MAX_PAGES,
        help="Maximum pages to crawl. Use 0 for no limit (default).",
    )
    parser.add_argument(
        "--follow-other-departments",
        action="store_true",
        help="Also follow branches outside the starting department.",
    )
    parser.add_argument(
        "--speed",
        choices=["safe", "fast", "ultra"],
        default=DEFAULT_SPEED,
        help="Crawl speed profile (default: ultra).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=0,
        help="Override concurrent requests globally (0 keeps profile default).",
    )
    parser.add_argument(
        "--autoscale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable real-time CPU/RAM/GPU based scaling (enabled by default).",
    )
    parser.add_argument(
        "--scale-interval",
        type=float,
        default=DEFAULT_SCALE_INTERVAL,
        help="Seconds between resource checks for autoscaling.",
    )
    return parser.parse_args()


def build_runtime_settings(
    args: argparse.Namespace, output_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    capacity = _detect_capacity(args.speed, args.concurrency)

    settings: dict = {
        "FEEDS": {
            str(output_path): {
                "format": "json",
                "encoding": "utf8",
                "overwrite": True,
            }
        },
        "ITEM_PIPELINES": {
            f"{__name__}.BranchDedupePipeline": 100,
        },
        "EXTENSIONS": {
            f"{__name__}.AdaptiveResourceScalerExtension": 500,
        },
        "LOG_LEVEL": "WARNING" if args.speed == "ultra" else "INFO",
        "TELNETCONSOLE_ENABLED": False,
        "COOKIES_ENABLED": False,
        "DNSCACHE_ENABLED": True,
        "DNSCACHE_SIZE": 20000,
        "REACTOR_THREADPOOL_MAXSIZE": 96 if args.speed == "ultra" else 48,
        "RANDOMIZE_DOWNLOAD_DELAY": False,
        "SCHEDULER_MEMORY_QUEUE": "scrapy.squeues.FifoMemoryQueue",
        "SCHEDULER_DISK_QUEUE": "scrapy.squeues.PickleFifoDiskQueue",
        "RETRY_HTTP_CODES": [500, 502, 503, 504, 408, 429],
        "DOWNLOAD_TIMEOUT": 20,
        "CONCURRENT_REQUESTS": capacity["startup_concurrency"],
        "CONCURRENT_REQUESTS_PER_DOMAIN": max(1, int(capacity["startup_concurrency"] * 0.70)),
        "RESOURCE_AUTOSCALE_ENABLED": bool(args.autoscale and args.concurrency <= 0),
        "RESOURCE_SCALE_INTERVAL": args.scale_interval,
        "RESOURCE_GPU_POLL_INTERVAL": max(2.0, args.scale_interval * 2.0),
        "RESOURCE_MIN_CONCURRENCY": capacity["min_concurrency"],
        "RESOURCE_MAX_CONCURRENCY": capacity["max_concurrency"],
        "RESOURCE_PER_DOMAIN_CAP": capacity["per_domain_concurrency"],
        "RESOURCE_SCALE_UP_STEP_RATIO": 0.70 if args.speed == "ultra" else 0.55,
        "RESOURCE_SCALE_DOWN_STEP_RATIO": 0.35 if args.speed == "ultra" else 0.40,
        "RESOURCE_CPU_WEIGHT": 0.50,
        "RESOURCE_RAM_WEIGHT": 0.35,
        "RESOURCE_GPU_WEIGHT": 0.15,
    }

    settings.update(SPEED_PROFILES[args.speed])

    if args.concurrency > 0:
        settings["CONCURRENT_REQUESTS"] = args.concurrency
        settings["CONCURRENT_REQUESTS_PER_DOMAIN"] = max(1, int(args.concurrency * 0.80))
        settings["RESOURCE_AUTOSCALE_ENABLED"] = False

    return settings, capacity


def dedupe_branches_output_file(output_path: Path) -> tuple[int, int]:
    if not output_path.exists():
        return 0, 0

    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception:
        return 0, 0

    if not isinstance(payload, list):
        return 0, 0

    original_count = len(payload)
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in payload:
        if not isinstance(item, dict):
            continue
        branch_url = str(item.get("branch_url", "")).strip()
        if not branch_url:
            continue

        canonical = _canonical_bestseller_url(branch_url)
        if canonical in seen:
            continue
        seen.add(canonical)

        row = dict(item)
        row["branch_url"] = canonical
        deduped.append(row)

    deduped_count = len(deduped)
    if deduped_count != original_count:
        output_path.write_text(
            json.dumps(deduped, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return original_count, deduped_count


def main():
    start_timer = time.perf_counter()
    args = parse_cli_args()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_start_url = args.start_url.strip()
    resolved_slug = ""
    known_department = True
    if not resolved_start_url:
        resolved_start_url, resolved_slug, known_department = _build_department_start_url(
            args.department_name
        )
    else:
        resolved_slug = AmazonBestSellerBranchesSpider._extract_department(
            resolved_start_url
        ) or ""

    runtime_settings, capacity = build_runtime_settings(args, output_path)
    process = CrawlerProcess(settings=runtime_settings)

    process.crawl(
        AmazonBestSellerBranchesSpider,
        start_url=resolved_start_url,
        department_name=args.department_name,
        max_pages=args.max_pages,
        follow_other_departments=str(args.follow_other_departments).lower(),
    )
    if not args.start_url:
        status = "matched" if known_department else "guessed"
        print(
            f"Department: {args.department_name} -> slug '{resolved_slug}' ({status})"
        )
    print(f"Starting crawl from: {resolved_start_url}")
    print(f"Output file: {output_path}")
    gpu_desc = "none"
    if capacity["gpu_snapshot"]:
        gpu = capacity["gpu_snapshot"]
        gpu_desc = (
            f"{int(gpu['gpu_count'])} GPU(s) "
            f"(mem={gpu['gpu_total_mem_gb']:.1f}GB, "
            f"util(avg/peak)={gpu['gpu_util_percent_avg']:.1f}%/{gpu['gpu_util_percent_peak']:.1f}%)"
        )
    print(
        "Server capacity (total):",
        f"CPU={capacity['cpu_cores_logical']} logical / {capacity['cpu_cores_physical']} physical cores @ {capacity['cpu_freq_ghz']:.2f}GHz,",
        f"RAM={capacity['ram_total_gb']:.1f}GB (free={capacity['ram_available_gb']:.1f}GB),",
        f"GPU={gpu_desc}",
    )
    print(f"Capacity index: {capacity['capacity_index']:.2f}")
    print(
        "Scaling:",
        "autoscale=on" if runtime_settings.get("RESOURCE_AUTOSCALE_ENABLED") else "autoscale=off",
        "| speed profile:",
        args.speed,
        "| min concurrency:",
        runtime_settings.get("RESOURCE_MIN_CONCURRENCY"),
        "| startup concurrency:",
        runtime_settings.get("CONCURRENT_REQUESTS"),
        "| max concurrency:",
        runtime_settings.get("RESOURCE_MAX_CONCURRENCY"),
        "| per-domain:",
        runtime_settings.get("CONCURRENT_REQUESTS_PER_DOMAIN"),
    )
    try:
        process.start()
        print("Crawl completed.")
        original_count, deduped_count = dedupe_branches_output_file(output_path)
        if original_count > 0:
            removed = max(0, original_count - deduped_count)
            print(
                f"Post-crawl dedupe: kept {deduped_count}/{original_count} "
                f"(removed duplicates: {removed})"
            )
    finally:
        elapsed = time.perf_counter() - start_timer
        print(f"Execution time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
