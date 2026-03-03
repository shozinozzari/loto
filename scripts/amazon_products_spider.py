#!/usr/bin/env python3
r"""
Scrape product URLs from the first branch in branches.json, following pagination.

Default branch source:
  <project_root>/data/branches.json

Run:
  python amazon_products_spider.py
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
from pathlib import Path
import re
import time
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.exceptions import DropItem


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        if (parent / "requirements.txt").exists():
            return parent
    return here.parent


PROJECT_ROOT = _project_root()
DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_BRANCHES_FILE = str(DATA_DIR / "branches.json")
DEFAULT_OUTPUT_FILE = str(DATA_DIR / "product.json")
DEFAULT_MAX_PAGES = 0
ASIN_RE = re.compile(r"^[A-Z0-9]{10}$", re.IGNORECASE)


def canonical_product_url(url: str) -> str:
    parsed = urlparse(url)
    match = re.search(r"/dp/([A-Z0-9]{10})", parsed.path, flags=re.IGNORECASE)
    if match:
        asin = match.group(1).upper()
        return f"https://www.amazon.in/dp/{asin}"
    match = re.search(r"/gp/product/([A-Z0-9]{10})", parsed.path, flags=re.IGNORECASE)
    if match:
        asin = match.group(1).upper()
        return f"https://www.amazon.in/dp/{asin}"
    # fallback: normalized no-query URL
    normalized = parsed._replace(scheme="https", netloc=parsed.netloc.lower(), query="", fragment="")
    return urlunparse(normalized)


def canonical_asin(asin: str) -> str:
    value = (asin or "").strip().upper()
    if ASIN_RE.fullmatch(value):
        return value
    return ""


def normalize_branch_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path
    if "/ref=" in path:
        path = path.split("/ref=", 1)[0]
    if path != "/":
        path = path.rstrip("/")
    normalized = parsed._replace(
        scheme="https",
        netloc=parsed.netloc.lower(),
        path=path,
        query="",
        params="",
        fragment="",
    )
    return urlunparse(normalized)


def extract_department_slug(url: str) -> str:
    parsed = urlparse(url)
    match = re.search(r"/gp/bestsellers/([^/?#]+)", parsed.path, flags=re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return "kitchen"


def branch_url_with_page(url: str, page: int) -> str:
    """
    Build robust Amazon bestseller pagination URL:
    /ref=zg_bs_pg_{n}_{slug}?ie=UTF8&pg={n}
    """
    parsed = urlparse(normalize_branch_url(url))
    slug = extract_department_slug(url)
    base_path = parsed.path.rstrip("/")
    if not base_path:
        base_path = "/gp/bestsellers"
    path = f"{base_path}/ref=zg_bs_pg_{page}_{slug}"
    query = urlencode({"ie": "UTF8", "pg": str(page)})
    return urlunparse(parsed._replace(path=path, query=query, params="", fragment=""))


def detect_page(url: str) -> int:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    if "pg" in query and query["pg"]:
        try:
            return int(query["pg"][0])
        except ValueError:
            pass

    # Also support /ref=zg_bs_pg_2_kitchen style.
    ref_match = re.search(r"zg_bs_pg_(\d+)_", parsed.path)
    if ref_match:
        try:
            return int(ref_match.group(1))
        except ValueError:
            pass
    return 1


def read_first_branch_url(branches_file: Path) -> str:
    raw = json.loads(branches_file.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("branches.json is empty or not a list.")

    for item in raw:
        if not isinstance(item, dict):
            continue
        url = str(item.get("branch_url", "")).strip()
        if not url:
            continue
        # Skip root bestsellers entry if present.
        if urlparse(url).path.rstrip("/") == "/gp/bestsellers":
            continue
        return url
    raise ValueError("No valid branch_url found in branches.json.")


class ProductDedupePipeline:
    def __init__(self):
        self._seen_urls: set[str] = set()

    def process_item(self, item, spider):
        url = str(item.get("product_url", "")).strip()
        if not url:
            raise DropItem("Missing product_url")
        if url in self._seen_urls:
            raise DropItem(f"Duplicate product_url: {url}")
        self._seen_urls.add(url)
        return item


class UrlOnlyWriterPipeline:
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self._urls: list[str] = []

    @classmethod
    def from_crawler(cls, crawler):
        output_path = crawler.settings.get("URLS_OUTPUT_PATH")
        if not output_path:
            raise ValueError("Missing URLS_OUTPUT_PATH setting")
        return cls(output_path)

    def open_spider(self, spider):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._urls = []

    def close_spider(self, spider):
        self.output_path.write_text(
            json.dumps(self._urls, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def process_item(self, item, spider):
        url = str(item.get("product_url", "")).strip()
        if not url:
            raise DropItem("Missing product_url")
        self._urls.append(url)
        return item


class AmazonFirstBranchProductsSpider(scrapy.Spider):
    name = "amazon_first_branch_products"
    allowed_domains = ["amazon.in", "www.amazon.in"]
    start_urls: list[str] = []

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

    def __init__(self, branch_url: str, max_pages: int | str = DEFAULT_MAX_PAGES, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.branch_url = normalize_branch_url(branch_url)
        self.max_pages = int(max_pages)
        self._scheduled_pages: set[int] = set()

        first_page = 1
        self.start_urls = [branch_url_with_page(self.branch_url, first_page)]
        self._scheduled_pages.add(first_page)

    async def start(self):
        for url in self.start_urls:
            yield scrapy.Request(url, callback=self.parse)

    def extract_asins_from_payload(self, response: scrapy.http.Response) -> set[str]:
        asins: set[str] = set()

        # 1) Primary source: explicit card data attributes.
        for raw_asin in response.css("[data-asin]::attr(data-asin)").getall():
            asin = canonical_asin(raw_asin)
            if asin:
                asins.add(asin)

        # 2) Structured payload that often contains full rank set (e.g., 51-100).
        for rec_list in response.css("[data-client-recs-list]::attr(data-client-recs-list)").getall():
            decoded = html_lib.unescape(rec_list or "")
            if not decoded:
                continue
            parsed = None
            try:
                parsed = json.loads(decoded)
            except json.JSONDecodeError:
                parsed = None

            if isinstance(parsed, list):
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    asin = canonical_asin(str(item.get("id", "")))
                    if asin:
                        asins.add(asin)
            else:
                for m in re.finditer(r'"id"\s*:\s*"([A-Z0-9]{10})"', decoded, flags=re.IGNORECASE):
                    asin = canonical_asin(m.group(1))
                    if asin:
                        asins.add(asin)

        return asins

    def parse(self, response: scrapy.http.Response):
        current_page = detect_page(response.url)

        # Product card links on Amazon bestsellers pages.
        product_links = response.css(
            "div.zg-grid-general-faceout a.a-link-normal[href], "
            "div.p13n-sc-uncoverable-faceout a.a-link-normal[href], "
            "a.a-link-normal[href*='/dp/'], a.a-link-normal[href*='/gp/product/']"
        )

        seen_on_page: set[str] = set()
        for link in product_links:
            href = link.attrib.get("href")
            if not href:
                continue

            absolute = response.urljoin(href)
            product_url = canonical_product_url(absolute)
            if "/dp/" not in product_url:
                continue
            if product_url in seen_on_page:
                continue
            seen_on_page.add(product_url)

            yield {
                "product_url": product_url,
            }

        # Additional extraction path: data-asin + data-client-recs-list payloads.
        for asin in sorted(self.extract_asins_from_payload(response)):
            product_url = f"https://www.amazon.in/dp/{asin}"
            if product_url in seen_on_page:
                continue
            seen_on_page.add(product_url)
            yield {"product_url": product_url}

        # Stop if page limit reached (0 means unlimited pages).
        if self.max_pages > 0 and current_page >= self.max_pages:
            return

        # Preferred: use actual next-page link when present.
        next_href = response.css(
            "ul.a-pagination li.a-last a::attr(href), "
            "a[aria-label='Go to next page']::attr(href), "
            "a.s-pagination-next::attr(href)"
        ).get()

        if next_href:
            next_url = response.urljoin(next_href)
            next_page = detect_page(next_url)
            if next_page not in self._scheduled_pages and (self.max_pages <= 0 or next_page <= self.max_pages):
                self._scheduled_pages.add(next_page)
                yield response.follow(next_url, callback=self.parse)
            return

        # Secondary: schedule from explicit pagination number links.
        page_links = response.css("ul.a-pagination li a::attr(href)").getall()
        scheduled_any = False
        for href in page_links:
            page_url = response.urljoin(href)
            page_num = detect_page(page_url)
            if page_num <= current_page:
                continue
            if page_num in self._scheduled_pages:
                continue
            if self.max_pages > 0 and page_num > self.max_pages:
                continue
            self._scheduled_pages.add(page_num)
            scheduled_any = True
            yield response.follow(page_url, callback=self.parse)

        if scheduled_any:
            return

        # Stop cleanly when pagination explicitly marks the last page as disabled.
        if response.css("ul.a-pagination li.a-disabled.a-last, ul.a-pagination li.a-last.a-disabled"):
            return

        # Final fallback: synthetic pagination only when no pagination UI was found.
        if seen_on_page and not response.css("ul.a-pagination"):
            next_page = current_page + 1
            if next_page not in self._scheduled_pages and (self.max_pages <= 0 or next_page <= self.max_pages):
                self._scheduled_pages.add(next_page)
                yield response.follow(branch_url_with_page(self.branch_url, next_page), callback=self.parse)


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape product URLs from first branch in branches.json with pagination."
    )
    parser.add_argument(
        "--branches-file",
        default=DEFAULT_BRANCHES_FILE,
        help="Path to branches.json",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILE,
        help="Output JSON file path (array of product URLs).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=DEFAULT_MAX_PAGES,
        help="Max pagination pages to crawl from first branch (0 = crawl all available pages).",
    )
    parser.add_argument(
        "--branch-url",
        default="",
        help="Optional direct branch URL override (skips reading branches file).",
    )
    return parser.parse_args()


def dedupe_product_output_file(output_path: Path) -> tuple[int, int]:
    if not output_path.exists():
        return 0, 0

    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception:
        return 0, 0

    if not isinstance(payload, list):
        return 0, 0

    original_count = len(payload)
    deduped: list[str] = []
    seen: set[str] = set()

    for item in payload:
        if isinstance(item, str):
            raw_url = item.strip()
        elif isinstance(item, dict):
            raw_url = str(item.get("product_url", "")).strip() or str(item.get("url", "")).strip()
        else:
            raw_url = ""

        if not raw_url:
            continue
        canonical = canonical_product_url(raw_url)
        if canonical in seen:
            continue
        seen.add(canonical)
        deduped.append(canonical)

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

    if args.branch_url.strip():
        first_branch_url = args.branch_url.strip()
    else:
        branches_path = Path(args.branches_file).expanduser().resolve()
        if not branches_path.exists():
            raise FileNotFoundError(f"branches file not found: {branches_path}")
        first_branch_url = read_first_branch_url(branches_path)

    process = CrawlerProcess(
        settings={
            "ITEM_PIPELINES": {
                f"{__name__}.ProductDedupePipeline": 100,
                f"{__name__}.UrlOnlyWriterPipeline": 200,
            },
            "URLS_OUTPUT_PATH": str(output_path),
            "COOKIES_ENABLED": False,
            "LOG_LEVEL": "INFO",
            "RETRY_TIMES": 2,
            "DOWNLOAD_TIMEOUT": 20,
        }
    )

    process.crawl(
        AmazonFirstBranchProductsSpider,
        branch_url=first_branch_url,
        max_pages=args.max_pages,
    )
    print(f"First branch URL: {first_branch_url}")
    print(f"Output file: {output_path}")
    print(f"Max pages: {args.max_pages}")
    try:
        process.start()
        print("Crawl completed.")
        original_count, deduped_count = dedupe_product_output_file(output_path)
        if deduped_count > 0:
            removed = max(0, original_count - deduped_count)
            print(
                f"Post-crawl dedupe: kept {deduped_count}/{original_count} "
                f"(removed duplicates: {removed})"
            )
        else:
            print("Saved URLs: 0")
    finally:
        elapsed = time.perf_counter() - start_timer
        print(f"Execution time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
