#!/usr/bin/env python3
"""
Generate a QR code image from a product URL.

Example:
  python product_url_to_qr.py --url "https://www.amazon.in/dp/B07WMS7TWB" --output "amazon_product_qr.png"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import qrcode
from qrcode.constants import ERROR_CORRECT_M


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        if (parent / "requirements.txt").exists():
            return parent
    return here.parent


PROJECT_ROOT = _project_root()

DEFAULT_URL = "https://www.amazon.in/dp/B07WMS7TWB"
DEFAULT_OUTPUT = str(PROJECT_ROOT / "assets" / "images" / "amazon_product_qr.png")
DEFAULT_AFFILIATE_TAG = "shozi-21"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a product URL into a QR code image.")
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Product URL to encode (default: {DEFAULT_URL}).",
    )
    parser.add_argument(
        "--affiliate-tag",
        default=DEFAULT_AFFILIATE_TAG,
        help=f"Affiliate tag to inject into URL (default: {DEFAULT_AFFILIATE_TAG}).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output PNG path (default: {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args()


def to_affiliate_url(url: str, affiliate_tag: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("URL must start with http:// or https://")

    tag = str(affiliate_tag or "").strip()
    if not tag:
        raise ValueError("Affiliate tag cannot be empty.")

    query_items = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if k.lower() != "tag"]
    query_items.append(("tag", tag))
    new_query = urlencode(query_items, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


def build_qr_image(url: str):
    if not url.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")

    qr = qrcode.QRCode(
        version=None,
        error_correction=ERROR_CORRECT_M,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white")


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    affiliate_url = to_affiliate_url(args.url, args.affiliate_tag)
    qr_image = build_qr_image(affiliate_url)
    qr_image.save(output_path)

    print(f"Original URL: {args.url}")
    print(f"Affiliate URL: {affiliate_url}")
    print(f"Affiliate tag: {args.affiliate_tag}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
