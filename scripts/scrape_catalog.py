#!/usr/bin/env python3
"""
Scrape the SHL Individual Test Solutions catalog.
Uses Playwright (headless Chromium) because the catalog is JavaScript-rendered.

Usage:
    uv run python scripts/scrape_catalog.py

Output:
    app/data/catalog_raw.json   — raw scraped data
    app/data/catalog.json       — cleaned, deduplicated, validated
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()

BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/?type=1"
DATA_DIR = Path(__file__).resolve().parent.parent / "app" / "data"
RAW_PATH = DATA_DIR / "catalog_raw.json"
CLEAN_PATH = DATA_DIR / "catalog.json"


async def scrape_catalog() -> list[dict]:
    """
    Main scraping function. Returns list of raw item dicts.
    Paginates through the SHL catalog (type=1 = Individual Test Solutions).
    """
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout

    raw_items: list[dict] = []
    seen_urls: set[str] = set()
    page_num = 0
    pages_visited = 0
    items_dropped = 0

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )
        page = await context.new_page()

        while True:
            url = f"{CATALOG_URL}&start={page_num * 12}"
            logger.info("Fetching page %d: %s", page_num + 1, url)

            try:
                await page.goto(url, wait_until="networkidle", timeout=30000)
                await asyncio.sleep(settings.scrape_delay_seconds)
                pages_visited += 1
            except PlaywrightTimeout:
                logger.warning("Timeout on page %d, retrying once", page_num + 1)
                await asyncio.sleep(3)
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    await asyncio.sleep(2)
                except Exception as exc:
                    logger.error("Failed to load page %d: %s", page_num + 1, exc)
                    break

            # Wait for catalog table/grid to load
            try:
                await page.wait_for_selector(
                    ".custom-select__list, table.custom-table, [class*='product'], [class*='catalog']",
                    timeout=15000,
                )
            except PlaywrightTimeout:
                logger.warning("Catalog content not found on page %d", page_num + 1)

            # Extract page content as HTML for parsing
            html = await page.content()
            items_on_page = await _extract_items_from_page(page, html)

            if not items_on_page:
                logger.info("No items found on page %d — end of catalog", page_num + 1)
                break

            new_items = 0
            for item in items_on_page:
                if item["url"] not in seen_urls:
                    seen_urls.add(item["url"])
                    raw_items.append(item)
                    new_items += 1
                else:
                    items_dropped += 1

            logger.info("Page %d: %d new items (total: %d)", page_num + 1, new_items, len(raw_items))

            if new_items == 0:
                logger.info("No new items — stopping pagination")
                break

            # Check for "next page" button
            has_next = await _has_next_page(page)
            if not has_next:
                logger.info("No next page button found — done")
                break

            page_num += 1
            # Rate limiting
            await asyncio.sleep(settings.scrape_delay_seconds)

        await browser.close()

    logger.info(
        "Scraping complete: %d pages, %d items, %d duplicates dropped",
        pages_visited, len(raw_items), items_dropped,
    )
    return raw_items


async def _extract_items_from_page(page, html: str) -> list[dict]:
    """
    Extract assessment items from the current page.
    Tries multiple strategies for robustness across SHL page layouts.
    """
    items = []

    # Strategy 1: Parse product rows from the catalog table
    try:
        rows = await page.query_selector_all("table tbody tr, .custom-table tbody tr")
        for row in rows:
            item = await _parse_table_row(row)
            if item:
                items.append(item)
    except Exception as exc:
        logger.debug("Table strategy failed: %s", exc)

    # Strategy 2: Parse from catalog grid/card layout
    if not items:
        try:
            cards = await page.query_selector_all(
                "[class*='product-item'], [class*='catalog-item'], "
                "[class*='assessment-item'], article"
            )
            for card in cards:
                item = await _parse_card(card)
                if item:
                    items.append(item)
        except Exception as exc:
            logger.debug("Card strategy failed: %s", exc)

    # Strategy 3: Parse from any <a> links that look like product pages
    if not items:
        try:
            links = await page.query_selector_all("a[href*='/product-catalog/view/']")
            for link in links:
                href = await link.get_attribute("href")
                name = await link.inner_text()
                if href and name.strip():
                    url = href if href.startswith("http") else urljoin(BASE_URL, href)
                    items.append({
                        "name": name.strip(),
                        "url": url,
                        "test_type": [],
                        "description": "",
                        "remote_testing": None,
                        "adaptive_irt": None,
                        "duration": None,
                        "job_levels": [],
                        "languages": [],
                    })
        except Exception as exc:
            logger.debug("Link strategy failed: %s", exc)

    return items


async def _parse_table_row(row) -> dict | None:
    """Parse a catalog table row into a structured dict."""
    try:
        cells = await row.query_selector_all("td")
        if len(cells) < 2:
            return None

        # Column structure varies; attempt best-effort extraction
        name_el = await row.query_selector("a[href*='product-catalog']")
        if not name_el:
            return None

        name = (await name_el.inner_text()).strip()
        href = await name_el.get_attribute("href")
        if not href:
            return None
        url = href if href.startswith("http") else urljoin(BASE_URL, href)

        # Attempt to read test type flags from icon cells
        test_type = []
        remote_testing = None
        adaptive_irt = None
        duration = None
        job_levels = []
        languages = []

        # Try to find icons/labels in cells
        for cell in cells:
            cell_text = (await cell.inner_text()).strip()
            cell_html = await cell.inner_html()

            # Test type badges
            for code, pattern in [
                ("A", r"\bA\b|ability|aptitude"),
                ("B", r"\bB\b|biodata|situational"),
                ("C", r"\bC\b|competen"),
                ("K", r"\bK\b|knowledge|skills"),
                ("P", r"\bP\b|personal"),
            ]:
                if re.search(pattern, cell_html, re.IGNORECASE) and code not in test_type:
                    # Only add if there's a short code-like badge
                    if re.search(rf'<span[^>]*>\s*{code}\s*</span>|class="[^"]*type[^"]*"', cell_html):
                        test_type.append(code)

            # Duration
            dur_match = re.search(r"(\d+)\s*min", cell_text, re.IGNORECASE)
            if dur_match:
                duration = f"{dur_match.group(1)} minutes"

            # Remote testing / adaptive
            if "remote" in cell_html.lower():
                remote_testing = True
            if "adaptive" in cell_html.lower() or "irt" in cell_html.lower():
                adaptive_irt = True

        return {
            "name": name,
            "url": url,
            "test_type": test_type,
            "description": "",
            "remote_testing": remote_testing,
            "adaptive_irt": adaptive_irt,
            "duration": duration,
            "job_levels": job_levels,
            "languages": languages,
        }

    except Exception as exc:
        logger.debug("Row parse error: %s", exc)
        return None


async def _parse_card(card) -> dict | None:
    """Parse a product card element."""
    try:
        link_el = await card.query_selector("a[href*='product-catalog']")
        if not link_el:
            return None
        name = (await link_el.inner_text()).strip()
        href = await link_el.get_attribute("href")
        if not href or not name:
            return None
        url = href if href.startswith("http") else urljoin(BASE_URL, href)
        return {
            "name": name,
            "url": url,
            "test_type": [],
            "description": "",
            "remote_testing": None,
            "adaptive_irt": None,
            "duration": None,
            "job_levels": [],
            "languages": [],
        }
    except Exception:
        return None


async def _has_next_page(page) -> bool:
    """Check if there's a next-page control."""
    selectors = [
        "[aria-label='Next']",
        ".pagination .next:not(.disabled)",
        "a[rel='next']",
        "button[class*='next']",
        "[class*='pagination'] a:last-child",
    ]
    for sel in selectors:
        try:
            el = await page.query_selector(sel)
            if el:
                disabled = await el.get_attribute("disabled")
                class_attr = await el.get_attribute("class") or ""
                if disabled is None and "disabled" not in class_attr:
                    return True
        except Exception:
            pass
    return False


async def scrape_detail_page(url: str, page) -> dict:
    """
    Scrape individual product detail page for richer metadata.
    Returns partial dict to merge into base item.
    """
    try:
        await page.goto(url, wait_until="networkidle", timeout=20000)
        await asyncio.sleep(0.5)
        html = await page.content()

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")

        detail = {}

        # Description
        desc_el = soup.find("div", class_=re.compile(r"description|overview|content", re.I))
        if desc_el:
            detail["description"] = desc_el.get_text(separator=" ", strip=True)[:1000]

        # Duration
        for el in soup.find_all(text=re.compile(r"\d+\s*min", re.I)):
            m = re.search(r"(\d+)\s*min", el, re.I)
            if m:
                detail["duration"] = f"{m.group(1)} minutes"
                break

        # Test type codes
        test_types = []
        for el in soup.find_all(text=re.compile(r"\b[ABCKP]\b")):
            for code in re.findall(r"\b([ABCKP])\b", el):
                if code not in test_types:
                    test_types.append(code)
        if test_types:
            detail["test_type"] = test_types

        # Remote testing
        if re.search(r"remote\s+testing", html, re.I):
            detail["remote_testing"] = True

        # Adaptive/IRT
        if re.search(r"adaptive|IRT", html, re.I):
            detail["adaptive_irt"] = True

        # Languages
        lang_section = soup.find(text=re.compile(r"language", re.I))
        if lang_section and lang_section.parent:
            lang_text = lang_section.parent.get_text()
            langs = [l.strip() for l in lang_text.split(",") if len(l.strip()) > 2 and len(l.strip()) < 50]
            if langs:
                detail["languages"] = langs[:20]

        return detail
    except Exception as exc:
        logger.debug("Detail page failed for %s: %s", url, exc)
        return {}


def clean_catalog(raw_items: list[dict]) -> list[dict]:
    """
    Clean and validate raw scraped items.
    - Normalize whitespace
    - Deduplicate by URL
    - Validate required fields
    - Log dropped items
    """
    seen_urls: set[str] = set()
    cleaned: list[dict] = []
    dropped = 0

    for item in raw_items:
        # Normalize
        item["name"] = re.sub(r"\s+", " ", item.get("name", "")).strip()
        item["url"] = item.get("url", "").strip()
        item["description"] = re.sub(r"\s+", " ", item.get("description", "")).strip()

        # Validate
        if not item["name"]:
            logger.debug("Dropped: empty name")
            dropped += 1
            continue
        if not item["url"] or not item["url"].startswith("https://www.shl.com"):
            logger.debug("Dropped: invalid URL '%s'", item.get("url"))
            dropped += 1
            continue
        if item["url"] in seen_urls:
            logger.debug("Dropped: duplicate URL '%s'", item["url"])
            dropped += 1
            continue

        seen_urls.add(item["url"])
        cleaned.append(item)

    logger.info(
        "Cleaning done: %d valid, %d dropped",
        len(cleaned), dropped,
    )
    return cleaned


async def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Starting SHL catalog scrape (Individual Test Solutions only)…")
    raw_items = await scrape_catalog()

    # Save raw
    with RAW_PATH.open("w") as f:
        json.dump(raw_items, f, indent=2)
    logger.info("Raw catalog saved: %s (%d items)", RAW_PATH, len(raw_items))

    # If we got very few items from direct scraping, try detail enrichment
    if len(raw_items) < 10:
        logger.warning(
            "Only %d items scraped — site may require additional interaction. "
            "Check catalog_raw.json and try running again.", len(raw_items)
        )

    # Clean
    cleaned = clean_catalog(raw_items)

    # Save clean
    with CLEAN_PATH.open("w") as f:
        json.dump(cleaned, f, indent=2)
    logger.info("Clean catalog saved: %s (%d items)", CLEAN_PATH, len(cleaned))

    # Stats
    print(f"\n{'='*50}")
    print(f"SCRAPE STATS")
    print(f"  Raw items:     {len(raw_items)}")
    print(f"  Clean items:   {len(cleaned)}")
    print(f"  Saved to:      {CLEAN_PATH}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    asyncio.run(main())
