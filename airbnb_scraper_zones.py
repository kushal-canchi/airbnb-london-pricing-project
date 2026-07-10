"""
Airbnb Listings Scraper (Selenium + BeautifulSoup)
---------------------------------------------------
Scrapes publicly visible listing data from Airbnb search results pages
(title, price, rating, review count, room type, location snippet, listing URL)
and saves everything to an Excel file.

WHY SELENIUM + BEAUTIFULSOUP:
Airbnb's search results are rendered with JavaScript, so a plain
requests.get() call won't show you the listing cards. Selenium drives a
real (headless) Chrome browser to load and scroll the page so the JS
renders, then BeautifulSoup parses the resulting HTML.

BEFORE YOU RUN THIS:
1. Install Google Chrome on your machine (if not already installed).
2. Install the Python packages:
   pip install selenium beautifulsoup4 pandas openpyxl webdriver-manager

3. Edit the CONFIG section below (search location, number of pages, etc.)

ETHICAL / LEGAL NOTE FOR YOUR PROJECT WRITE-UP:
- This script only reads publicly visible search-result HTML — no login,
  no bypassing paywalls or CAPTCHAs, no private data.
- It rate-limits requests (random delays) to avoid hammering Airbnb's
  servers, which is both good etiquette and reduces the chance of
  getting blocked.
- Airbnb's Terms of Service restrict automated scraping. For a university
  project this is normally fine under fair-use/academic-research
  justification, but you should say explicitly in your methodology
  section that: (a) you only collected publicly available fields relevant
  to your research questions, (b) you rate-limited your requests, and
  (c) no personal host/guest data was collected. This is exactly the kind
  of transparency your marker flagged as missing in the review-of-methods
  feedback.
- If Airbnb changes their page layout, the CSS selectors below WILL need
  updating — that's normal for web scraping and worth noting as a
  limitation in your report.
"""

import time
import random
import re
from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service


# ============================================================
# CONFIG — edit these to suit your project
# ============================================================
# A single London-wide search will start repeating listings after roughly
# a couple hundred results (Airbnb caps how deep one query's pagination
# goes), so to reliably reach 500-600 listings we search several distinct
# London areas and combine the results. This also gives you the
# geographic spread you'll need for the Inner/Outer London comparison.
LOCATIONS = [
    "Camden, London, United Kingdom",
    "Hackney, London, United Kingdom",
    "Westminster, London, United Kingdom",
    "Croydon, London, United Kingdom",
    "Ealing, London, United Kingdom",
    "Greenwich, London, United Kingdom",
]
PAGES_PER_LOCATION = 5                # ~18-20 listings/page -> ~100/location -> ~600 total across 6 areas

# IMPORTANT: fixed dates are required so every listing quotes the SAME
# number of nights. Without this, Airbnb assigns each card a random
# check-in/check-out window, and the total price becomes incomparable
# across rows (a 4-night £400 stay vs a 10-night £1000 stay).
CHECKIN = "2026-09-01"
CHECKOUT = "2026-09-05"               # 4 nights - change both together if you want a different length
EXPECTED_NIGHTS = (
    datetime.strptime(CHECKOUT, "%Y-%m-%d") - datetime.strptime(CHECKIN, "%Y-%m-%d")
).days
HEADLESS = True                       # False = watch the browser work (good for debugging)
OUTPUT_FILE = f"airbnb_london_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
MIN_DELAY, MAX_DELAY = 3, 7           # seconds between page loads (be polite)

# Optional: borough -> zone mapping for when you scale up to multiple
# London areas. Using a small number of zones (e.g. Inner/Outer) instead
# of one dummy per borough avoids the overparameterization issue flagged
# on the original proposal. Fill this in as you add search locations.
BOROUGH_ZONE_MAP = {
    "Camden": "Inner London",
    "Islington": "Inner London",
    "Hackney": "Inner London",
    "Westminster": "Inner London",
    "Tower Hamlets": "Inner London",
    "Southwark": "Inner London",
    "Lambeth": "Inner London",
    "Wandsworth": "Inner London",
    "Hammersmith and Fulham": "Inner London",
    "Kensington and Chelsea": "Inner London",
    "Croydon": "Outer London",
    "Bromley": "Outer London",
    "Barnet": "Outer London",
    "Ealing": "Outer London",
    "Enfield": "Outer London",
    "Havering": "Outer London",
    "Hillingdon": "Outer London",
    "Redbridge": "Outer London",
    "Sutton": "Outer London",
    "Bexley": "Outer London",
    # add remaining boroughs as you expand
}


def build_search_url(location, page_offset=0, checkin=None, checkout=None):
    """Builds an Airbnb search URL. page_offset is items_offset (20 per page)."""
    base = "https://www.airbnb.co.uk/s/{}/homes".format(location.replace(", ", "--").replace(" ", "-"))
    params = [f"items_offset={page_offset}"]
    if checkin:
        params.append(f"checkin={checkin}")
    if checkout:
        params.append(f"checkout={checkout}")
    return base + "?" + "&".join(params)


def start_driver(headless=True):
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--window-size=1400,1000")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver


def scroll_page(driver, pause=1.5, scrolls=6):
    """Airbnb lazy-loads cards as you scroll — scroll down in steps to force them in."""
    last_height = driver.execute_script("return document.body.scrollHeight")
    for _ in range(scrolls):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height


def parse_total_price(raw_text):
    """Extracts the numeric TOTAL stay price from Airbnb's price strings."""
    if not raw_text:
        return None
    match = re.search(r"[\d,]+", raw_text.replace("£", ""))
    if match:
        return float(match.group().replace(",", ""))
    return None


def parse_nights(full_text):
    """
    Looks for a 'for X night(s)' phrase in the card text (Airbnb includes
    this alongside the total price, e.g. '£420 for 4 nights'). Returns None
    if not found, in which case we fall back to EXPECTED_NIGHTS from config.
    """
    m = re.search(r"for\s+(\d+)\s+nights?", full_text, re.IGNORECASE)
    return int(m.group(1)) if m else None


def parse_listing_card(card):
    """
    Pulls the fields we care about out of one listing card's HTML.

    KEY FIX vs the previous version: we build ONE concatenated text blob
    (`full_text`) from the entire card first, and run our regexes against
    that blob rather than against a single BeautifulSoup text node. Airbnb
    frequently splits the rating and the review count into two separate
    text nodes in the DOM (e.g. "4.85" in one span, "(32 reviews)" in a
    sibling span) - searching only one node was why review_count came back
    empty even though rating was being found correctly.
    """
    data = {}
    full_text = card.get_text(" ", strip=True)

    # Title / name
    title_tag = card.select_one('[data-testid="listing-card-title"]')
    data["title"] = title_tag.get_text(strip=True) if title_tag else None

    # Price - grab the raw text, then split into total price and nights
    price_tag = card.select_one('span[class*="price"], div[data-testid*="price"]')
    price_text = price_tag.get_text(" ", strip=True) if price_tag else full_text
    data["price_raw"] = price_text

    total_price = parse_total_price(price_text)
    nights = parse_nights(full_text) or parse_nights(price_text)
    data["nights"] = nights if nights else EXPECTED_NIGHTS
    data["nights_source"] = "parsed_from_card" if nights else "assumed_from_search_dates"
    data["price_gbp_total"] = total_price
    data["price_gbp_per_night"] = (
        round(total_price / data["nights"], 2) if total_price and data["nights"] else None
    )

    # Rating + review count - searched against the FULL concatenated text,
    # not a single node, so this catches cases where they're in separate
    # sibling elements (see docstring above).
    m = re.search(r"(\d\.\d{1,2})\s*(?:out of 5)?[^\d(]*\(?(\d+)\)?\s*(?:reviews?)?", full_text)
    if m:
        data["rating"] = float(m.group(1))
        data["review_count"] = int(m.group(2)) if m.group(2) else None
    else:
        data["rating"] = None
        data["review_count"] = None

    # Listing URL
    link_tag = card.select_one("a[href*='/rooms/']")
    data["url"] = ("https://www.airbnb.co.uk" + link_tag["href"]) if link_tag else None

    return data


def zone_for_location(location):
    """Maps a search location string to Inner/Outer London using BOROUGH_ZONE_MAP."""
    for borough, zone in BOROUGH_ZONE_MAP.items():
        if borough.lower() in location.lower():
            return zone
    return None


def scrape_one_location(driver, location, max_pages, checkin=None, checkout=None):
    listings = []
    for page in range(max_pages):
        offset = page * 20
        url = build_search_url(location, offset, checkin, checkout)
        print(f"  [Page {page + 1}/{max_pages}] Loading: {url}")

        driver.get(url)
        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="listing-card-title"]'))
            )
        except Exception:
            print("    No listing cards found on this page — stopping this location "
                  "(may be end of results, or Airbnb blocked/changed layout).")
            break

        scroll_page(driver)
        soup = BeautifulSoup(driver.page_source, "html.parser")

        cards = soup.select('[itemprop="itemListElement"], div[data-testid="card-container"]')
        if not cards:
            cards = soup.select('div:has([data-testid="listing-card-title"])')

        print(f"    Found {len(cards)} listing cards")

        for card in cards:
            listing = parse_listing_card(card)
            listing["search_location"] = location
            listing["zone"] = zone_for_location(location)
            listing["page_scraped"] = page + 1
            listing["scraped_at"] = datetime.now().isoformat(timespec="seconds")
            listings.append(listing)

        delay = random.uniform(MIN_DELAY, MAX_DELAY)
        print(f"    Waiting {delay:.1f}s before next page...")
        time.sleep(delay)

    return listings


def scrape_airbnb(locations, pages_per_location, checkin=None, checkout=None, headless=True):
    driver = start_driver(headless=headless)
    all_listings = []

    try:
        for location in locations:
            print(f"\n=== Scraping: {location} ===")
            listings = scrape_one_location(driver, location, pages_per_location, checkin, checkout)
            print(f"  -> {len(listings)} listings from {location}")
            all_listings.extend(listings)
    finally:
        driver.quit()

    return all_listings


def save_to_excel(listings, output_file):
    df = pd.DataFrame(listings)

    # De-duplicate by listing URL (Airbnb sometimes repeats cards across pages)
    if "url" in df.columns:
        df = df.drop_duplicates(subset="url")

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="listings", index=False)

        # Auto-fit column widths roughly
        worksheet = writer.sheets["listings"]
        for i, col in enumerate(df.columns, start=1):
            max_len = max(df[col].astype(str).map(len).max() if not df.empty else 0, len(col)) + 2
            worksheet.column_dimensions[worksheet.cell(row=1, column=i).column_letter].width = min(max_len, 50)

    print(f"\nSaved {len(df)} unique listings to {output_file}")


if __name__ == "__main__":
    print(f"Starting Airbnb scrape across {len(LOCATIONS)} London areas: {LOCATIONS}")
    listings = scrape_airbnb(LOCATIONS, PAGES_PER_LOCATION, CHECKIN, CHECKOUT, headless=HEADLESS)

    if listings:
        save_to_excel(listings, OUTPUT_FILE)
    else:
        print("No listings scraped. Airbnb may have changed their page structure, "
              "shown a CAPTCHA, or blocked the request. Try HEADLESS=False to watch "
              "what the browser sees.")
