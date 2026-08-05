# Airbnb London Pricing Project

MSc Business Analytics dissertation project — *What Drives Nightly Prices in
London's Airbnb Market?* A hedonic pricing analysis of Airbnb listings
across all 33 London boroughs.

## Which file produced the dataset

**`airbnb_scraper_full33_pricebands_final.ipynb` is the scraper that
produced the committed dataset.** It is the only file in this repository
that should be run to collect data. The other scraper files
(`scraper_21_07_fix.ipynb`, `airbnb_scraper_zones.py`) are earlier,
superseded iterations kept in the repository for transparency about the
project's development, not because they were used to produce any data
used in the analysis. See **File-by-file breakdown** below for what each
one is and why it was superseded.

## Pipeline overview

```
airbnb_scraper_full33_pricebands_final.ipynb
        │  (33 boroughs × 6 price bands = 198 queries, resumable)
        ▼
checkpoint_listings.csv + checkpoint_progress.json
        │  (written incrementally as the scrape runs; re-running the
        │   notebook resumes automatically rather than starting over)
        ▼
airbnb_london_full33_pricebands_<timestamp>.xlsx  +  .csv (canonical)
        │
        ▼
apply_minimum_borough_fix.py
        │  (renames search_location → search_borough, drops rows outside
        │   Greater London)
        ▼
<dataset>_borough_fixed.csv
        │
        ▼
hedonic_pricing_model.py
        │  (log-price OLS regression: excludes price_band_label, keeps
        │   unrated listings via is_new_listing dummy, excludes hotel-
        │   brand rows from the main model)
        ▼
Regression results (main model + supplementary hotel-brand model)
```

## File-by-file breakdown

| File | Role |
|---|---|
| `airbnb_scraper_full33_pricebands_final.ipynb` | **The scraper used to produce the committed dataset.** Selenium + BeautifulSoup, 33 boroughs × 6 price bands (198 queries), checkpointed/resumable, deduplicated by `room_id`. Outputs a timestamped `.xlsx` and canonical `.csv`. Run this top-to-bottom to reproduce or extend the dataset. |
| `airbnb_london_full33_pricebands_20260729_1148.xlsx` | Output of the scraper above — the raw, deduplicated dataset before the borough/cleaning fixes below. Kept as the original artifact; the `.csv` companion (produced when the notebook is re-run) is the canonical copy going forward, since `room_id` is a long identifier string that Excel can silently corrupt on re-save. |
| `apply_minimum_borough_fix.py` | Post-collection cleaning step. Renames the borough column to `search_borough` (making explicit that it records the *queried* borough, not a verified per-listing location) and drops rows naming places outside Greater London. Run this against the scraper's output before modelling. |
| `hedonic_pricing_model.py` | The regression script. Log-transforms nightly price as the dependent variable, excludes `price_band_label` and other price-derived columns as predictors, keeps unrated listings via an `is_new_listing` dummy instead of imputing ratings, and fits the main model on peer-to-peer listings only (hotel-brand listings are modelled separately as a supplementary result). |
| `scraper_21_07_fix.ipynb` | **Superseded.** An earlier 33-borough version that queried each borough once (no price-band splitting), which plateaued at roughly one page of results per borough (~1,000 rows total) before Airbnb started re-serving already-seen listings. Replaced by the price-band-split design in the final notebook, which multiplies unique results per borough instead of fighting that per-query depth limit. Kept for transparency on how the design evolved; not used to produce any data in the final dataset. |
| `airbnb_scraper_zones.py` | **Superseded.** The original prototype scraper — 6 boroughs only, no price-band splitting, no checkpointing, no `room_id`-based deduplication. Superseded by both later scrapers above. Kept for transparency; not used to produce any data in the final dataset. |
| `requirements.txt` | Pinned package versions for the scraper and modelling environment. |

## How to run the scraper

1. Install Google Chrome, if not already installed.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Open `airbnb_scraper_full33_pricebands_final.ipynb` and run the single
   code cell. On a first run this scrapes all 198 (borough, price band)
   queries; expect this to take a couple of hours given the built-in
   rate-limiting delays between page loads.
4. **If a run is interrupted** (crash, CAPTCHA, dropped connection, or a
   locked checkpoint file): just re-run the same cell. `checkpoint_listings.csv`
   and `checkpoint_progress.json` track exactly which queries already
   completed, and the notebook automatically skips them and scrapes only
   what's left. The first thing the cell prints on any run is a report of
   exactly which queries are outstanding, so you can see what a re-run will
   do before it does it.
5. The notebook saves both a `.xlsx` (for quick inspection) and a `.csv`
   (canonical — use this one downstream) with the same timestamped base name.

## How to run the cleaning and modelling steps

```
pip install -r requirements.txt

# 1. Apply the minimum borough-assignment fix to the scraper's output
#    (edit INPUT_FILE at the top of the script to point at your .csv)
python apply_minimum_borough_fix.py

# 2. Fit the hedonic pricing regression
#    (edit INPUT_FILE at the top of the script to point at the
#    *_borough_fixed.csv produced above)
python hedonic_pricing_model.py
```

## Known limitations

- **Spatial measurement error.** `search_borough` records which borough
  was queried, not a verified location for each listing — Airbnb's search
  radius crosses borough lines. Rows outside Greater London are dropped
  by `apply_minimum_borough_fix.py`, but deduplication still credits an
  overlapping listing to whichever borough sorts first alphabetically,
  which understates density in boroughs bordering many others (Newham,
  Tower Hamlets, City of London in particular).
- **Sampling depth.** Most (borough, price band) queries returned only
  one page of results before the no-new-listings guard stopped them, so
  the dataset should be read as a sample of Airbnb's own top-ranked
  listings per query, not a random sample of London's full Airbnb supply.

Both are discussed in full in the dissertation's methodology/limitations
chapter.

## Ethical note

This project only reads publicly visible Airbnb search-result HTML — no
login, no bypassing paywalls or CAPTCHAs, no private host/guest data.
Requests are rate-limited between page loads. See the docstring at the
top of `airbnb_scraper_full33_pricebands_final.ipynb` for the full
ethical/legal note included for the methodology write-up.
