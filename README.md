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
        │  (renames search_location → search_borough; screens each row's
        │   title-derived location against Greater London and drops rows
        │   outside it — search_borough itself can't be used for this,
        │   since it only ever holds the 33 queried borough strings)
        ▼
<dataset>_borough_fixed.csv
        │
        ▼
feature_engineering.py
        │  (drops constant/dead columns, resolves one rating data-quality
        │   exception, adds listing_type from the title and
        │   log_review_count)
        ▼
<dataset>_features.csv
        │
        ▼
hedonic_pricing_model.py
        │  (RQ1: fits the peer-to-peer/rated main model + three
        │   supplementary subgroups, with VIF diagnostics on each)
        ▼
moderation_analysis.py
        │  (RQ2: borough/zone × listing_type interaction models,
        │   compared via adjusted R² and AIC)
        ▼
Regression results (main model + supplementary hotel-brand model)
```

## File-by-file breakdown

| File | Role |
|---|---|
| `airbnb_scraper_full33_pricebands_final.ipynb` | **The scraper used to produce the committed dataset.** Selenium + BeautifulSoup, 33 boroughs × 6 price bands (198 queries), checkpointed/resumable, deduplicated by `room_id`. Outputs a timestamped `.xlsx` and canonical `.csv`. Run this top-to-bottom to reproduce or extend the dataset. |
| `airbnb_london_full33_pricebands_20260804_1347.xlsx` | Output of the scraper above — the raw, deduplicated dataset before the borough/cleaning fixes below, regenerated from the current canonical `.csv` (the previous `_20260729_1148.xlsx` was a stale July export with a different row count/column set and has been removed). The `.csv` companion is still the canonical copy for downstream work, since `room_id` is a long identifier string that Excel can silently corrupt on re-save. |
| `apply_minimum_borough_fix.py` | Post-collection cleaning step. Renames the borough column to `search_borough` (making explicit that it records the *queried* borough, not a verified per-listing location) and drops rows whose listing **title** names a place outside Greater London — `search_borough` itself only ever holds the 33 queried borough strings, so it can't be used to detect this. Run this against the scraper's output before modelling. |
| `airbnb_london_full33_pricebands_20260804_1347_borough_fixed.csv` | Output of `apply_minimum_borough_fix.py`. Intermediate file, kept for transparency on what the geographic filter removed — not the file to build on further. |
| `feature_engineering.py` | Cleaning + feature engineering step, run on the borough-fixed dataset. Drops `nights`/`nights_source` (constant across every row), drops one row with an unresolvable rating/review_count/is_new_listing inconsistency, and adds two predictors: `listing_type` (a finer-grained category parsed from the listing title — e.g. separates "Room" from "Guest suite" and "Townhouse", which `property_type` folds together) and `log_review_count` (log1p of `review_count`, since it's heavily right-skewed). |
| `airbnb_london_full33_pricebands_20260804_1347_features.csv` | **Output of `feature_engineering.py` — this is the canonical file for all further work** (modelling, write-up, any additional analysis). 2,519 rows, cleaned and feature-engineered. `hedonic_pricing_model.py` reads this file directly. |
| `hedonic_pricing_model.py` | The regression script. Log-transforms nightly price as the dependent variable, excludes `price_band_label` and other price-derived columns as predictors, and uses `listing_type`/`log_review_count` from the feature engineering step. Fits **four subgroups separately** rather than one pooled model: peer-to-peer/rated (main model, answers RQ1), peer-to-peer/new-unrated, hotel-brand/rated, and hotel-brand/new-unrated (too few rows to model, reported as a count only). This split replaced an earlier version that zero-filled rating for unrated listings and used an `is_new_listing` dummy in one pooled model — VIF diagnostics showed that produced near-perfect structural collinearity (VIF > 120) between `is_new_listing` and `rating`, since one was almost entirely predictable from the other by construction. Also runs VIF diagnostics on every fitted subgroup (proposal's threshold: VIF < 5) and only triggers a LASSO robustness check if a subgroup exceeds it — currently only the small peer-to-peer/new-unrated supplementary model exceeds 5 (max 6.3), the main RQ1 model is clean (max 2.7). |
| `moderation_analysis.py` | RQ2 — does borough moderate the relationship between listing characteristics and price? Fits the RQ1 baseline plus three moderation specifications on the same peer-to-peer/rated sample: borough main effects only, the proposal's literal `search_borough × listing_type` interaction, and a `zone × listing_type` interaction as a fallback. The literal borough interaction wins on AIC/adjusted R² but 67% of its 256 interaction terms rest on fewer than 5 listings in that cell — an overfitting artifact flagged automatically by comparing each interaction term's actual cell count against a threshold, not by how significant it looks. Per the proposal's own Section 4.3 risk mitigation ("group boroughs into Inner/Outer if the full model is poorly identified"), the `zone × listing_type` model is reported as the primary RQ2 finding instead: a modest, genuine improvement (adj. R² 0.615→0.620, AIC 3105→3082) with moderation concentrated in minor listing-type categories, not the dominant ones. |
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

# 2. Clean and feature-engineer
#    (edit INPUT_FILE at the top of the script to point at the
#    *_borough_fixed.csv produced above)
python feature_engineering.py

# 3. Fit the hedonic pricing regression (RQ1)
#    (edit INPUT_FILE at the top of the script to point at the
#    *_features.csv produced above)
python hedonic_pricing_model.py

# 4. Moderation analysis (RQ2) - reuses hedonic_pricing_model.py's
#    config directly, so run it from the same directory
python moderation_analysis.py
```

## Known limitations

- **Spatial measurement error.** `search_borough` records which borough
  was queried, not a verified location for each listing — Airbnb's search
  radius crosses borough lines. Rows outside Greater London are identified
  from the listing title and dropped by `apply_minimum_borough_fix.py`
  (205 of 2,725 rows, 7.5%, concentrated in Hillingdon, Havering, Sutton,
  Croydon, Hounslow, and Harrow), but deduplication still credits an
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
