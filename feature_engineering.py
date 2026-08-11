"""
feature_engineering.py
------------------------------------------------------------------------------------
Cleaning and feature engineering step, run AFTER apply_minimum_borough_fix.py and
BEFORE hedonic_pricing_model.py.

CLEANING
  1. Drops `nights` and `nights_source` - both are constant across every row
     (nights == 4 for all 2,520 rows; nights_source == "assumed_from_search_dates"
     for all of them), so they carry zero information and would only add
     confusing dead columns to the modelling dataset.
  2. Resolves one data-quality exception: room_id 1732660849058230636 has
     review_count == 2 but rating is missing, despite is_new_listing == False.
     Every other row is internally consistent (is_new_listing is True iff
     rating is missing - see the crosstab printed at runtime). This one row
     looks like a scrape parsing gap (the review count rendered but the star
     rating didn't), not a genuinely unrated listing, so it doesn't fit the
     is_new_listing / zero-fill rule that hedonic_pricing_model.py applies to
     every other missing rating. Given it is a single row out of 2,520 and
     cannot be reliably recovered, it is dropped here with the reason logged,
     rather than silently zero-filled or left to break the regression's
     assumptions downstream.
  3. Range checks on rating (0-5), review_count (>= 0), and
     price_gbp_per_night (> 0) - prints a warning (does not silently drop) if
     any are violated, since none were expected to be given the round-1 fixes.

FEATURE ENGINEERING
  1. listing_type: a finer-grained category than `property_type` (which only
     has 5 buckets: Flat/Room/House/Hotel/Other-Unknown), extracted from the
     listing-type word(s) at the START of the title (e.g. "Room in Croydon"
     -> "Room", "Guest suite in Newham" -> "Guest suite"). This is the
     standard Airbnb listing-type distinction (entire flat vs private room vs
     shared room vs guest house etc.) that hedonic pricing studies of Airbnb
     consistently find to be one of the strongest price predictors, and
     property_type collapses several of these distinctions together (e.g.
     "Home", "Townhouse", and "Guest suite" all fold into `property_type ==
     House`, even though a shared "Room" and an entire "Home" price very
     differently). Categories with fewer than 15 listings are grouped into
     "Other" so the model isn't fitting near-singleton dummy levels; the 96
     hotel-brand listings, whose titles don't follow the "<type> in <place>"
     pattern, are labelled "Hotel".
     NOTE: listing_type is a refinement of property_type, not an independent
     feature - the two are collinear by construction. Use ONE of them in the
     model, not both (hedonic_pricing_model.py has been updated to use
     listing_type; property_type is kept in the output for reference/checks).
  2. log_review_count = log1p(review_count). review_count is heavily
     right-skewed (a small number of listings have hundreds of reviews, most
     have single digits), so a handful of very-reviewed listings would
     otherwise dominate a linear term. log1p handles the review_count == 0
     rows (new/unrated listings) without producing -inf.

USAGE
    python feature_engineering.py

Edit INPUT_FILE below to point at the output of apply_minimum_borough_fix.py.
"""

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
INPUT_FILE = "airbnb_london_full33_pricebands_20260804_1347_borough_fixed.csv"
OUTPUT_FILE = "airbnb_london_full33_pricebands_20260804_1347_features.csv"

# Title-prefix categories kept as their own listing_type level; anything
# else (excluding hotel-brand rows, handled separately) collapses to "Other".
MIN_LISTING_TYPE_COUNT = 15


def load_dataset(path):
    return pd.read_csv(path, dtype={"room_id": str})


def clean(df):
    df = df.copy()
    before = len(df)

    # --- Drop constant columns ---
    constant_cols = [c for c in ["nights", "nights_source"] if c in df.columns]
    for c in constant_cols:
        n_unique = df[c].nunique()
        print(f"Dropping '{c}': constant across all rows (nunique={n_unique}).")
    df = df.drop(columns=constant_cols)

    # --- Resolve the single rating/review_count/is_new_listing exception ---
    mismatched = df.loc[df["is_new_listing"] != df["rating"].isna()]
    if not mismatched.empty:
        print(f"\nDropping {len(mismatched)} row(s) where is_new_listing is "
              f"inconsistent with a missing rating (data-quality exception, "
              f"not fixable from available fields):")
        print(mismatched[["room_id", "title", "rating", "review_count", "is_new_listing"]]
              .to_string(index=False))
        df = df.loc[~df.index.isin(mismatched.index)].reset_index(drop=True)

    # --- Range checks (warn, don't silently drop - none expected here) ---
    bad_rating = df["rating"].notna() & ~df["rating"].between(0, 5)
    bad_reviews = df["review_count"].notna() & (df["review_count"] < 0)
    bad_price = df["price_gbp_per_night"] <= 0
    for name, mask in [("rating outside 0-5", bad_rating),
                        ("negative review_count", bad_reviews),
                        ("non-positive price_gbp_per_night", bad_price)]:
        if mask.any():
            print(f"  WARNING: {mask.sum()} row(s) with {name} - inspect before modelling.")

    after = len(df)
    print(f"\nCleaning: {before} rows -> {after} rows ({before - after} dropped).")
    return df


def add_listing_type(df):
    df = df.copy()
    prefix = df["title"].str.extract(r"^(.*?) in ")[0]

    is_hotel_brand = df["is_hotel_brand"].astype(bool)
    prefix = prefix.where(~is_hotel_brand, "Hotel")
    # Any remaining non-hotel-brand row with no extractable prefix (shouldn't
    # happen post-borough-fix, but guard rather than silently mislabel).
    unresolved = prefix.isna()
    if unresolved.any():
        print(f"  WARNING: {unresolved.sum()} non-hotel-brand row(s) with no "
              f"extractable listing_type - labelled 'Other'.")
        prefix = prefix.fillna("Other")

    counts = prefix.value_counts()
    keep = set(counts[counts >= MIN_LISTING_TYPE_COUNT].index) | {"Hotel"}
    df["listing_type"] = prefix.where(prefix.isin(keep), "Other")

    print("\nlisting_type distribution:")
    print(df["listing_type"].value_counts().to_string())
    print("\n(cross-check against property_type - listing_type is a refinement of it)")
    print(pd.crosstab(df["listing_type"], df["property_type"]).to_string())
    return df


def add_log_review_count(df):
    df = df.copy()
    df["log_review_count"] = np.log1p(df["review_count"].fillna(0))
    return df


if __name__ == "__main__":
    print(f"Loading {INPUT_FILE} ...")
    df = load_dataset(INPUT_FILE)

    df = clean(df)
    df = add_listing_type(df)
    df = add_log_review_count(df)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved cleaned + feature-engineered dataset to {OUTPUT_FILE}")
    print(f"Final shape: {df.shape}")
    print(
        "\nNext step: hedonic_pricing_model.py's INPUT_FILE has been updated to "
        "point at this file, with listing_type replacing property_type and "
        "log_review_count added alongside rating as predictors."
    )
