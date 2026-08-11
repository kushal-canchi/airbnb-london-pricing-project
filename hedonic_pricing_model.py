"""
hedonic_pricing_model.py
------------------------------------------------------------------------------------
Baseline hedonic pricing regression (OLS via statsmodels) on the cleaned
Airbnb London dataset, implementing four housekeeping rules from
supervisor feedback:

  1. price_band_label is EXCLUDED as a predictor. It is derived directly
     from price (price_min/price_max used to build the search query), so
     including it would put a function of the dependent variable on the
     right-hand side of the equation.
  2. The dependent variable is log-transformed. Nightly price is heavily
     right-skewed (median ~£156, max ~£2,900+), which is standard for
     accommodation pricing data and is the reason hedonic pricing models
     since Rosen (1974) conventionally use log(price) as the DV.
  3. The 337 listings with no rating/review_count are KEPT, not dropped
     and not imputed with a plausible rating. is_new_listing (already a
     boolean column from the scraper) is used as a dummy predictor, and
     rating/review_count are zero-filled ONLY for the rows where
     is_new_listing is True, so the model can still be estimated on the
     full sample without pretending to know a rating that was never
     observed. is_new_listing absorbs the effect of that placeholder.
  4. The 105 hotel-brand rows (is_hotel_brand) are excluded from the main
     model - a hedonic model is about peer-to-peer host pricing, and a
     commercial hotel product priced by different logic would distort
     the coefficients if pooled in. They are modelled separately below
     as a supplementary result, not silently dropped from the project.

USAGE
    python hedonic_pricing_model.py

Edit INPUT_FILE below to point at your cleaned, borough-fixed dataset
(the output of apply_minimum_borough_fix.py, or your final feature-
engineered CSV if that step has already folded this one in).

Requires: pandas, numpy, statsmodels
    pip install pandas numpy statsmodels
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# ------------------------------------------------------------------
# CONFIG - edit to match your file and column names
# ------------------------------------------------------------------
INPUT_FILE = "airbnb_london_full33_pricebands_20260804_1347_features.csv"

DV_COLUMN = "price_gbp_per_night"          # raw nightly price before log transform

# Predictors to include in the main model. Adjust to match whatever
# feature-engineering has added (e.g. distance_to_center, bedrooms) -
# just never add price_band_label, price_band_min, price_band_max,
# price_gbp_total, or price_raw: all of these are derived from price.
# listing_type (from feature_engineering.py) replaces property_type here -
# it's a finer-grained version of the same information (e.g. it separates
# "Room" from "Guest suite" and "Townhouse", which property_type folds
# together), so use one or the other, not both.
CATEGORICAL_PREDICTORS = ["zone", "listing_type"]      # add "search_borough" here instead of "zone" for the 33-borough version of the model
NUMERIC_PREDICTORS = ["rating", "log_review_count"]      # rating zero-filled for is_new_listing rows, see prepare_model_data(); log_review_count from feature_engineering.py
DUMMY_PREDICTORS = ["is_new_listing"]

# Columns that must NEVER be used as predictors, kept here as an explicit
# guard so a later edit can't accidentally reintroduce them.
FORBIDDEN_PREDICTORS = {
    "price_band_label", "price_band_min", "price_band_max",
    "price_gbp_total", "price_raw", "log_price", DV_COLUMN,
}


def load_dataset(path):
    return pd.read_csv(path, dtype={"room_id": str})


def prepare_model_data(df):
    """
    Applies housekeeping rules 2 and 3, and returns the DataFrame used to
    build the main-model formula, plus the two other views needed for
    rules 1 and 4 (asserted / applied at the formula-building stage).
    """
    df = df.copy()

    # Rule 2: log-transform the DV. A small floor guards against log(0)
    # if any zero/negative prices slipped through upstream cleaning -
    # these should already be filtered out, but this keeps the model
    # from silently producing -inf if one does.
    invalid_price = df[DV_COLUMN].isna() | (df[DV_COLUMN] <= 0)
    if invalid_price.any():
        print(f"  Dropping {invalid_price.sum()} rows with missing/non-positive "
              f"{DV_COLUMN} before log transform (cannot take log of these).")
        df = df.loc[~invalid_price].copy()
    df["log_price"] = np.log(df[DV_COLUMN])

    # Rule 3: keep unrated listings, zero-fill rating/review_count ONLY
    # for those rows (is_new_listing == True), and rely on is_new_listing
    # as the dummy that tells the model "this zero is a placeholder, not
    # a genuinely low rating/zero reviews".
    for col in ["rating", "review_count"]:
        if col in df.columns:
            missing_before = df[col].isna().sum()
            df[col] = df[col].fillna(0)
            print(f"  {col}: zero-filled {missing_before} missing values "
                  f"(all should correspond to is_new_listing == True rows).")

    mismatched = df.loc[df["is_new_listing"] != (df["rating"] == 0), "is_new_listing"]
    if not mismatched.empty:
        print(f"  NOTE: {len(mismatched)} rows have is_new_listing inconsistent with "
              f"a zero rating - worth spot-checking before trusting the dummy fully.")

    return df


def build_formula(dv, categorical, numeric, dummy):
    """
    Builds a statsmodels formula string, wrapping categorical predictors
    in C(...) so they're treated as factors (dummy-coded) rather than
    numeric. Raises if a forbidden predictor sneaks into the call, so a
    later edit can't silently reintroduce price_band_label.
    """
    all_predictors = list(categorical) + list(numeric) + list(dummy)
    used_forbidden = FORBIDDEN_PREDICTORS.intersection(all_predictors)
    if used_forbidden:
        raise ValueError(
            f"Forbidden predictor(s) in the model: {used_forbidden}. "
            f"These are derived from price and must not be on the right-hand side."
        )

    terms = [f"C({col})" for col in categorical] + list(numeric) + list(dummy)
    return f"{dv} ~ " + " + ".join(terms)


def run_model(df, formula, label):
    print(f"\n=== {label} ===")
    print(f"Formula: {formula}")
    print(f"N = {len(df)}")
    model = smf.ols(formula=formula, data=df).fit()
    print(model.summary())
    return model


if __name__ == "__main__":
    print(f"Loading {INPUT_FILE} ...")
    df = load_dataset(INPUT_FILE)
    df = prepare_model_data(df)

    formula = build_formula(
        dv="log_price",
        categorical=CATEGORICAL_PREDICTORS,
        numeric=NUMERIC_PREDICTORS,
        dummy=DUMMY_PREDICTORS,
    )

    # Rule 4: main model excludes hotel-brand rows entirely.
    if "is_hotel_brand" in df.columns:
        main_df = df.loc[~df["is_hotel_brand"]].copy()
        hotel_df = df.loc[df["is_hotel_brand"]].copy()
        print(f"\nExcluding {len(hotel_df)} hotel-brand rows from the main model "
              f"({len(main_df)} peer-to-peer listings remain).")
    else:
        print("\nWARNING: is_hotel_brand column not found - main model will include "
              "any hotel/aparthotel-brand listings. Check the input file.")
        main_df = df.copy()
        hotel_df = pd.DataFrame()

    main_model = run_model(main_df, formula, "MAIN MODEL (peer-to-peer listings only)")

    # Supplementary result: hotel-brand listings modelled separately,
    # not silently discarded from the project.
    if len(hotel_df) >= 20:  # arbitrary floor - too few rows to fit meaningfully below this
        hotel_model = run_model(hotel_df, formula, "SUPPLEMENTARY MODEL (hotel-brand listings only)")
    else:
        print(f"\nSkipping supplementary hotel-brand model - only {len(hotel_df)} rows, "
              f"too few to fit meaningfully. Report the count and describe them "
              f"qualitatively instead.")
