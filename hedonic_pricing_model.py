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
  3. Listings with no rating (is_new_listing == True) are NOT pooled into
     the same model as rated listings. An earlier version zero-filled
     rating/review_count for these rows and added is_new_listing as a
     dummy predictor - but VIF diagnostics showed this creates near-
     perfect structural collinearity (VIF > 120 on both is_new_listing
     and rating), because rating == 0 if and only if is_new_listing ==
     True for most of the sample: one variable is almost entirely
     predictable from the other by construction, not because of any real
     relationship. Rather than mask that with LASSO, the sample is now
     split the same way hotel-brand listings already were (rule 4):
     rated and unrated listings are modelled separately, so no model
     ever has to hold both a variable and its own "this was missing"
     flag at the same time.
  4. The 115 hotel-brand rows (is_hotel_brand) are excluded from the main
     model - a hedonic model is about peer-to-peer host pricing, and a
     commercial hotel product priced by different logic would distort
     the coefficients if pooled in. They are modelled separately below
     as a supplementary result, not silently dropped from the project.

This produces four groups (peer-to-peer / hotel-brand, crossed with
rated / new-and-unrated), modelled separately wherever there are enough
rows to fit meaningfully:

  A. MAIN MODEL           peer-to-peer, rated       (~2,096 rows) - RQ1
  B. SUPPLEMENTARY        peer-to-peer, new/unrated (~308 rows)
  C. SUPPLEMENTARY        hotel-brand, rated        (~103 rows)
  D. (not modelled)       hotel-brand, new/unrated  (~12 rows - too few)

USAGE
    python hedonic_pricing_model.py

Edit INPUT_FILE below to point at your cleaned, borough-fixed dataset
(the output of apply_minimum_borough_fix.py, or your final feature-
engineered CSV if that step has already folded this one in).

Requires: pandas, numpy, statsmodels, patsy
    pip install pandas numpy statsmodels patsy
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan

# Proposal's own success criterion (Section 5): VIF < 5 for all retained
# predictors. The commonly-cited rule-of-thumb ceiling is VIF > 10 -
# reported separately below since the proposal's threshold is stricter.
VIF_THRESHOLD = 5

# Below this row count, a subgroup is reported (count, descriptive stats)
# but not modelled - too few rows to fit meaningfully.
MIN_ROWS_TO_MODEL = 20

# ------------------------------------------------------------------
# CONFIG - edit to match your file and column names
# ------------------------------------------------------------------
INPUT_FILE = "airbnb_london_full33_pricebands_20260804_1347_features.csv"

DV_COLUMN = "price_gbp_per_night"          # raw nightly price before log transform

# Predictors used for RATED subgroups (groups A and C above) - includes
# rating/log_review_count, since those are meaningful here.
# listing_type (from feature_engineering.py) replaces property_type here -
# it's a finer-grained version of the same information (e.g. it separates
# "Room" from "Guest suite" and "Townhouse", which property_type folds
# together), so use one or the other, not both.
CATEGORICAL_PREDICTORS = ["zone", "listing_type"]      # add "search_borough" here instead of "zone" for the 33-borough version of the model
RATED_NUMERIC_PREDICTORS = ["rating", "log_review_count"]

# Predictors used for the NEW/UNRATED subgroup (group B) - rating and
# log_review_count are constant (0) for every row in this subgroup by
# definition, so including them would just be dead columns; is_new_listing
# is dropped too since it's constant True.
UNRATED_NUMERIC_PREDICTORS = []

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
    Applies housekeeping rule 2 (log-transform the DV) and returns the
    full DataFrame. Rule 3's rated/unrated split now happens in __main__,
    not here - no zero-filling of rating/review_count is needed since
    the two subgroups are modelled separately.
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


def compute_vif(df, formula, label):
    """
    Builds the same design matrix statsmodels fits on (dummy-coded
    categoricals included) and computes VIF per column. The Intercept
    column is excluded from the report - its VIF is not meaningful.
    """
    y, X = dmatrices(formula, data=df, return_type="dataframe")
    vif = pd.DataFrame({
        "predictor": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
    })
    vif = vif.loc[vif["predictor"] != "Intercept"].sort_values("VIF", ascending=False)

    print(f"\n--- VIF diagnostics: {label} ---")
    print(vif.to_string(index=False))

    over_proposal_threshold = vif.loc[vif["VIF"] >= VIF_THRESHOLD]
    over_rule_of_thumb = vif.loc[vif["VIF"] >= 10]
    if over_rule_of_thumb.empty and over_proposal_threshold.empty:
        print(f"All predictors are below VIF {VIF_THRESHOLD} - no multicollinearity "
              f"concern, proposal's success criterion is met, LASSO robustness "
              f"check is not required by the proposal's own conditional trigger.")
    elif over_rule_of_thumb.empty:
        print(f"No predictor exceeds the VIF>10 rule-of-thumb ceiling, but "
              f"{len(over_proposal_threshold)} exceed the proposal's stricter "
              f"VIF<{VIF_THRESHOLD} success criterion - worth noting as a limitation "
              f"even if a LASSO check isn't strictly triggered.")
    else:
        print(f"WARNING: {len(over_rule_of_thumb)} predictor(s) exceed VIF>10 - "
              f"multicollinearity detected, run LASSO as a robustness check per "
              f"the proposal's Section 4.3 risk mitigation.")
    return vif


def compute_robust_and_clustered_se(model, df, formula, cluster_col, label):
    """
    Refits `model` with HC3-robust and cluster-robust (by `cluster_col`)
    standard errors, and reports a residual intraclass correlation (ICC)
    for `cluster_col`. This is the same specification comparison reported
    in the dissertation (Section 3.4, Table 2): a Breusch-Pagan test
    motivates HC3, but HC3 alone does not correct for within-cluster
    error correlation, so the residual ICC is checked directly and
    borough-clustered standard errors are adopted as the PRIMARY
    specification whenever that ICC is non-negligible.

    Returns (model_hc3, model_clustered, icc) so callers can use the
    clustered p-values as the primary significance test.
    """
    model_hc3 = smf.ols(formula=formula, data=df).fit(cov_type="HC3")
    model_clustered = smf.ols(formula=formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df[cluster_col]}
    )

    resid_df = pd.DataFrame({"resid": model.resid, "group": df[cluster_col]})
    grand_mean = resid_df["resid"].mean()
    group_means = resid_df.groupby("group")["resid"].mean()
    group_ns = resid_df.groupby("group")["resid"].size()
    k = len(group_ns)
    n = len(resid_df)
    ss_between = (group_ns * (group_means - grand_mean) ** 2).sum()
    ss_total = ((resid_df["resid"] - grand_mean) ** 2).sum()
    ms_between = ss_between / (k - 1)
    ms_within = (ss_total - ss_between) / (n - k)
    n0 = (n - (group_ns ** 2).sum() / n) / (k - 1)
    icc = (ms_between - ms_within) / (ms_between + (n0 - 1) * ms_within)

    print(f"\n--- Standard-error specification comparison: {label} ---")
    bp_stat, bp_p, _, _ = het_breuschpagan(model.resid, model.model.exog)
    print(f"Breusch-Pagan LM statistic: {bp_stat:.2f}, p-value: {bp_p:.2e} "
          f"({'rejects' if bp_p < 0.05 else 'fails to reject'} homoskedasticity)")
    print(f"Residual ICC by {cluster_col} ({k} groups, N={n}): {icc:.4f} "
          f"(~{icc*100:.1f}% of residual variance sits between groups)")
    print("PRIMARY specification used throughout the dissertation: standard errors "
          f"clustered by {cluster_col}, since HC3 alone does not correct for the "
          "within-group correlation the ICC above indicates.")

    comparison = pd.DataFrame({
        "coef": model.params,
        "p_conventional": model.pvalues,
        "p_hc3": model_hc3.pvalues,
        "p_clustered": model_clustered.pvalues,
    }).round(4)
    print(comparison.to_string())

    return model_hc3, model_clustered, icc


if __name__ == "__main__":
    print(f"Loading {INPUT_FILE} ...")
    df = load_dataset(INPUT_FILE)
    df = prepare_model_data(df)

    if "is_hotel_brand" not in df.columns or "is_new_listing" not in df.columns:
        raise KeyError("Expected both is_hotel_brand and is_new_listing columns - "
                        "check the input file is the feature-engineered dataset.")

    rated_formula = build_formula(
        dv="log_price",
        categorical=CATEGORICAL_PREDICTORS,
        numeric=RATED_NUMERIC_PREDICTORS,
        dummy=[],
    )
    unrated_formula = build_formula(
        dv="log_price",
        categorical=CATEGORICAL_PREDICTORS,
        numeric=UNRATED_NUMERIC_PREDICTORS,
        dummy=[],
    )

    is_hotel = df["is_hotel_brand"].astype(bool)
    is_new = df["is_new_listing"].astype(bool)

    groups = {
        "MAIN MODEL (peer-to-peer, rated listings)": (~is_hotel & ~is_new, rated_formula),
        "SUPPLEMENTARY (peer-to-peer, new/unrated listings)": (~is_hotel & is_new, unrated_formula),
        "SUPPLEMENTARY (hotel-brand, rated listings)": (is_hotel & ~is_new, rated_formula),
        "NOT MODELLED (hotel-brand, new/unrated listings)": (is_hotel & is_new, None),
    }

    print("\nGroup sizes:")
    for label, (mask, _) in groups.items():
        print(f"  {label}: {mask.sum()} rows")

    fitted_models = {}
    fitted_dfs = {}
    for label, (mask, formula) in groups.items():
        group_df = df.loc[mask].copy()
        if formula is None or len(group_df) < MIN_ROWS_TO_MODEL:
            print(f"\nSkipping '{label}' - only {len(group_df)} rows, "
                  f"too few to fit meaningfully. Report the count and describe "
                  f"qualitatively instead.")
            continue
        model = run_model(group_df, formula, label)
        compute_vif(group_df, formula, label)
        fitted_models[label] = model
        fitted_dfs[label] = group_df

    # The main RQ1 model (peer-to-peer, rated listings) is the one Table 2 of
    # the dissertation reports, and it is reported there under borough-
    # clustered standard errors as the primary specification (Section 3.4) -
    # not the classical/nonrobust ones printed by run_model() above. This
    # reproduces that comparison here so this script's own output matches
    # what the dissertation actually reports, rather than only ever printing
    # the classical specification.
    main_label = "MAIN MODEL (peer-to-peer, rated listings)"
    if main_label in fitted_models:
        compute_robust_and_clustered_se(
            fitted_models[main_label], fitted_dfs[main_label], rated_formula,
            cluster_col="search_borough", label=main_label,
        )
