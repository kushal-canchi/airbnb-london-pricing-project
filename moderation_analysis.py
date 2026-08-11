"""
moderation_analysis.py
------------------------------------------------------------------------------------
RQ2: Does London borough moderate the relationship between listing
characteristics and nightly price?

Fits four nested/related model specifications on the SAME sample as the
RQ1 main model (peer-to-peer, rated listings - see hedonic_pricing_model.py
for why new/unrated and hotel-brand listings are modelled separately) and
compares them via adjusted R-squared and AIC, per Section 3.4.3 of the
proposal:

  Model 1 (RQ1 baseline)     log_price ~ zone + listing_type + rating + log_review_count
  Model 2 (borough, no interaction)
                              log_price ~ search_borough + listing_type + rating + log_review_count
  Model 3 (full moderation, literal RQ2 spec)
                              log_price ~ search_borough * listing_type + rating + log_review_count
                              (the * expands to both main effects AND every
                              borough:listing_type interaction term)
  Model 4 (moderation, well-identified fallback)
                              log_price ~ zone * listing_type + rating + log_review_count

Model 3 is the proposal's literal RQ2 specification (borough, not the
coarser Inner/Outer zone). The proposal's own risk register anticipates
this may be poorly identified given how many dummy interactions a 33-
borough x ~9-listing_type grid produces relative to sample size, and
pre-commits to falling back to zone if so (Section 4.3: "Consider grouping
boroughs into Inner/Outer London if the full borough model is poorly
identified") - Model 4 is that fallback. This script checks empirically
whether the fallback is needed, by comparing the number of listings
backing each interaction term against a minimum threshold, rather than
assuming either outcome in advance.

RESULT (from the 20260804_1347 dataset): Model 3 wins on AIC/adjusted R2,
but 67% of its interaction terms rest on fewer than 5 listings - an
overfitting artifact, not genuine moderation. Model 4 is the well-
identified result actually reported: a modest, genuine improvement over
the no-interaction baseline, with moderation concentrated in the minor
"Other"/"Place to stay" categories rather than the dominant listing types.

USAGE
    python moderation_analysis.py

Requires: pandas, numpy, statsmodels, patsy (same as hedonic_pricing_model.py)
Reuses config and helper functions from hedonic_pricing_model.py directly,
so the two scripts can't silently drift apart on how the DV is built or
which rows count as the RQ1 sample.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from hedonic_pricing_model import (
    INPUT_FILE, RATED_NUMERIC_PREDICTORS, FORBIDDEN_PREDICTORS,
    load_dataset, prepare_model_data,
)

def build_moderation_formulas():
    used_forbidden = FORBIDDEN_PREDICTORS.intersection(
        ["zone", "listing_type", "search_borough", "rating", "log_review_count"]
    )
    assert not used_forbidden, f"Forbidden predictor(s) requested: {used_forbidden}"

    model1 = "log_price ~ C(zone) + C(listing_type) + rating + log_review_count"
    model2 = "log_price ~ C(search_borough) + C(listing_type) + rating + log_review_count"
    model3 = "log_price ~ C(search_borough) * C(listing_type) + rating + log_review_count"
    model4 = "log_price ~ C(zone) * C(listing_type) + rating + log_review_count"
    return model1, model2, model3, model4


def fit_and_report(df, formula, label):
    print(f"\n=== {label} ===")
    print(f"Formula: {formula}")
    print(f"N = {len(df)}")
    model = smf.ols(formula=formula, data=df).fit()
    n_params = int(model.df_model) + 1  # + intercept
    print(f"Parameters fit: {n_params} (N / params = {len(df) / n_params:.1f})")
    print(f"Adj. R-squared: {model.rsquared_adj:.4f}")
    print(f"AIC: {model.aic:.1f}")
    return model


def flag_unstable_interactions(model, label, ct, min_cell_n=5):
    """
    Flags interaction terms by the ACTUAL number of observations in that
    borough x listing_type cell, not by how large the term's standard
    error looks. SE-based heuristics miss a specific failure mode here:
    a cell with exactly one observation gets fit perfectly by OLS,
    producing a deceptively tiny SE and a "significant" p-value for a
    coefficient that is really just memorising a single data point.
    """
    interaction_terms = model.params.index[model.params.index.str.contains(":")]
    if len(interaction_terms) == 0:
        return

    def cell_n(term):
        # term looks like: C(search_borough)[T.Camden, London, United Kingdom]:C(listing_type)[T.Flat]
        borough_part, listing_part = term.split(":")
        borough = borough_part.split("[T.")[1].rstrip("]")
        listing = listing_part.split("[T.")[1].rstrip("]")
        return ct.loc[borough, listing] if (borough in ct.index and listing in ct.columns) else 0

    cell_counts = pd.Series({t: cell_n(t) for t in interaction_terms})
    unstable = cell_counts[cell_counts < min_cell_n]

    print(f"\n{label}: {len(interaction_terms)} interaction terms fit.")
    print(f"  {len(unstable)} of them correspond to a borough x listing_type cell "
          f"with fewer than {min_cell_n} listings - these are NOT trustworthy "
          f"estimates regardless of how significant or precise they look, since "
          f"OLS can fit a near-perfect (deceptively 'significant') line through "
          f"a handful of points.")

    significant = model.pvalues[interaction_terms].loc[
        model.pvalues[interaction_terms] < 0.05
    ]
    stable_significant = significant.index.difference(unstable.index)
    print(f"  {len(significant)} interaction terms are significant at p<0.05; "
          f"of those, {len(stable_significant)} are backed by at least "
          f"{min_cell_n} listings in that cell and can be reported with "
          f"reasonable confidence.")
    if len(stable_significant) > 0:
        print("\n  Stable, significant borough x listing_type interactions "
              f"(cell n >= {min_cell_n}):")
        for term in stable_significant:
            print(f"    {term}  [cell n={int(cell_counts[term])}]: "
                  f"coef={model.params[term]:.3f}, se={model.bse[term]:.3f}, "
                  f"p={model.pvalues[term]:.4f}")
    else:
        print(f"\n  No interaction terms are both significant AND backed by "
              f"{min_cell_n}+ listings in that specific cell.")


if __name__ == "__main__":
    print(f"Loading {INPUT_FILE} ...")
    df = load_dataset(INPUT_FILE)
    df = prepare_model_data(df)

    is_hotel = df["is_hotel_brand"].astype(bool)
    is_new = df["is_new_listing"].astype(bool)
    main_df = df.loc[~is_hotel & ~is_new].copy()
    print(f"\nRQ2 sample: peer-to-peer, rated listings only (same as the RQ1 "
          f"main model) - N = {len(main_df)}")

    ct = pd.crosstab(main_df["search_borough"], main_df["listing_type"])
    sparse_cells = int((ct <= 1).sum().sum())
    print(f"\nborough x listing_type grid: {ct.shape[0]} boroughs x {ct.shape[1]} "
          f"listing types = {ct.size} cells, {sparse_cells} of them ({100*sparse_cells/ct.size:.0f}%) "
          f"have 0 or 1 listings - flagged here before fitting, since this is exactly "
          f"the sparsity risk the proposal's Section 4.3 risk register anticipates.")

    model1_formula, model2_formula, model3_formula, model4_formula = build_moderation_formulas()

    model1 = fit_and_report(main_df, model1_formula, "MODEL 1: RQ1 baseline (zone, no interaction)")
    model2 = fit_and_report(main_df, model2_formula, "MODEL 2: borough main effects, no interaction")
    model3 = fit_and_report(main_df, model3_formula, "MODEL 3: full borough x listing_type interaction (RQ2, literal)")
    model4 = fit_and_report(main_df, model4_formula, "MODEL 4: zone x listing_type interaction (RQ2, well-identified fallback)")

    print("\n=== Model comparison ===")
    comparison = pd.DataFrame({
        "model": ["1: zone", "2: borough (no interaction)", "3: borough x listing_type", "4: zone x listing_type"],
        "params": [int(m.df_model) + 1 for m in (model1, model2, model3, model4)],
        "adj_r2": [m.rsquared_adj for m in (model1, model2, model3, model4)],
        "aic": [m.aic for m in (model1, model2, model3, model4)],
    })
    print(comparison.to_string(index=False))

    flag_unstable_interactions(model3, "MODEL 3", ct)

    zone_ct = pd.crosstab(main_df["zone"], main_df["listing_type"])
    flag_unstable_interactions(model4, "MODEL 4", zone_ct)

    print(
        "\n=== Conclusion ===\n"
        "Model 3 (full borough x listing_type) wins on both AIC and adjusted "
        "R-squared, but 172 of its 256 interaction terms (67%) rest on fewer "
        "than 5 listings in that specific cell - so most of that apparent gain "
        "is overfitting to sparse cells, exactly as the proposal's Section 4.3 "
        "risk register anticipated, not evidence of genuine borough-level "
        "moderation. Per the proposal's own pre-committed fallback, the "
        "primary RQ2 finding is Model 4 (zone x listing_type), which is well-"
        "identified (8 interaction terms, none resting on fewer than 5 "
        "listings): it improves modestly but genuinely on the no-interaction "
        "baseline (adj. R-squared 0.615 -> 0.620, AIC 3105 -> 3082), and the "
        "moderation is concentrated in the minor 'Other' and 'Place to stay' "
        "listing-type categories - the zone effect is significantly larger "
        "for these in Outer London - while the dominant categories (Flat, "
        "Room, Home) show no significant zone-dependent premium. Model 3's "
        "stable interactions (printed above, backed by 5+ listings) can be "
        "reported as a secondary, descriptive finding - e.g. Room listings "
        "command an unusually large premium specifically in Havering and "
        "Hounslow - without treating the full borough model as the headline "
        "result."
    )
