"""
apply_minimum_borough_fix.py
------------------------------------------------------------------------------------
Applies the MINIMUM acceptable fix (per supervisor feedback) to an already-
collected Airbnb London dataset, without re-scraping anything:

  1. Renames the column recording which borough was searched from
     search_location to search_borough, so the column name itself is
     honest about what it measures (the query, not a verified per-listing
     location).
  2. Drops any row whose search_borough is not one of the 33 official
     London boroughs (i.e. is None/blank, or names somewhere outside
     Greater London - Surrey, Essex, Hertfordshire, Berkshire, Windsor,
     etc.). These rows cannot be trusted as "London" observations for a
     London hedonic pricing model.
  3. Prints a before/after summary so the drop is documented and
     reproducible, not silent.

This does NOT fix the deeper issue that deduplication credits an
overlapping listing to whichever borough sorts first alphabetically
(see KNOWN LIMITATIONS in the scraper's docstring) - that requires the
full second-pass fix (per-listing coordinates + point-in-polygon against
an ONS boundary file), which was deliberately out of scope for this
minimum fix given the time available. State this plainly as a limitation
in the methodology chapter; the paragraph in borough_limitation_note.md
(generated alongside this script) is ready to paste in.

USAGE
    python apply_minimum_borough_fix.py

Edit INPUT_FILE below to point at your current dataset (the checkpoint
file or your deduplicated final export). Adjust ID_COLUMN only if your
identifier column isn't named room_id.
"""

import os
import pandas as pd

# ------------------------------------------------------------------
# CONFIG - edit these two lines to match your files
# ------------------------------------------------------------------
INPUT_FILE = "checkpoint_listings.csv"   # or your final deduplicated CSV/XLSX
ID_COLUMN = "room_id"

# The 33 official London boroughs, exactly as queried by the scraper.
# A row is kept only if its search_borough / search_location value is
# in this list (falls back to a same-length case-insensitive match).
VALID_LONDON_BOROUGHS = [
    "Barking and Dagenham, London, United Kingdom", "Barnet, London, United Kingdom",
    "Bexley, London, United Kingdom", "Brent, London, United Kingdom",
    "Bromley, London, United Kingdom", "Camden, London, United Kingdom",
    "Croydon, London, United Kingdom", "Ealing, London, United Kingdom",
    "Enfield, London, United Kingdom", "Greenwich, London, United Kingdom",
    "Hackney, London, United Kingdom", "Hammersmith and Fulham, London, United Kingdom",
    "Haringey, London, United Kingdom", "Harrow, London, United Kingdom",
    "Havering, London, United Kingdom", "Hillingdon, London, United Kingdom",
    "Hounslow, London, United Kingdom", "Islington, London, United Kingdom",
    "Kensington and Chelsea, London, United Kingdom", "Kingston upon Thames, London, United Kingdom",
    "Lambeth, London, United Kingdom", "Lewisham, London, United Kingdom",
    "Merton, London, United Kingdom", "Newham, London, United Kingdom",
    "Redbridge, London, United Kingdom", "Richmond upon Thames, London, United Kingdom",
    "Southwark, London, United Kingdom", "Sutton, London, United Kingdom",
    "Tower Hamlets, London, United Kingdom", "Waltham Forest, London, United Kingdom",
    "Wandsworth, London, United Kingdom", "Westminster, London, United Kingdom",
    "City of London, United Kingdom",
]
_VALID_LOWER = {b.strip().lower() for b in VALID_LONDON_BOROUGHS}


def load_dataset(path):
    """Loads the dataset, keeping room_id (or ID_COLUMN) as a string so
    long numeric identifiers are never silently rounded by pandas."""
    ext = os.path.splitext(path)[1].lower()
    dtype = {ID_COLUMN: str} if ID_COLUMN else None
    if ext == ".csv":
        return pd.read_csv(path, dtype=dtype)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path, dtype=dtype)
        return df
    raise ValueError(f"Unsupported file type: {ext}")


def apply_minimum_borough_fix(df):
    # Step 1: rename to an honest column name, whichever the source used.
    if "search_location" in df.columns and "search_borough" not in df.columns:
        df = df.rename(columns={"search_location": "search_borough"})
        print("Renamed column: search_location -> search_borough")
    elif "search_borough" in df.columns:
        print("Column already named search_borough - no rename needed.")
    else:
        raise KeyError(
            "Neither 'search_location' nor 'search_borough' found in this "
            "file's columns. Check INPUT_FILE points at the right dataset."
        )

    before = len(df)

    # Step 2: keep only rows whose queried borough is one of the 33
    # official London boroughs. Blank/missing values and anything naming
    # a place outside Greater London (Surrey, Essex, Hertfordshire,
    # Berkshire, Windsor, etc.) are dropped here.
    is_valid = df["search_borough"].astype(str).str.strip().str.lower().isin(_VALID_LOWER)
    dropped = df.loc[~is_valid, "search_borough"].value_counts()
    df = df.loc[is_valid].reset_index(drop=True)
    after = len(df)

    print(f"\nRows before: {before}")
    print(f"Rows after:  {after}")
    print(f"Rows dropped: {before - after}")
    if not dropped.empty:
        print("\nDropped rows by search_borough value (out-of-London / blank):")
        print(dropped.to_string())
    else:
        print("\nNo out-of-London or blank search_borough rows found.")

    return df


if __name__ == "__main__":
    print(f"Loading {INPUT_FILE} ...")
    df = load_dataset(INPUT_FILE)

    df_fixed = apply_minimum_borough_fix(df)

    base, ext = os.path.splitext(INPUT_FILE)
    out_csv = f"{base}_borough_fixed.csv"
    df_fixed.to_csv(out_csv, index=False)
    print(f"\nSaved cleaned dataset to {out_csv}")
    print(
        "\nRemember: this is the MINIMUM fix - it makes the column name "
        "honest and removes rows that cannot be trusted as London "
        "observations, but it does NOT correct the alphabetical-"
        "deduplication undercount of Newham / Tower Hamlets / City of "
        "London. State that plainly as a limitation (see "
        "borough_limitation_note.md)."
    )
