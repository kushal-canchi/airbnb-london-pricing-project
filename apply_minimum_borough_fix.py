"""
apply_minimum_borough_fix.py
------------------------------------------------------------------------------------
Applies the minimum acceptable fix (per supervisor feedback, round 2) to an
already-collected Airbnb London dataset, without re-scraping anything:

  1. Renames search_location -> search_borough, so the column name is
     honest about what it measures (the query, not a verified per-listing
     location).
  2. Screens for out-of-Greater-London listings using the LISTING TITLE,
     not search_borough. search_borough only ever holds the 33 borough
     query strings this scraper supplies (see FULL_LOCATIONS in the
     scraper), so a row-level check against that column can never fail
     and drops nothing - this was flagged in supervisor feedback as the
     bug in the previous version of this script.
     Airbnb listing titles follow a "<type> in <place>" pattern (e.g.
     "Room in Croydon", "Cabin in Esher"). The trailing place name is
     extracted and checked against NON_LONDON_PLACES, a curated list of
     Home Counties towns/villages and county names that fall outside the
     Greater London boundary but get pulled into borough-radius searches
     near the edge of London (concentrated under Hillingdon, Havering,
     Harrow, Sutton, Kingston, Croydon, Hounslow, Enfield, Redbridge -
     i.e. the boroughs that actually border Surrey/Essex/Herts/Bucks/
     Berks/Kent). Matches are dropped.
     Hotel-brand listings (is_hotel_brand == True) don't follow the
     "<type> in <place>" title pattern, so no location can be extracted
     for them. None of their titles reference any non-London county name,
     so they are kept as-is rather than dropped for having no extractable
     location.
     A trailing place name that doesn't parse, or isn't unambiguously
     identifiable as London or non-London, is dropped as unverifiable
     rather than assumed valid.
  3. Prints a before/after summary, broken down by search_borough and by
     the specific place name that caused a drop, so the fix is documented
     and reproducible, not silent.

USAGE
    python apply_minimum_borough_fix.py

Edit INPUT_FILE below to point at your current dataset (the checkpoint
file or your deduplicated final export). Adjust ID_COLUMN only if your
identifier column isn't named room_id.
"""

import os
import re
import pandas as pd

# ------------------------------------------------------------------
# CONFIG - edit these two lines to match your files
# ------------------------------------------------------------------
INPUT_FILE = "airbnb_london_full33_pricebands_20260804_1347.csv"
ID_COLUMN = "room_id"

# Regex for the trailing place name in an Airbnb listing title, e.g.
# "Room in Croydon" -> "Croydon", "Home in Northwood, London" ->
# "Northwood, London". Requires a leading space before "in" so it can't
# false-match inside a word ending in "in" (e.g. "Cabin in Hackney").
TITLE_LOCATION_RE = re.compile(r" in ([^,]+(?:, [^,]+)?)$")

# Place names / counties that appear in listing titles but fall outside
# the Greater London boundary. Curated by cross-checking every unique
# trailing title location in the 04/08 dataset against Greater London's
# 33-borough boundary. Matched case-insensitively, exact match on the
# extracted trailing location (not substring), to avoid accidentally
# rejecting a London place whose name happens to contain one of these
# as a substring.
NON_LONDON_PLACES = {
    # Explicit county / unitary authority names
    "surrey", "essex", "hertfordshire", "berkshire", "buckinghamshire",
    "kent", "thurrock", "bracknell forest",
    # Surrey towns/villages (outside Greater London)
    "addlestone", "banstead", "chipstead", "chobham", "dorking",
    "east molesey", "esher", "ewell", "fetcham", "godstone",
    "kingswood", "lower kingswood", "merstham", "staines-upon-thames",
    "stoke d'abernon", "sunbury-on-thames", "warlingham", "woldingham",
    # Hertfordshire towns/villages
    "cheshunt", "chorleywood", "elstree", "potters bar",
    # Buckinghamshire towns/villages
    "beaconsfield", "chalfont saint peter", "denham", "farnham common",
    "gerrards cross", "hedgerley", "middle green", "stoke poges",
    # Berkshire towns/villages
    "colnbrook", "dorney", "englefield green", "langley", "old windsor",
    "slough", "taplow", "windsor", "windsor and maidenhead", "wraysbury",
    # Essex / Thurrock towns/villages
    "aveley", "buckhurst hill", "bulphan", "chafford hundred", "chigwell",
    "dunton", "fyfield", "great warley", "loughton", "margaretting",
    "moreton", "navestock", "shenfield", "southend-on-sea",
    "stapleford abbotts", "stapleford tawney", "theydon bois", "warley",
    # Kent towns/villages
    "cliffe", "medway", "meopham south",
}

# Trailing title locations that cannot be verified either way (too
# generic / not a real place name) and are dropped as unverifiable
# rather than assumed to be inside London.
UNVERIFIABLE_PLACES = {"uk"}

# The 33 official London boroughs, exactly as queried by the scraper -
# kept for the search_borough validity check (step 2 of the ORIGINAL
# fix). Still useful as a sanity check even though the real screen now
# runs on title, not this column.
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
_VALID_BOROUGH_LOWER = {b.strip().lower() for b in VALID_LONDON_BOROUGHS}


def load_dataset(path):
    """Loads the dataset, keeping room_id (or ID_COLUMN) as a string so
    long numeric identifiers are never silently rounded by pandas."""
    ext = os.path.splitext(path)[1].lower()
    dtype = {ID_COLUMN: str} if ID_COLUMN else None
    if ext == ".csv":
        return pd.read_csv(path, dtype=dtype)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(path, dtype=dtype)
    raise ValueError(f"Unsupported file type: {ext}")


def extract_title_location(title):
    """Pulls the trailing '<type> in <place>' location out of a listing
    title. Returns None if the title doesn't follow that pattern (this
    is expected and fine for hotel-brand listings)."""
    if not isinstance(title, str):
        return None
    m = TITLE_LOCATION_RE.search(title)
    return m.group(1).strip() if m else None


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

    if "search_borough" in df.columns:
        bad_query = ~df["search_borough"].astype(str).str.strip().str.lower().isin(_VALID_BOROUGH_LOWER)
        if bad_query.any():
            print(f"  NOTE: {bad_query.sum()} row(s) have a search_borough value outside "
                  f"the 33 queried boroughs - unexpected, worth a look.")

    before = len(df)

    # Step 2: screen using the LISTING TITLE, not search_borough (see
    # module docstring for why search_borough can't do this job).
    df["_title_location"] = df["title"].apply(extract_title_location)

    is_hotel_brand = df.get("is_hotel_brand", pd.Series(False, index=df.index)).astype(str).str.lower().eq("true")
    loc_lower = df["_title_location"].astype(str).str.strip().str.lower()

    is_non_london = loc_lower.isin(NON_LONDON_PLACES)
    is_unverifiable = loc_lower.isin(UNVERIFIABLE_PLACES)
    # No location could be extracted AND it isn't a hotel-brand listing
    # (hotel-brand titles legitimately don't follow the "in <place>"
    # pattern; anything else with no extractable location is suspect).
    no_location_non_hotel = df["_title_location"].isna() & ~is_hotel_brand

    to_drop = is_non_london | is_unverifiable | no_location_non_hotel

    dropped_by_place = df.loc[is_non_london | is_unverifiable, "_title_location"].value_counts()
    dropped_by_borough = df.loc[to_drop, "search_borough"].value_counts() if "search_borough" in df.columns else None

    df = df.loc[~to_drop].drop(columns=["_title_location"]).reset_index(drop=True)
    after = len(df)

    print(f"\nRows before: {before}")
    print(f"Rows after:  {after}")
    print(f"Rows dropped: {before - after} ({100 * (before - after) / before:.1f}%)")
    if not dropped_by_place.empty:
        print("\nDropped rows by title location (out-of-Greater-London / unverifiable):")
        print(dropped_by_place.to_string())
    if dropped_by_borough is not None and not dropped_by_borough.empty:
        print("\nDropped rows by search_borough (i.e. which borough's radius pulled them in):")
        print(dropped_by_borough.to_string())
    if no_location_non_hotel.any():
        print(f"\n{no_location_non_hotel.sum()} row(s) dropped for having no extractable "
              f"title location and not being a hotel-brand listing - worth a manual look.")

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
        "dataset_limitations_note.md)."
    )
