"""Clean CRMLS sold listings into the AVM modeling table (CA single-family homes).

Usage (from the repo root):
    python src/cleaning.py

Writes:
    data/processed/sfr_clean.csv      - one row per sale, git-ignored
    data/processed/sfr_clean.parquet  - same table with dtypes preserved (faster to load)
    docs/cleaning_log.csv             - rows removed / values nulled per step (counts only)

Every rule lives in RULES so thresholds can be changed in one place.
See notebooks/02_preprocessing.ipynb for the reasoning behind each one.

This step removes rows that are clearly wrong (typos, impossible values) using fixed
thresholds on all data. The 0.5th/99.5th percentile ClosePrice cut is a separate step
applied at the train/test split, with cutoffs computed on the training set only.
"""

import glob
import os

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
OUT_PATH = os.path.join(DATA_DIR, "processed", "sfr_clean.parquet")
CSV_PATH = os.path.join(DATA_DIR, "processed", "sfr_clean.csv")
LOG_PATH = os.path.join(REPO_ROOT, "docs", "cleaning_log.csv")

RULES = {
    "min_close_price": 10_000,
    # Close/list ratio outside this band is almost always a typo (extra/missing zero)
    "close_to_list_ratio": (0.5, 2.0),
    "living_area_sqft": (300, 20_000),
    "price_per_sqft": (50, 5_000),
    "year_built_min": 1850,
    "max_lot_acres": 1_000,
    "max_bed_bath": 20,
    "max_garage_spaces": 20,
    "max_parking_total": 50,
    "max_days_on_market": 3_650,
    # Rough California bounding box
    "lat_range": (32.3, 42.1),
    "lon_range": (-124.5, -114.0),
}

DATE_COLS = ["CloseDate", "ListingContractDate", "PurchaseContractDate", "ContractStatusChangeDate"]

# Dropped from the processed table:
# - PII / agent & office identity (not property features, and buyer side is only known after the sale)
# - IDs other than ListingKey
# - Constant after filtering, or 100% / near-100% empty for SFR
DROP_COLS = [
    "ListAgentEmail", "ListAgentFirstName", "ListAgentLastName", "ListAgentFullName",
    "CoListAgentFirstName", "CoListAgentLastName", "BuyerAgentFirstName", "BuyerAgentLastName",
    "CoBuyerAgentFirstName", "BuyerAgentMlsId", "UnparsedAddress", "StreetNumberNumeric",
    "ListOfficeName", "CoListOfficeName", "BuyerOfficeName", "ListAgentAOR", "BuyerAgentAOR",
    "BuyerOfficeAOR", "BuyerAgencyCompensation", "BuyerAgencyCompensationType",
    "ListingId", "ListingKeyNumeric",
    "MlsStatus", "StateOrProvince", "PropertyType", "PropertySubType",
    "OriginatingSystemName", "OriginatingSystemSubName", "BusinessType",
    "TaxAnnualAmount", "TaxYear", "FireplacesTotal", "CoveredSpaces", "AboveGradeFinishedArea",
    "ElementarySchoolDistrict", "MiddleOrJuniorSchoolDistrict",
    "BelowGradeFinishedArea", "BuildingAreaTotal", "LotSizeDimensions", "LotSizeArea",
    "latfilled", "lonfilled",
]

# Kept in the clean table (needed for data-quality checks) but never used as model features.
# Per the IDX Best Practices doc: list prices and DOM leak the answer, and these dates are
# only known once the home is listed / under contract / closed.
LEAKAGE_COLS = [
    "ListPrice", "OriginalListPrice", "DaysOnMarket",
    "ListingContractDate", "PurchaseContractDate", "ContractStatusChangeDate", "CloseDate",
]

# Only "True" is ever recorded for these, so missing means "not flagged"
TRUE_ONLY_FLAGS = ["BasementYN", "WaterfrontYN"]


def load_raw(data_dir=DATA_DIR):
    frames = []
    for f in sorted(glob.glob(os.path.join(data_dir, "CRMLSSold*.csv"))):
        tmp = pd.read_csv(f, low_memory=False)
        tmp["source_file"] = os.path.basename(f)
        frames.append(tmp)
    return pd.concat(frames, ignore_index=True)


class _Log:
    """Records what each step did so the pipeline is auditable."""

    def __init__(self, n_start):
        self.rows = [{"step": "start", "rows_removed": 0, "values_nulled": 0, "rows_after": n_start}]

    def drop(self, df, mask, step):
        out = df[~mask]
        self.rows.append({"step": step, "rows_removed": int(mask.sum()), "values_nulled": 0, "rows_after": len(out)})
        return out

    def null(self, df, col, mask, step):
        mask = mask & df[col].notna()
        df.loc[mask, col] = np.nan
        self.rows.append({"step": step, "rows_removed": 0, "values_nulled": int(mask.sum()), "rows_after": len(df)})

    def frame(self):
        cols = ["step", "rows_removed", "values_nulled", "values_filled", "rows_after"]
        out = pd.DataFrame(self.rows).reindex(columns=cols)
        out[cols[1:]] = out[cols[1:]].fillna(0).astype(int)
        return out


def clean(raw, rules=RULES):
    """Return (clean_df, log_df). Does not modify `raw`."""
    df = raw.copy()
    log = _Log(len(df))

    for c in DATE_COLS:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    # 1. Modeling population
    df = log.drop(df, ~((df["PropertyType"] == "Residential") & (df["PropertySubType"] == "SingleFamilyResidence")),
                  "not Residential / SingleFamilyResidence")
    df = log.drop(df, df["StateOrProvince"] != "CA", "state not CA")

    # 2. Duplicates: same listing re-recorded (usually a corrected CloseDate).
    #    Keep the most recently updated record.
    # na_position="first" so a record with no update date never counts as the latest
    df = df.sort_values(["ListingKey", "ContractStatusChangeDate", "CloseDate", "source_file"], na_position="first")
    df = log.drop(df, df.duplicated("ListingKey", keep="last"), "duplicate ListingKey (kept latest update)")

    # 3. Target sanity
    df = log.drop(df, df["ClosePrice"].isna() | (df["ClosePrice"] < rules["min_close_price"]),
                  f"ClosePrice missing or < ${rules['min_close_price']:,}")
    ratio = df["ClosePrice"] / df["ListPrice"].where(df["ListPrice"] > 0)
    lo, hi = rules["close_to_list_ratio"]
    df = log.drop(df, ratio.isna() | (ratio < lo) | (ratio > hi), f"close/list ratio outside [{lo}, {hi}] or no ListPrice")

    # 4. Size (key feature and needed for the price-per-sqft check)
    lo, hi = rules["living_area_sqft"]
    df = log.drop(df, df["LivingArea"].isna() | (df["LivingArea"] < lo) | (df["LivingArea"] > hi),
                  f"LivingArea missing or outside [{lo:,}, {hi:,}]")
    ppsf = df["ClosePrice"] / df["LivingArea"]
    lo, hi = rules["price_per_sqft"]
    df = log.drop(df, (ppsf < lo) | (ppsf > hi), f"price per sqft outside [${lo}, ${hi:,}]")

    df = df.copy()

    # 5. Implausible feature values -> missing (keep the sale, let the model impute)
    close_year = df["CloseDate"].dt.year
    log.null(df, "YearBuilt", (df["YearBuilt"] < rules["year_built_min"]) | (df["YearBuilt"] > close_year + 1),
             f"YearBuilt < {rules['year_built_min']} or after close year + 1")
    for c in ["BedroomsTotal", "BathroomsTotalInteger"]:
        log.null(df, c, (df[c] <= 0) | (df[c] > rules["max_bed_bath"]), f"{c} <= 0 or > {rules['max_bed_bath']}")
    log.null(df, "GarageSpaces", (df["GarageSpaces"] < 0) | (df["GarageSpaces"] > rules["max_garage_spaces"]),
             f"GarageSpaces < 0 or > {rules['max_garage_spaces']}")
    log.null(df, "ParkingTotal", (df["ParkingTotal"] < 0) | (df["ParkingTotal"] > rules["max_parking_total"]),
             f"ParkingTotal < 0 or > {rules['max_parking_total']}")
    log.null(df, "DaysOnMarket", (df["DaysOnMarket"] < 0) | (df["DaysOnMarket"] > rules["max_days_on_market"]),
             f"DaysOnMarket < 0 or > {rules['max_days_on_market']:,}")
    log.null(df, "AssociationFee", df["AssociationFee"] < 0, "AssociationFee < 0")

    # Lot size: fill sqft from acres, then null zero / absurd lots. Keep one column.
    lot_missing = df["LotSizeSquareFeet"].isna() & df["LotSizeAcres"].notna()
    df.loc[lot_missing, "LotSizeSquareFeet"] = df.loc[lot_missing, "LotSizeAcres"] * 43_560
    log.rows.append({"step": "LotSizeSquareFeet filled from LotSizeAcres", "rows_removed": 0,
                     "values_nulled": 0, "rows_after": len(df), "values_filled": int(lot_missing.sum())})
    log.null(df, "LotSizeSquareFeet",
             (df["LotSizeSquareFeet"] <= 0) | (df["LotSizeSquareFeet"] > rules["max_lot_acres"] * 43_560),
             f"LotSizeSquareFeet <= 0 or > {rules['max_lot_acres']:,} acres")
    df = df.drop(columns="LotSizeAcres")

    # Coordinates outside California (includes 0,0 and sign errors)
    (lat_lo, lat_hi), (lon_lo, lon_hi) = rules["lat_range"], rules["lon_range"]
    bad_xy = ~(df["Latitude"].between(lat_lo, lat_hi) & df["Longitude"].between(lon_lo, lon_hi))
    log.null(df, "Latitude", bad_xy, "Latitude/Longitude outside California")
    df.loc[bad_xy, "Longitude"] = np.nan

    # 6. Types and flags
    # latfilled/lonfilled are True/False flags (coordinate was imputed), only present in *_filled files
    imputed = df["latfilled"].astype("string").str.lower().map({"true": True, "false": False})
    df["coords_imputed"] = imputed.astype("boolean")
    for c in TRUE_ONLY_FLAGS:
        df[c] = df[c].fillna(False)
    for c in [c for c in df.columns if c.endswith("YN")]:
        df[c] = df[c].astype("boolean")

    # 7. Derived columns
    df["close_month"] = df["CloseDate"].dt.to_period("M").astype(str)
    df["log_close_price"] = np.log(df["ClosePrice"])
    df["property_age"] = close_year - df["YearBuilt"]

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = df.sort_values(["CloseDate", "ListingKey"]).reset_index(drop=True)
    log.rows.append({"step": "final", "rows_removed": 0, "values_nulled": 0, "rows_after": len(df)})
    return df, log.frame()


def main():
    raw = load_raw()
    df, log = clean(raw)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    df.to_csv(CSV_PATH, index=False)
    log.to_csv(LOG_PATH, index=False)
    print(log.to_string(index=False))
    print(f"\n{len(df):,} rows x {df.shape[1]} columns -> {os.path.relpath(CSV_PATH, REPO_ROOT)} (+ .parquet)")


if __name__ == "__main__":
    main()
