"""Value one home with the final model (used by app.py; game plan Week 9).

Loads the final model chosen in Week 8 (docs/metrics_summary.csv) together with its fitted
preprocessing, so a home goes through exactly the steps used in training. Fills in what a user
should not have to type:
- city, county, MLS area and high-school district from the ZIP code (most common value among
  past sales in that ZIP), and the ZIP's median coordinates when no coordinates are given
- school districts from the coordinates (geo.lookup)
- age at valuation, HOA fee per month and seasonality (features.add_features)

Before predicting it checks the feature schema against the one the model was trained on
(IDX Best Practices: assert the feature schema) and warns when inputs fall outside the
training data. The price range comes from docs/prediction_intervals.csv (Week 8).

Usage:
    from predict import Valuer
    v = Valuer()
    v.value({"PostalCode": "92618", "LivingArea": 2100, "BedroomsTotal": 4, ...})
"""

import os
from datetime import date

import joblib
import numpy as np
import pandas as pd

import cleaning
import features
import geo
import preprocessing

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(REPO_ROOT, "docs")
MODEL_DIR = os.path.join(REPO_ROOT, "models")
ZIP_FIELDS = ["City", "CountyOrParish", "MLSAreaMajor", "HighSchoolDistrict"]
MIN_ZIP_SALES = 5  # ZIPs with fewer past sales are treated as unknown

# Every input the model uses besides the auto-filled ones. None = unknown (the model imputes it).
INPUTS = [
    "PostalCode", "Latitude", "Longitude", "LivingArea", "LotSizeSquareFeet", "BedroomsTotal",
    "BathroomsTotalInteger", "YearBuilt", "GarageSpaces", "ParkingTotal", "Stories", "MainLevelBedrooms",
    "AssociationFee", "AssociationFeeFrequency", "Levels", "Flooring",
    "ViewYN", "PoolPrivateYN", "NewConstructionYN", "FireplaceYN", "AttachedGarageYN", "BasementYN", "WaterfrontYN",
    *ZIP_FIELDS,
]


def final_model_stem():
    """Artifact name of the final model picked in Week 8 (lowest validation MdAPE)."""
    summary = pd.read_csv(os.path.join(DOCS_DIR, "metrics_summary.csv"))
    name = summary.loc[summary["final_model"], "model"].iloc[0]
    import evaluate_final  # local import: pulls in matplotlib, only needed here
    return evaluate_final.stem(name), evaluate_final.LABELS[name]


def build_zip_lookup(clean):
    """Per ZIP: most common city / county / MLS area / high-school district, median coordinates."""
    g = clean.dropna(subset=["PostalCode"]).groupby("PostalCode")
    out = g[ZIP_FIELDS].agg(lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)
    out[["Latitude", "Longitude"]] = g[["Latitude", "Longitude"]].median()
    out["sales"] = g.size()
    return out[out["sales"] >= MIN_ZIP_SALES]


class Valuer:
    def __init__(self, clean=None, districts=None):
        stem, self.model_label = final_model_stem()
        art = joblib.load(os.path.join(MODEL_DIR, f"{stem}.joblib"))
        self.model, self.feature_set, self.train_info = art["model"], art["feature_set"], art["train_info"]
        self.intervals = pd.read_csv(os.path.join(DOCS_DIR, "prediction_intervals.csv"))
        clean = preprocessing.load_clean() if clean is None else clean
        self.zips = build_zip_lookup(clean)
        self.districts = geo.load_districts() if districts is None else districts
        self.levels = sorted(clean["Levels"].dropna().value_counts().loc[lambda s: s >= 50].index)

    def complete(self, home, valuation_date=None, fill_from_zip=True):
        """One-row clean-table frame for `home`, with ZIP, coordinate and district fields filled in."""
        valuation_date = pd.Timestamp(valuation_date or date.today())
        row = {k: home.get(k) for k in INPUTS}
        z = str(row["PostalCode"] or "").strip()
        row["PostalCode"] = z or None
        filled = []
        if fill_from_zip and z in self.zips.index:
            for k in ZIP_FIELDS + ["Latitude", "Longitude"]:
                if row[k] is None or (isinstance(row[k], float) and np.isnan(row[k])):
                    row[k] = self.zips.at[z, k]
                    filled.append(k)
        if isinstance(row["Flooring"], (list, tuple, set)):
            row["Flooring"] = ",".join(row["Flooring"]) or None
        row["CloseDate"] = valuation_date
        row["property_age"] = valuation_date.year - row["YearBuilt"] if row["YearBuilt"] else None
        row["ListingKey"] = "app"
        if row["Latitude"] is not None and row["Longitude"] is not None:
            row.update(geo.lookup(row["Latitude"], row["Longitude"], self.districts))
        df = pd.DataFrame([row])
        for c in ["ViewYN", "PoolPrivateYN", "NewConstructionYN", "FireplaceYN", "AttachedGarageYN", "BasementYN", "WaterfrontYN"]:
            df[c] = df[c].astype("boolean")
        return df, filled

    def check(self, df):
        """Warnings for inputs outside the training data."""
        r, warn = df.iloc[0], []
        lo, hi = cleaning.RULES["living_area_sqft"]
        if r["LivingArea"] is None or not (lo <= r["LivingArea"] <= hi):
            warn.append(f"Living area should be between {lo:,} and {hi:,} sq ft (the range the model was trained on).")
        if r["PostalCode"] not in self.zips.index:
            warn.append("This ZIP code has fewer than 5 past sales in the data, so its location effect is a statewide average.")
        (lat_lo, lat_hi), (lon_lo, lon_hi) = cleaning.RULES["lat_range"], cleaning.RULES["lon_range"]
        if r["Latitude"] is None or not (lat_lo <= r["Latitude"] <= lat_hi and lon_lo <= r["Longitude"] <= lon_hi):
            warn.append("Coordinates are missing or outside California; location relies on the ZIP code only.")
        elif pd.isna(r.get("school_district_k8")):
            warn.append("No school district found at these coordinates.")
        if r["YearBuilt"] and not (cleaning.RULES["year_built_min"] <= r["YearBuilt"] <= pd.Timestamp(r["CloseDate"]).year + 1):
            warn.append("Year built looks implausible.")
        return warn

    def value(self, home, valuation_date=None, fill_from_zip=True):
        """Predicted price, 80% / 90% ranges, filled-in fields and warnings for one home."""
        df, filled = self.complete(home, valuation_date, fill_from_zip)
        X, _ = features.make_xy(df)
        expected = list(self.model.named_steps["preprocess"].feature_names_in_)
        assert list(X.columns) == expected, f"feature schema mismatch: {sorted(set(X.columns) ^ set(expected))}"
        price = float(np.exp(self.model.predict(X))[0])
        band = self.intervals[(self.intervals["predicted_low"] <= price) & (price < self.intervals["predicted_high"])].iloc[0]
        warnings = self.check(df)
        if not (self.train_info["price_low"] <= price <= self.train_info["price_high"]):
            warnings.append("The estimate is outside the price range of the training sales; treat it with caution.")
        return {
            "price": price,
            "range_80": (price * (1 + band["low_80"]), price * (1 + band["high_80"])),
            "range_90": (price * (1 + band["low_90"]), price * (1 + band["high_90"])),
            "band": band["band"],
            "filled": {k: df.iloc[0][k] for k in filled},
            "districts": {k: df.iloc[0].get(k) for k in ["school_district_k8", "school_district_hs", "district_type"]},
            "warnings": warnings,
            "model": self.model_label,
            "trained_on": f"{self.train_info['train_start']} to {self.train_info['train_end']}",
        }
