"""Feature set and leakage-safe preprocessing pipeline for the AVM.

Two layers, kept separate on purpose (IDX Best Practices §05–§07):
- add_features(): row-wise derived columns that need nothing learned from data
  (seasonality, HOA fee per month). Safe to run on any split.
- build_preprocessor(): a scikit-learn ColumnTransformer holding EVERY step that learns
  from data (imputation medians, missing-value flags, one-hot vocabularies, target encodings,
  scaling). Fit it on the training set only; transform() is all that touches validation/test.
  High-cardinality location columns use TargetEncoder, which cross-fits on the training set
  so a row's own price never feeds its own encoding.

Usage:
    X_train, y_train = make_xy(train)
    pre = build_preprocessor()
    Xt_train = pre.fit_transform(X_train, y_train)   # learns from train only
    Xt_val = pre.transform(make_xy(val)[0])
"""

import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, TargetEncoder

from cleaning import LEAKAGE_COLS

RANDOM_STATE = 42
TARGET = "log_close_price"

# Physical facts known before any sale; right-skewed sizes get a log transform.
LOG_NUMERIC = ["LivingArea", "LotSizeSquareFeet"]
NUMERIC = [
    "BedroomsTotal", "BathroomsTotalInteger", "property_age", "GarageSpaces", "ParkingTotal",
    "Stories", "MainLevelBedrooms", "hoa_monthly", "Latitude", "Longitude",
    "month_sin", "month_cos",
]
BOOLEAN = [
    "ViewYN", "PoolPrivateYN", "NewConstructionYN", "FireplaceYN", "AttachedGarageYN",
    "BasementYN", "WaterfrontYN",
]
LOW_CARDINALITY = ["CountyOrParish", "Levels"]
HIGH_CARDINALITY = ["PostalCode", "City", "MLSAreaMajor", "HighSchoolDistrict"]

FEATURES = LOG_NUMERIC + NUMERIC + BOOLEAN + LOW_CARDINALITY + HIGH_CARDINALITY

# Columns in the clean table that are deliberately NOT features, with the reason.
# Threshold for "too sparse": more than 60% missing in the training window.
MISSINGNESS_THRESHOLD = 0.60
EXCLUDED = {
    **{c: "leakage: list price / DOM / listing-contract-close dates (Best Practices §04)" for c in LEAKAGE_COLS},
    "ClosePrice": "target (modeled as log_close_price)",
    "log_close_price": "target",
    "ListingKey": "identifier",
    "source_file": "data lineage, not a property attribute",
    "close_month": "split key; seasonality enters through month_sin / month_cos",
    "YearBuilt": "replaced by property_age (age at time of sale)",
    "AssociationFee": "replaced by hoa_monthly (fee normalized by its frequency)",
    "AssociationFeeFrequency": "used only to build hoa_monthly",
    "coords_imputed": "artifact of which export file a row came from, unknown for most rows",
    "SubdivisionName": "too sparse (> 60% missing) and mostly placeholder values",
    "ElementarySchool": "too sparse (> 60% missing)",
    "MiddleOrJuniorSchool": "too sparse (> 60% missing)",
    "HighSchool": "too sparse (> 60% missing)",
    "BuilderName": "too sparse (> 60% missing)",
    "Flooring": "comma-separated list; planned as multi-hot features in Week 6",
}

FEE_PERIODS_PER_MONTH = {"Monthly": 1, "Quarterly": 3, "SemiAnnually": 6, "Annually": 12}


def add_features(df):
    """Row-wise features that learn nothing from data, so they are safe on any split."""
    out = df.copy()
    month = pd.to_datetime(out["CloseDate"]).dt.month
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    # Assumption: a fee with no stated frequency is monthly (the most common frequency).
    periods = out["AssociationFeeFrequency"].map(FEE_PERIODS_PER_MONTH).fillna(1)
    out["hoa_monthly"] = out["AssociationFee"] / periods
    return out


def make_xy(df):
    """Feature frame and target. Nullable pandas types become plain float/object for sklearn."""
    df = add_features(df)
    X = df[FEATURES].copy()
    for c in BOOLEAN:
        X[c] = X[c].astype("float")
    for c in LOW_CARDINALITY + HIGH_CARDINALITY:
        X[c] = X[c].astype("object").where(X[c].notna(), np.nan)
    y = df[TARGET].to_numpy() if TARGET in df else None
    return X, y


def build_preprocessor():
    """ColumnTransformer with every fit-time step. Fit on training data only."""
    numeric = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
    log_numeric = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
        StandardScaler(),
    )
    boolean = SimpleImputer(strategy="most_frequent", add_indicator=True)
    low_card = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OneHotEncoder(min_frequency=50, handle_unknown="infrequent_if_exist", sparse_output=False),
    )
    high_card = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        TargetEncoder(target_type="continuous", cv=KFold(5, shuffle=True, random_state=RANDOM_STATE)),
        StandardScaler(),
    )
    return ColumnTransformer(
        [
            ("log_num", log_numeric, LOG_NUMERIC),
            ("num", numeric, NUMERIC),
            ("bool", boolean, BOOLEAN),
            ("low_card", low_card, LOW_CARDINALITY),
            ("high_card", high_card, HIGH_CARDINALITY),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def leakage_audit(clean_columns):
    """One row per clean-table column: used as a feature or excluded, and why."""
    assert not set(FEATURES) & set(LEAKAGE_COLS), "leakage column listed as a feature"
    season = "month of sale only (at prediction time: the valuation month), no price information"
    derived = {
        "month_sin": season,
        "month_cos": season,
        "hoa_monthly": "AssociationFee divided by its payment frequency; HOA dues are known before a sale",
        "property_age": "close year minus YearBuilt; at prediction time: valuation year minus YearBuilt",
    }
    rows = []
    for c in list(clean_columns) + [d for d in derived if d not in clean_columns]:
        if c in FEATURES:
            rows.append({"column": c, "verdict": "feature", "reason": derived.get(c, "physical or location fact known before any sale")})
        elif c in EXCLUDED:
            verdict = "excluded: leakage" if c in LEAKAGE_COLS else "excluded"
            rows.append({"column": c, "verdict": verdict, "reason": EXCLUDED[c]})
        else:
            rows.append({"column": c, "verdict": "UNREVIEWED", "reason": ""})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from preprocessing import load_clean

    audit = leakage_audit(load_clean().columns)
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "leakage_audit.csv")
    audit.to_csv(path, index=False)
    print(audit.to_string(index=False))
