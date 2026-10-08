"""Feature set and leakage-safe preprocessing pipeline for the AVM.

Two layers, kept separate on purpose (IDX Best Practices §05–§07):
- add_features(): row-wise derived columns that need nothing learned from data
  (seasonality, HOA fee per month, bed/bath ratio, school districts from a spatial join).
  Safe to run on any split.
- build_preprocessor(): a scikit-learn ColumnTransformer holding EVERY step that learns
  from data (imputation medians, missing-value flags, one-hot vocabularies, target encodings,
  scaling). Fit it on the training set only; transform() is all that touches validation/test.
  High-cardinality location columns use TargetEncoder, which cross-fits on the training set
  so a row's own price never feeds its own encoding.

Feature sets (game plan Week 6 compares old vs new):
    "week5"  - the Week 3-5 feature set
    "week6"  - week5 + flooring types + school-district layer (default; chosen on validation)
    plus one-addition sets ("week5+ratios", "week5+flooring", "week5+school_districts")
    and "week6+ratios" (all three additions)
make_xy() returns every candidate column; each preprocessor only uses its own set's columns.

Usage:
    X_train, y_train = make_xy(train)
    pre = build_preprocessor("week6")
    Xt_train = pre.fit_transform(X_train, y_train)   # learns from train only
    Xt_val = pre.transform(make_xy(val)[0])
"""

import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, TargetEncoder

from cleaning import LEAKAGE_COLS

RANDOM_STATE = 42
TARGET = "log_close_price"

# Week 3-5 feature set. Physical facts known before any sale; skewed sizes get a log transform.
WEEK5 = {
    "log_numeric": ["LivingArea", "LotSizeSquareFeet"],
    "numeric": [
        "BedroomsTotal", "BathroomsTotalInteger", "property_age", "GarageSpaces", "ParkingTotal",
        "Stories", "MainLevelBedrooms", "hoa_monthly", "Latitude", "Longitude",
        "month_sin", "month_cos",
    ],
    "boolean": [
        "ViewYN", "PoolPrivateYN", "NewConstructionYN", "FireplaceYN", "AttachedGarageYN",
        "BasementYN", "WaterfrontYN",
    ],
    "low_cardinality": ["CountyOrParish", "Levels"],
    "high_cardinality": ["PostalCode", "City", "MLSAreaMajor", "HighSchoolDistrict"],
    "multi_hot": [],
}

# Week 6 additions, by family
ADDITIONS = {
    "ratios": {"numeric": ["bed_bath_ratio", "sqft_per_bedroom"]},
    "flooring": {"multi_hot": ["Flooring"]},
    "school_districts": {
        "log_numeric": ["district_k8_enrollment"],
        "low_cardinality": ["district_type", "district_k8_assistance"],
        "high_cardinality": ["school_district_k8", "school_district_hs"],
    },
}


def _combine(base, *additions):
    out = {k: list(v) for k, v in base.items()}
    for add in additions:
        for group, cols in add.items():
            out[group] += cols
    return out


FEATURE_SETS = {
    "week5": WEEK5,
    **{f"week5+{name}": _combine(WEEK5, add) for name, add in ADDITIONS.items()},
    # Chosen on the validation month (docs/feature_set_comparison.csv): ratios hurt both tree
    # models, so the final Week 6 set is flooring + school districts.
    "week6": _combine(WEEK5, ADDITIONS["flooring"], ADDITIONS["school_districts"]),
    "week6+ratios": _combine(WEEK5, *ADDITIONS.values()),
}
DEFAULT_FEATURE_SET = "week6"

# Columns of the default set, by group (kept as module constants for notebooks and the app)
LOG_NUMERIC, NUMERIC, BOOLEAN, LOW_CARDINALITY, HIGH_CARDINALITY, MULTI_HOT = (
    FEATURE_SETS[DEFAULT_FEATURE_SET][g]
    for g in ["log_numeric", "numeric", "boolean", "low_cardinality", "high_cardinality", "multi_hot"])
FEATURES = LOG_NUMERIC + NUMERIC + BOOLEAN + LOW_CARDINALITY + HIGH_CARDINALITY + MULTI_HOT
ALL_COLUMNS = sorted({c for fs in FEATURE_SETS.values() for cols in fs.values() for c in cols})

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
    "bed_bath_ratio": "tested in Week 6; worsened both tree models on validation (docs/feature_set_comparison.csv)",
    "sqft_per_bedroom": "tested in Week 6; worsened both tree models on validation (docs/feature_set_comparison.csv)",
}

FEE_PERIODS_PER_MONTH = {"Monthly": 1, "Quarterly": 3, "SemiAnnually": 6, "Annually": 12}


SCHOOL_DISTRICTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     "data", "processed", "school_districts.parquet")
_district_cache = {}


def _school_districts():
    """Precomputed spatial join (python src/geo.py), keyed by ListingKey."""
    if "table" not in _district_cache:
        _district_cache["table"] = pd.read_parquet(SCHOOL_DISTRICTS_PATH).set_index("ListingKey")
    return _district_cache["table"]


def add_features(df):
    """Row-wise features that learn nothing from data, so they are safe on any split."""
    out = df.copy()
    # Ratios: room layout given size. Undefined (NaN) when the denominator is missing or zero.
    out["bed_bath_ratio"] = out["BedroomsTotal"] / out["BathroomsTotalInteger"]
    out["sqft_per_bedroom"] = out["LivingArea"] / out["BedroomsTotal"]
    # School districts: rows that already carry them (e.g. from geo.lookup in the app) keep theirs
    if "school_district_k8" not in out:
        districts = _school_districts().reindex(out["ListingKey"])
        for c in districts.columns:
            out[c] = districts[c].to_numpy()
    month = pd.to_datetime(out["CloseDate"]).dt.month
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    # Assumption: a fee with no stated frequency is monthly (the most common frequency).
    periods = out["AssociationFeeFrequency"].map(FEE_PERIODS_PER_MONTH).fillna(1)
    out["hoa_monthly"] = out["AssociationFee"] / periods
    return out


def make_xy(df):
    """Every candidate feature column and the target. Each preprocessor picks its own set by name.

    Nullable pandas types become plain float/object for sklearn.
    """
    df = add_features(df)
    X = df[ALL_COLUMNS].copy()
    groups = {g: {c for fs in FEATURE_SETS.values() for c in fs[g]} for g in WEEK5}
    for c in groups["boolean"]:
        X[c] = X[c].astype("float")
    for c in groups["low_cardinality"] | groups["high_cardinality"] | groups["multi_hot"]:
        X[c] = X[c].astype("object").where(X[c].notna(), np.nan)
    y = df[TARGET].to_numpy() if TARGET in df else None
    return X, y


class MultiHot(BaseEstimator, TransformerMixin):
    """Comma-separated list column -> one 0/1 column per value seen at least `min_count` times
    in training, plus a missing flag. The vocabulary is learned in fit (training data only)."""

    def __init__(self, min_count=500):
        self.min_count = min_count

    def _split(self, X):
        col = pd.Series(np.asarray(X).ravel(), dtype="object")
        return col.isna(), col.fillna("").str.split(",").apply(lambda xs: {x.strip() for x in xs if x.strip()})

    def fit(self, X, y=None):
        _, sets = self._split(X)
        counts = pd.Series([v for s in sets for v in s]).value_counts()
        self.vocabulary_ = sorted(counts[counts >= self.min_count].index)
        self.feature_names_in_ = np.asarray(getattr(X, "columns", ["x0"]), dtype=object)
        return self

    def transform(self, X):
        missing, sets = self._split(X)
        cols = [sets.apply(lambda s, v=v: v in s).to_numpy() for v in self.vocabulary_]
        return np.column_stack(cols + [missing.to_numpy()]).astype(float)

    def get_feature_names_out(self, input_features=None):
        name = (input_features if input_features is not None else self.feature_names_in_)[0]
        return np.asarray([f"{name}_{v}" for v in self.vocabulary_] + [f"{name}_missing"], dtype=object)


def build_preprocessor(feature_set=DEFAULT_FEATURE_SET):
    """ColumnTransformer with every fit-time step for one feature set. Fit on training data only."""
    fs = FEATURE_SETS[feature_set]
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
    steps = [
        ("log_num", log_numeric, fs["log_numeric"]),
        ("num", numeric, fs["numeric"]),
        ("bool", boolean, fs["boolean"]),
        ("low_card", low_card, fs["low_cardinality"]),
        ("high_card", high_card, fs["high_cardinality"]),
        *[(f"multi_hot_{c}", MultiHot(), [c]) for c in fs["multi_hot"]],
    ]
    return ColumnTransformer(steps, remainder="drop", verbose_feature_names_out=False)


def leakage_audit(clean_columns):
    """One row per clean-table column: used as a feature or excluded, and why."""
    assert not set(FEATURES) & set(LEAKAGE_COLS), "leakage column listed as a feature"
    season = "month of sale only (at prediction time: the valuation month), no price information"
    district = "spatial join of coordinates to CA School District Areas 2024-25 (public boundary data)"
    derived = {
        "month_sin": season,
        "month_cos": season,
        "hoa_monthly": "AssociationFee divided by its payment frequency; HOA dues are known before a sale",
        "property_age": "close year minus YearBuilt; at prediction time: valuation year minus YearBuilt",
        "bed_bath_ratio": "BedroomsTotal / BathroomsTotalInteger (physical layout)",
        "sqft_per_bedroom": "LivingArea / BedroomsTotal (physical layout)",
        "Flooring": "flooring types as 0/1 columns (vocabulary learned on training data)",
        "school_district_k8": district + "; target-encoded",
        "school_district_hs": district + "; target-encoded",
        "district_type": district + "; Unified vs Elementary + High",
        "district_k8_enrollment": district + "; district size (demographic shares deliberately excluded)",
        "district_k8_assistance": district + "; state accountability status (school performance)",
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
