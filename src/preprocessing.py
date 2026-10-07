"""Train/validation/test split and ClosePrice outlier cut for the AVM.

Rules (IDX Best Practices doc / game plan):
- Test = the latest close month, touched only once at the end.
- Validation = the month right before test, used for tuning and model/window selection.
  Never tune on test.
- Train = the `train_months` months immediately before the validation month. The training
  window length is tunable; pick it by validation error.
- Outliers: cut ClosePrice at the 0.5th / 99.5th percentile computed on the TRAINING set only,
  then apply the same frozen cutoffs to validation and test. Computing them on all data would
  leak information from the held-out months.

Usage (from the repo root, after python src/cleaning.py):
    python src/preprocessing.py                      # default 12-month window
    python src/preprocessing.py --train-months 6

Writes:
    data/processed/train.csv, val.csv, test.csv        - git-ignored
    docs/split_summary.csv                             - rows and cutoffs per window length (counts only)
"""

import argparse
import os

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_PATH = os.path.join(REPO_ROOT, "data", "processed", "sfr_clean.parquet")
TRAIN_PATH = os.path.join(REPO_ROOT, "data", "processed", "train.csv")
VAL_PATH = os.path.join(REPO_ROOT, "data", "processed", "val.csv")
TEST_PATH = os.path.join(REPO_ROOT, "data", "processed", "test.csv")
SUMMARY_PATH = os.path.join(REPO_ROOT, "docs", "split_summary.csv")

DEFAULT_TRAIN_MONTHS = 12
PRICE_QUANTILES = (0.005, 0.995)
WINDOWS_TO_COMPARE = [3, 6, 12, 24]


def load_clean(path=CLEAN_PATH):
    return pd.read_parquet(path)


def time_split(df, train_months=DEFAULT_TRAIN_MONTHS, test_month=None):
    """Return (train, val, test).

    test = test_month (default: latest month), val = the month before it,
    train = the `train_months` months before val.
    """
    months = pd.PeriodIndex(df["close_month"], freq="M")
    test_period = months.max() if test_month is None else pd.Period(test_month, freq="M")
    val_period = test_period - 1
    first_train = val_period - train_months
    available = (val_period - months.min()).n
    if train_months < 1 or train_months > available:
        raise ValueError(f"train_months must be between 1 and {available} for test month {test_period}")
    train = df[(months >= first_train) & (months < val_period)]
    val = df[months == val_period]
    test = df[months == test_period]
    return train.copy(), val.copy(), test.copy()


def fit_price_cutoffs(train, quantiles=PRICE_QUANTILES):
    """ClosePrice cutoffs from the training set only."""
    lo, hi = train["ClosePrice"].quantile(list(quantiles))
    return float(lo), float(hi)


def apply_price_cutoffs(df, cutoffs):
    lo, hi = cutoffs
    return df[df["ClosePrice"].between(lo, hi)].copy()


def make_split(df, train_months=DEFAULT_TRAIN_MONTHS, test_month=None, quantiles=PRICE_QUANTILES):
    """Split by time, fit cutoffs on train, apply them to all three. Returns (train, val, test, info)."""
    train_raw, val_raw, test_raw = time_split(df, train_months, test_month)
    cutoffs = fit_price_cutoffs(train_raw, quantiles)
    train, val, test = (apply_price_cutoffs(d, cutoffs) for d in (train_raw, val_raw, test_raw))
    info = {
        "train_months": train_months,
        "train_start": train_raw["close_month"].min(),
        "train_end": train_raw["close_month"].max(),
        "val_month": val_raw["close_month"].iloc[0],
        "test_month": test_raw["close_month"].iloc[0],
        "price_low": round(cutoffs[0]),
        "price_high": round(cutoffs[1]),
        "train_rows_before_cut": len(train_raw),
        "train_rows": len(train),
        "val_rows_before_cut": len(val_raw),
        "val_rows": len(val),
        "test_rows_before_cut": len(test_raw),
        "test_rows": len(test),
    }
    return train, val, test, info


def compare_windows(df, windows=WINDOWS_TO_COMPARE, test_month=None):
    """One row per training-window length: date range, cutoffs and row counts."""
    return pd.DataFrame([make_split(df, w, test_month)[3] for w in windows])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-months", type=int, default=DEFAULT_TRAIN_MONTHS)
    args = parser.parse_args()

    df = load_clean()
    summary = compare_windows(df, sorted(set(WINDOWS_TO_COMPARE + [args.train_months])))
    summary.to_csv(SUMMARY_PATH, index=False)
    print(summary.to_string(index=False))

    train, val, test, info = make_split(df, args.train_months)
    for part, path in ((train, TRAIN_PATH), (val, VAL_PATH), (test, TEST_PATH)):
        part.to_csv(path, index=False)
    print(f"\n{info['train_months']}-month window ({info['train_start']} to {info['train_end']}): "
          f"{len(train):,} train, {len(val):,} validation ({info['val_month']}), "
          f"{len(test):,} test ({info['test_month']}) -> data/processed/{{train,val,test}}.csv")


if __name__ == "__main__":
    main()
