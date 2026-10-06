"""Train/test split and ClosePrice outlier cut for the AVM.

Rules (IDX Best Practices doc / game plan):
- Test = the latest close month. Train = the `train_months` months immediately before it.
  The training window length is tunable; compare a few lengths before picking one.
- Outliers: cut ClosePrice at the 0.5th / 99.5th percentile computed on the TRAINING set only,
  then apply the same cutoffs to test. Computing them on all data would leak test info.

Usage (from the repo root, after python src/cleaning.py):
    python src/preprocessing.py                      # default 12-month window
    python src/preprocessing.py --train-months 6

Writes:
    data/processed/train.csv, data/processed/test.csv  - git-ignored
    docs/split_summary.csv                             - rows and cutoffs per window length (counts only)
"""

import argparse
import os

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_PATH = os.path.join(REPO_ROOT, "data", "processed", "sfr_clean.parquet")
TRAIN_PATH = os.path.join(REPO_ROOT, "data", "processed", "train.csv")
TEST_PATH = os.path.join(REPO_ROOT, "data", "processed", "test.csv")
SUMMARY_PATH = os.path.join(REPO_ROOT, "docs", "split_summary.csv")

DEFAULT_TRAIN_MONTHS = 12
PRICE_QUANTILES = (0.005, 0.995)
WINDOWS_TO_COMPARE = [3, 6, 12, 24]


def load_clean(path=CLEAN_PATH):
    return pd.read_parquet(path)


def time_split(df, train_months=DEFAULT_TRAIN_MONTHS, test_month=None):
    """Return (train, test): test = test_month (default: latest), train = the months right before it."""
    months = pd.PeriodIndex(df["close_month"], freq="M")
    test_period = months.max() if test_month is None else pd.Period(test_month, freq="M")
    first_train = test_period - train_months
    available = (test_period - months.min()).n
    if train_months < 1 or train_months > available:
        raise ValueError(f"train_months must be between 1 and {available} for test month {test_period}")
    train = df[(months >= first_train) & (months < test_period)]
    test = df[months == test_period]
    return train.copy(), test.copy()


def fit_price_cutoffs(train, quantiles=PRICE_QUANTILES):
    """ClosePrice cutoffs from the training set only."""
    lo, hi = train["ClosePrice"].quantile(list(quantiles))
    return float(lo), float(hi)


def apply_price_cutoffs(df, cutoffs):
    lo, hi = cutoffs
    return df[df["ClosePrice"].between(lo, hi)].copy()


def make_split(df, train_months=DEFAULT_TRAIN_MONTHS, test_month=None, quantiles=PRICE_QUANTILES):
    """Split by time, fit cutoffs on train, apply to both. Returns (train, test, info)."""
    train_raw, test_raw = time_split(df, train_months, test_month)
    cutoffs = fit_price_cutoffs(train_raw, quantiles)
    train, test = apply_price_cutoffs(train_raw, cutoffs), apply_price_cutoffs(test_raw, cutoffs)
    info = {
        "train_months": train_months,
        "train_start": train_raw["close_month"].min(),
        "train_end": train_raw["close_month"].max(),
        "test_month": test_raw["close_month"].iloc[0],
        "price_low": round(cutoffs[0]),
        "price_high": round(cutoffs[1]),
        "train_rows_before_cut": len(train_raw),
        "train_rows": len(train),
        "test_rows_before_cut": len(test_raw),
        "test_rows": len(test),
        "test_rows_cut": len(test_raw) - len(test),
    }
    return train, test, info


def compare_windows(df, windows=WINDOWS_TO_COMPARE, test_month=None):
    """One row per training-window length: date range, cutoffs and row counts."""
    return pd.DataFrame([make_split(df, w, test_month)[2] for w in windows])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-months", type=int, default=DEFAULT_TRAIN_MONTHS)
    args = parser.parse_args()

    df = load_clean()
    summary = compare_windows(df, sorted(set(WINDOWS_TO_COMPARE + [args.train_months])))
    summary.to_csv(SUMMARY_PATH, index=False)
    print(summary.to_string(index=False))

    train, test, info = make_split(df, args.train_months)
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)
    print(f"\n{info['train_months']}-month window ({info['train_start']} to {info['train_end']}): "
          f"{len(train):,} train rows, {len(test):,} test rows ({info['test_month']}) "
          f"-> {os.path.relpath(TRAIN_PATH, REPO_ROOT)}, {os.path.relpath(TEST_PATH, REPO_ROOT)}")


if __name__ == "__main__":
    main()
