"""Model training, training-window selection, test evaluation and rolling backtest.

Procedure (IDX Best Practices §01, §09; game plan Week 4):
1. Window selection: for each candidate window length, fit on the `w` months before the
   validation month and score on the validation month. Pick the lowest validation MdAPE.
   The test month is not looked at.
2. Test: refit with the chosen window on the `w` months immediately before the test month
   (so the validation month is included, as in production), then score the test month once.
3. Rolling backtest: repeat step 2 with the same window for earlier test months to check
   the metrics are stable over time.

Every model is a Pipeline(preprocessor, estimator), so preprocessing is always fit on the
training rows only. Models predict log(ClosePrice); predictions are converted back to dollars.

Usage (from the repo root, after cleaning.py):
    python src/models.py                 # Linear Regression baseline
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

import evaluation
import features
import preprocessing

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(REPO_ROOT, "models")
DOCS_DIR = os.path.join(REPO_ROOT, "docs")

WINDOWS = [3, 6, 12, 24]
BACKTEST_MONTHS = ["2026-03", "2026-02", "2026-01"]


def make_model(estimator):
    return Pipeline([("preprocess", features.build_preprocessor()), ("model", estimator)])


def fit(model, train):
    X, y = features.make_xy(train)
    return model.fit(X, y)


def predict_dollars(model, df):
    X, _ = features.make_xy(df)
    return np.exp(model.predict(X))


def score(model, df):
    return evaluation.regression_metrics(df["ClosePrice"], predict_dollars(model, df))


def select_window(df, estimator_factory, windows=WINDOWS, test_month=None):
    """Fit on each window, score on the validation month. Returns a table; never touches test."""
    rows = []
    for w in windows:
        train, val, _, info = preprocessing.make_split(df, w, test_month)
        model = fit(make_model(estimator_factory()), train)
        rows.append({"train_months": w, "train_start": info["train_start"], "train_end": info["train_end"],
                     "val_month": info["val_month"], "train_rows": len(train), **score(model, val)})
    return pd.DataFrame(rows)


def fit_for_test(df, estimator_factory, train_months, test_month=None):
    """Refit on the `train_months` months right before the test month. Returns (model, test, info)."""
    train, test, info = preprocessing.make_final_split(df, train_months, test_month)
    return fit(make_model(estimator_factory()), train), test, info


def backtest(df, estimator_factory, train_months, test_months=BACKTEST_MONTHS):
    rows = []
    for m in test_months:
        model, test, info = fit_for_test(df, estimator_factory, train_months, m)
        rows.append({**info, **score(model, test)})
    return pd.DataFrame(rows)


def run(name, estimator_factory, df=None):
    """Full procedure for one model type. Writes docs/metrics_<name>.csv and models/<name>.joblib."""
    df = preprocessing.load_clean() if df is None else df
    windows = select_window(df, estimator_factory)
    best = int(windows.loc[windows["MdAPE"].idxmin(), "train_months"])

    model, test, info = fit_for_test(df, estimator_factory, best)
    pred = predict_dollars(model, test)
    test_metrics = {**info, **evaluation.regression_metrics(test["ClosePrice"], pred)}
    by_band = evaluation.metrics_by_group(test["ClosePrice"], pred, evaluation.price_bands(test["ClosePrice"]))
    by_county = evaluation.metrics_by_group(test["ClosePrice"], pred, test["CountyOrParish"], min_n=200)
    history = backtest(df, estimator_factory, best)

    summary = pd.concat([
        windows.assign(stage="validation (window selection)"),
        pd.DataFrame([test_metrics]).assign(stage="test"),
        history.assign(stage="backtest"),
    ], ignore_index=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    summary.to_csv(os.path.join(DOCS_DIR, f"metrics_{name}.csv"), index=False)
    joblib.dump({"model": model, "features": features.FEATURES, "train_info": info},
                os.path.join(MODEL_DIR, f"{name}.joblib"))
    return {"windows": windows, "best_window": best, "test": test_metrics, "by_band": by_band,
            "by_county": by_county, "backtest": history, "model": model, "test_rows": test, "pred": pred}


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    result = run("baseline_linear_regression", LinearRegression)
    cols = ["n", "R2", "MAPE", "MdAPE", "MAE", "RMSE", "within_10pct"]
    print("Window selection (validation 2026-03):")
    print(result["windows"][["train_months", "train_rows"] + cols].to_string(index=False))
    print(f"\nChosen window: {result['best_window']} months")
    print("\nTest (2026-04):", {k: round(v, 3) if isinstance(v, float) else v for k, v in result["test"].items()})
    print("\nBy price band:\n", result["by_band"][["group"] + cols].to_string(index=False))
    print("\nBacktest:\n", result["backtest"][["test_month", "train_start", "train_end"] + cols].to_string(index=False))
