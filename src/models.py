"""Model training, model/window selection, test evaluation and rolling backtest.

Procedure (IDX Best Practices §01, §09; game plan Weeks 4-5):
1. Selection on validation only: for each training-window length and each hyperparameter
   setting, fit on the `w` months before the validation month and score the validation month.
   Keep the combination with the lowest validation MdAPE. The test month is not looked at.
2. Test: refit the chosen combination on the `w` months immediately before the test month
   (so the validation month is included, as in production), then score the test month once.
3. Rolling backtest: repeat step 2 with the same choice for earlier test months to check the
   metrics are stable over time.

Every model is a Pipeline(preprocessor, estimator), so preprocessing is always fit on the
training rows only. Models predict log(ClosePrice); predictions are converted back to dollars.
Random seeds are fixed (RANDOM_STATE) so results reproduce.

Usage (from the repo root, after cleaning.py):
    python src/models.py                      # all models in MODELS
    python src/models.py random_forest        # one model
"""

import os
import sys
from functools import partial

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

import evaluation
import features
import preprocessing

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(REPO_ROOT, "models")
DOCS_DIR = os.path.join(REPO_ROOT, "docs")

WINDOWS = [3, 6, 12, 24]
BACKTEST_MONTHS = ["2026-03", "2026-02", "2026-01"]
RANDOM_STATE = features.RANDOM_STATE

# name -> (estimator factory, hyperparameter grid searched on the validation month)
MODELS = {
    "baseline_linear_regression": (LinearRegression, {}),
    "decision_tree": (
        partial(DecisionTreeRegressor, random_state=RANDOM_STATE),
        {"max_depth": [8, 12, 16, 20, None], "min_samples_leaf": [5, 20, 50]},
    ),
    "random_forest": (
        # 200 trees: more trees only reduce variance; leaf size and features per split are tuned
        partial(RandomForestRegressor, n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE),
        {"min_samples_leaf": [1, 5], "max_features": [0.33, 0.5]},
    ),
}


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


def select(df, estimator_factory, param_grid=None, windows=WINDOWS, test_month=None):
    """Fit every (window, params) combination and score it on the validation month.

    Returns one row per combination; never touches the test month.
    """
    rows = []
    for w in windows:
        train, val, _, info = preprocessing.make_split(df, w, test_month)
        for params in ParameterGrid(param_grid or {}):
            model = fit(make_model(estimator_factory(**params)), train)
            rows.append({"train_months": w, "params": params, "train_start": info["train_start"],
                         "train_end": info["train_end"], "val_month": info["val_month"],
                         "train_rows": len(train), **score(model, val)})
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


def run(name, estimator_factory, param_grid=None, df=None, save=True):
    """Full procedure for one model type. Writes docs/metrics_<name>.csv and models/<name>.joblib."""
    df = preprocessing.load_clean() if df is None else df
    selection = select(df, estimator_factory, param_grid)
    best_row = selection.loc[selection["MdAPE"].idxmin()]
    best_window, best_params = int(best_row["train_months"]), best_row["params"]
    chosen = partial(estimator_factory, **best_params)

    model, test, info = fit_for_test(df, chosen, best_window)
    pred = predict_dollars(model, test)
    test_metrics = {**info, "params": best_params, **evaluation.regression_metrics(test["ClosePrice"], pred)}
    by_band = evaluation.metrics_by_group(test["ClosePrice"], pred, evaluation.price_bands(test["ClosePrice"]))
    by_county = evaluation.metrics_by_group(test["ClosePrice"], pred, test["CountyOrParish"], min_n=200)
    history = backtest(df, chosen, best_window).assign(params=[best_params] * len(BACKTEST_MONTHS))

    summary = pd.concat([
        selection.assign(stage="validation (selection)"),
        pd.DataFrame([test_metrics]).assign(stage="test"),
        history.assign(stage="backtest"),
    ], ignore_index=True)
    summary.insert(0, "model", name)
    summary.to_csv(os.path.join(DOCS_DIR, f"metrics_{name}.csv"), index=False)
    if save:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump({"model": model, "features": features.FEATURES, "train_info": info, "params": best_params},
                    os.path.join(MODEL_DIR, f"{name}.joblib"), compress=3)
    return {"name": name, "selection": selection, "best_window": best_window, "best_params": best_params,
            "test": test_metrics, "by_band": by_band, "by_county": by_county, "backtest": history,
            "model": model, "test_rows": test, "pred": pred}


def run_all(names=None, df=None):
    df = preprocessing.load_clean() if df is None else df
    return {n: run(n, *MODELS[n], df=df) for n in (names or MODELS)}


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    cols = ["R2", "MAPE", "MdAPE", "MAE", "within_10pct"]
    for name, result in run_all(sys.argv[1:] or None).items():
        print(f"\n=== {name}: {result['best_window']}-month window, params {result['best_params']}")
        print("test:", {k: round(result["test"][k], 3) for k in cols})
        print(result["backtest"][["test_month"] + cols].to_string(index=False))
