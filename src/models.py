"""Model training, model/window selection, test evaluation and rolling backtest.

Models: Linear Regression baseline, Decision Tree, Random Forest (Week 5) and gradient-boosted
trees with LightGBM and XGBoost (Week 7).

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

Outputs are tagged with the feature set (features.FEATURE_SETS):
docs/metrics_<model>__<feature set>.csv and models/<model>__<feature set>.joblib.

Usage (from the repo root, after cleaning.py and geo.py):
    python src/models.py                              # all models, default feature set
    python src/models.py random_forest                # one model
    python src/models.py --features week5 random_forest
    python src/models.py lightgbm xgboost             # Week 7 boosting models
"""

import ast
import os
import sys
from functools import partial

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

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
    # Gradient boosting (game plan Week 7). Learning rate and tree count are fixed (a small rate
    # with many trees is the safe default); light tuning covers tree size and minimum leaf size,
    # the two settings that control how closely each tree fits. Row and column subsampling add
    # randomness that usually helps on noisy price data.
    "lightgbm": (
        partial(lgb.LGBMRegressor, n_estimators=2000, learning_rate=0.03, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
        {"num_leaves": [63, 255], "min_child_samples": [10, 40]},
    ),
    "xgboost": (
        partial(XGBRegressor, n_estimators=2000, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1),
        {"max_depth": [6, 10], "min_child_weight": [1, 10]},
    ),
}


def make_model(estimator, feature_set=features.DEFAULT_FEATURE_SET):
    return Pipeline([("preprocess", features.build_preprocessor(feature_set)), ("model", estimator)])


def fit(model, train):
    X, y = features.make_xy(train)
    return model.fit(X, y)


def predict_dollars(model, df):
    X, _ = features.make_xy(df)
    return np.exp(model.predict(X))


def score(model, df):
    return evaluation.regression_metrics(df["ClosePrice"], predict_dollars(model, df))


def select(df, estimator_factory, param_grid=None, windows=WINDOWS, test_month=None,
           feature_set=features.DEFAULT_FEATURE_SET):
    """Fit every (window, params) combination and score it on the validation month.

    Returns one row per combination; never touches the test month.
    """
    rows = []
    for w in windows:
        train, val, _, info = preprocessing.make_split(df, w, test_month)
        for params in ParameterGrid(param_grid or {}):
            model = fit(make_model(estimator_factory(**params), feature_set), train)
            rows.append({"train_months": w, "params": params, "train_start": info["train_start"],
                         "train_end": info["train_end"], "val_month": info["val_month"],
                         "train_rows": len(train), **score(model, val)})
    return pd.DataFrame(rows)


def fit_for_test(df, estimator_factory, train_months, test_month=None, feature_set=features.DEFAULT_FEATURE_SET):
    """Refit on the `train_months` months right before the test month. Returns (model, test, info)."""
    train, test, info = preprocessing.make_final_split(df, train_months, test_month)
    return fit(make_model(estimator_factory(), feature_set), train), test, info


def backtest(df, estimator_factory, train_months, test_months=BACKTEST_MONTHS,
             feature_set=features.DEFAULT_FEATURE_SET):
    rows = []
    for m in test_months:
        model, test, info = fit_for_test(df, estimator_factory, train_months, m, feature_set)
        rows.append({**info, **score(model, test)})
    return pd.DataFrame(rows)


def artifact_stem(name, feature_set):
    return f"{name}__{feature_set}"


def run(name, estimator_factory, param_grid=None, df=None, save=True, feature_set=features.DEFAULT_FEATURE_SET):
    """Full procedure for one model type and feature set.

    Writes docs/metrics_<name>__<feature set>.csv and models/<name>__<feature set>.joblib.
    """
    df = preprocessing.load_clean() if df is None else df
    selection = select(df, estimator_factory, param_grid, feature_set=feature_set)
    best_row = selection.loc[selection["MdAPE"].idxmin()]
    best_window, best_params = int(best_row["train_months"]), best_row["params"]
    chosen = partial(estimator_factory, **best_params)

    model, test, info = fit_for_test(df, chosen, best_window, feature_set=feature_set)
    pred = predict_dollars(model, test)
    test_metrics = {**info, "params": best_params, **evaluation.regression_metrics(test["ClosePrice"], pred)}
    by_band = evaluation.metrics_by_group(test["ClosePrice"], pred, evaluation.price_bands(test["ClosePrice"]))
    by_county = evaluation.metrics_by_group(test["ClosePrice"], pred, test["CountyOrParish"], min_n=200)
    history = backtest(df, chosen, best_window, feature_set=feature_set).assign(
        params=[best_params] * len(BACKTEST_MONTHS))

    summary = pd.concat([
        selection.assign(stage="validation (selection)"),
        pd.DataFrame([test_metrics]).assign(stage="test"),
        history.assign(stage="backtest"),
    ], ignore_index=True)
    summary.insert(0, "model", name)
    summary.insert(1, "feature_set", feature_set)
    stem = artifact_stem(name, feature_set)
    summary.to_csv(os.path.join(DOCS_DIR, f"metrics_{stem}.csv"), index=False)
    if save:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump({"model": model, "feature_set": feature_set, "features": features.FEATURE_SETS[feature_set],
                     "train_info": info, "params": best_params},
                    os.path.join(MODEL_DIR, f"{stem}.joblib"), compress=3)
    return {"name": name, "feature_set": feature_set, "selection": selection, "best_window": best_window, "best_params": best_params,
            "test": test_metrics, "by_band": by_band, "by_county": by_county, "backtest": history,
            "model": model, "test_rows": test, "pred": pred}


def compare_feature_sets(df=None, names=None, feature_sets=None, reference_set="week5"):
    """Old vs new feature sets (game plan Week 6), holding each model's setup fixed.

    Each model keeps the window and hyperparameters it chose on validation with
    `reference_set`, so the only thing that changes between rows is the feature set.
    Scores the validation month (used to decide which features to adopt) and the test month.
    Writes docs/feature_set_comparison.csv.
    """
    df = preprocessing.load_clean() if df is None else df
    feature_sets = feature_sets or list(features.FEATURE_SETS)
    rows = []
    for name in names or MODELS:
        factory, _ = MODELS[name]
        ref = pd.read_csv(os.path.join(DOCS_DIR, f"metrics_{artifact_stem(name, reference_set)}.csv"))
        ref = ref[ref["stage"] == "test"].iloc[0]
        window, params = int(ref["train_months"]), ast.literal_eval(ref["params"])
        chosen = partial(factory, **params)
        train, val, _, _ = preprocessing.make_split(df, window)
        for fs in feature_sets:
            val_model = fit(make_model(chosen(), fs), train)
            test_model, test, _ = fit_for_test(df, chosen, window, feature_set=fs)
            for stage, model, part in (("validation", val_model, val), ("test", test_model, test)):
                rows.append({"model": name, "feature_set": fs, "stage": stage, "train_months": window,
                             "params": params, **score(model, part)})
            print(f"{name:28s} {fs:24s} val MdAPE {rows[-2]['MdAPE']:.3f}  test MdAPE {rows[-1]['MdAPE']:.3f}",
                  flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(DOCS_DIR, "feature_set_comparison.csv"), index=False)
    return out


def run_all(names=None, df=None, feature_set=features.DEFAULT_FEATURE_SET):
    df = preprocessing.load_clean() if df is None else df
    return {n: run(n, *MODELS[n], df=df, feature_set=feature_set) for n in (names or MODELS)}


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    cols = ["R2", "MAPE", "MdAPE", "MAE", "within_10pct"]
    args = sys.argv[1:]
    feature_set = features.DEFAULT_FEATURE_SET
    if "--features" in args:
        i = args.index("--features")
        feature_set = args[i + 1]
        args = args[:i] + args[i + 2:]
    for name, result in run_all(args or None, feature_set=feature_set).items():
        print(f"\n=== {name} [{feature_set}]: {result['best_window']}-month window, params {result['best_params']}")
        print("test:", {k: round(result["test"][k], 3) for k in cols})
        print(result["backtest"][["test_month"] + cols].to_string(index=False))
