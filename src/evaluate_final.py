"""Final evaluation of the tuned models (game plan Week 8).

1. Final model = the tuned model with the lowest validation MdAPE (2026-03). The test month
   plays no part in the choice.
2. Every model is scored on the same test homes (preprocessing.eval_set): overall, by price
   quintile and by county, plus the rolling backtest already in docs/metrics_<model>__week6__tuned.csv.
3. Prediction intervals for the final model: the spread of its percentage errors on the
   validation month (fit on the months before it), by band of PREDICTED price, since the true
   price is unknown when a home is valued. Their coverage is then checked on the test month.

Writes:
    docs/metrics_summary.csv        every model x stage / month x segment (overall, price band, county)
    docs/prediction_intervals.csv   final model's interval widths per predicted-price band + test coverage
    docs/figures/model_mdape_by_month.png, docs/figures/final_model_by_price_band.png

Usage (from the repo root, after python src/models.py --tuned):
    python src/evaluate_final.py
"""

import ast
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import evaluation
import features
import models
import preprocessing
from make_figures import FIG_DIR, GRID, SERIES, TEXT, TEXT_MUTED, _save

FEATURE_SET = features.DEFAULT_FEATURE_SET
LABEL = models.TUNED_LABEL
NAMES = list(models.MODELS)
LABELS = {"baseline_linear_regression": "Linear Regression", "decision_tree": "Decision Tree",
          "random_forest": "Random Forest", "lightgbm": "LightGBM", "xgboost": "XGBoost"}
INTERVAL_LEVELS = (0.8, 0.9)
SUMMARY_PATH = os.path.join(models.DOCS_DIR, "metrics_summary.csv")
INTERVALS_PATH = os.path.join(models.DOCS_DIR, "prediction_intervals.csv")


def stem(name):
    return models.artifact_stem(name, FEATURE_SET, LABEL)


def load_metrics(names=NAMES):
    return pd.concat([pd.read_csv(os.path.join(models.DOCS_DIR, f"metrics_{stem(n)}.csv")) for n in names],
                     ignore_index=True)


def pick_final(metrics):
    """Name of the model with the lowest validation MdAPE (best window and setting per model)."""
    val = metrics[metrics["stage"] == "validation (selection)"]
    return val.groupby("model")["MdAPE"].min().idxmin()


def load_models(names=NAMES):
    out = {}
    for n in names:
        path = os.path.join(models.MODEL_DIR, f"{stem(n)}.joblib")
        out[n] = {**joblib.load(path), "file_mb": os.path.getsize(path) / 1e6}
    return out


def test_predictions(df, arts):
    """Shared test homes and each model's predictions in dollars."""
    test = preprocessing.eval_set(df, df["close_month"].max())
    preds = pd.DataFrame({n: models.predict_dollars(a["model"], test) for n, a in arts.items()}, index=test.index)
    return test, preds


def summary_table(metrics, test, preds):
    """Long table: model x stage x month x segment."""
    rows = []
    for n in preds:
        m = metrics[metrics["model"] == n]
        best_val = m[m["stage"] == "validation (selection)"].sort_values("MdAPE").iloc[0]
        rows.append({"model": n, "stage": "validation", "month": best_val["val_month"], "segment_type": "overall",
                     "segment": "all", "train_months": best_val["train_months"], "params": best_val["params"],
                     **{k: best_val[k] for k in ["n", "R2", "MAPE", "MdAPE", "MAE", "RMSE", "within_10pct", "median_pct_error"]}})
        for _, r in m[m["stage"] == "backtest"].iterrows():
            rows.append({"model": n, "stage": "backtest", "month": r["test_month"], "segment_type": "overall",
                         "segment": "all", "train_months": r["train_months"], "params": r["params"],
                         **{k: r[k] for k in ["n", "R2", "MAPE", "MdAPE", "MAE", "RMSE", "within_10pct", "median_pct_error"]}})
        test_info = m[m["stage"] == "test"].iloc[0]
        base = {"model": n, "stage": "test", "month": test["close_month"].iloc[0],
                "train_months": test_info["train_months"], "params": test_info["params"]}
        rows.append({**base, "segment_type": "overall", "segment": "all",
                     **evaluation.regression_metrics(test["ClosePrice"], preds[n])})
        for kind, groups, min_n in (("price_band", evaluation.price_bands(test["ClosePrice"]), 1),
                                    ("county", test["CountyOrParish"], 200)):
            g = evaluation.metrics_by_group(test["ClosePrice"], preds[n], groups, min_n=min_n)
            rows += [{**base, "segment_type": kind, **r} for r in g.rename(columns={"group": "segment"}).to_dict("records")]
    out = pd.DataFrame(rows)
    out.insert(1, "model_label", out["model"].map(LABELS))
    return out


def prediction_intervals(df, metrics, final, test, test_pred):
    """Empirical intervals from the final model's validation errors; coverage checked on test."""
    chosen = metrics[(metrics["model"] == final) & (metrics["stage"] == "test")].iloc[0]
    window, params = int(chosen["train_months"]), ast.literal_eval(chosen["params"])
    factory, _ = models.MODELS[final]
    train, _, _, info = preprocessing.make_split(df, window)
    val = preprocessing.eval_set(df, info["val_month"])
    model = models.fit(models.make_model(factory(**params), FEATURE_SET), train)
    val_pred = models.predict_dollars(model, val)
    ratio = val["ClosePrice"].to_numpy() / val_pred  # actual / predicted

    edges = np.quantile(val_pred, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    edges[0], edges[-1] = 0, np.inf
    val_band = pd.cut(val_pred, edges, labels=False)
    test_band = pd.cut(np.asarray(test_pred), edges, labels=False)
    test_ratio = test["ClosePrice"].to_numpy() / np.asarray(test_pred)
    rows = []
    for b in range(5):
        r = ratio[val_band == b]
        row = {"band": f"P{b + 1}", "predicted_low": edges[b], "predicted_high": edges[b + 1],
               "val_n": len(r), "test_n": int((test_band == b).sum())}
        for level in INTERVAL_LEVELS:
            lo, hi = np.quantile(r, [(1 - level) / 2, (1 + level) / 2])
            t = test_ratio[test_band == b]
            pct = int(level * 100)
            row.update({f"low_{pct}": lo - 1, f"high_{pct}": hi - 1,
                        f"test_coverage_{pct}": float(((t >= lo) & (t <= hi)).mean())})
        rows.append(row)
    return pd.DataFrame(rows)


def fig_by_month(metrics, final):
    hist = metrics[metrics["stage"].isin(["test", "backtest"])]
    n_months = hist["test_month"].nunique()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    last = {}
    for n in NAMES:
        h = hist[hist["model"] == n].sort_values("test_month")
        is_final = n == final
        ax.plot(h["test_month"], h["MdAPE"], marker="o", markersize=6 if is_final else 4,
                linewidth=2.2 if is_final else 1.4, color=SERIES if is_final else "#a3a29d", zorder=3 if is_final else 2)
        last[n] = h["MdAPE"].iloc[-1]
    # Direct labels at the last point, pushed apart so close lines stay readable
    min_gap, placed = 0.6, {}
    for n in sorted(last, key=last.get):
        y = last[n] if not placed else max(last[n], max(placed.values()) + min_gap)
        placed[n] = y
    for n, y in placed.items():
        is_final = n == final
        ax.annotate(f"{LABELS[n]}  {last[n]:.2f}%", xy=(n_months - 1, last[n]), xytext=(n_months - 1 + 0.12, y),
                    va="center", fontsize=9, color=TEXT if is_final else TEXT_MUTED,
                    fontweight="bold" if is_final else "normal")
    ax.set_ylim(0, None)
    ax.set_xlim(-0.2, n_months - 1 + 1.35)
    ax.set_ylabel("MdAPE (%)")
    ax.set_title("Median absolute % error by test month (tuned models)")
    _save(fig, "model_mdape_by_month.png")


def fig_by_band(summary, final):
    t = summary[(summary["model"] == final) & (summary["stage"] == "test") & (summary["segment_type"] == "price_band")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    # "Q1 ($200k–$575k)" -> "Q1\n$200k–575k"
    labels = [seg.replace(" ($", "\n$").replace("–$", "–").rstrip(")") for seg in t["segment"]]
    axes[0].bar(labels, t["MdAPE"], color=SERIES, width=0.6)
    axes[0].set_title("MdAPE (%) by price quintile")
    axes[1].bar(labels, t["median_pct_error"], color=SERIES, width=0.6)
    axes[1].axhline(0, color=TEXT_MUTED, linewidth=0.8)
    lo, hi = t["median_pct_error"].min(), t["median_pct_error"].max()
    axes[1].set_ylim(min(lo, 0) * 1.25 - 0.3, max(hi, 0) * 1.25 + 0.3)
    axes[1].set_title("Median signed error (%), > 0 = over-valued")
    for ax, col in zip(axes, ["MdAPE", "median_pct_error"]):
        for x, v in enumerate(t[col]):
            ax.annotate(f"{v:+.1f}" if col == "median_pct_error" else f"{v:.1f}", (x, v),
                        xytext=(0, 3 if v >= 0 else -3), textcoords="offset points",
                        ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5, color=TEXT)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.grid(axis="x", visible=False)
    fig.suptitle(f"{LABELS[final]} on the 2026-04 test month", x=0.01, ha="left", color=TEXT_MUTED, fontsize=10)
    fig.tight_layout()
    _save(fig, "final_model_by_price_band.png")


def main():
    df = preprocessing.load_clean()
    metrics = load_metrics()
    final = pick_final(metrics)
    arts = load_models()
    test, preds = test_predictions(df, arts)
    summary = summary_table(metrics, test, preds)
    summary["final_model"] = summary["model"] == final
    summary.to_csv(SUMMARY_PATH, index=False)
    intervals = prediction_intervals(df, metrics, final, test, preds[final])
    intervals.to_csv(INTERVALS_PATH, index=False)
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_by_month(metrics, final)
    fig_by_band(summary, final)

    pd.set_option("display.width", 200)
    overall = summary[(summary["segment_type"] == "overall") & (summary["stage"].isin(["validation", "test"]))]
    print(f"final model (lowest validation MdAPE): {LABELS[final]}\n")
    print(overall.pivot(index="model_label", columns="stage", values="MdAPE").round(3).to_string())
    print("\n", intervals.round(3).to_string(index=False))
    print("\nmodel files (MB):", {LABELS[n]: round(a["file_mb"], 1) for n, a in arts.items()})


if __name__ == "__main__":
    main()
