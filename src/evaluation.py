"""Evaluation metrics for the AVM (IDX Best Practices §08).

All metrics are computed in dollars on ClosePrice, even though models are trained on
log(ClosePrice). MdAPE is the headline number: it is scale-free and robust to the few
properties with very large percentage errors.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred):
    """R², MAPE, MdAPE, MAE and RMSE for dollar-valued predictions."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    pct_error = (y_pred - y_true) / y_true
    ape = np.abs(pct_error)
    return {
        "n": len(y_true),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": float(100 * ape.mean()),
        "MdAPE": float(100 * np.median(ape)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "within_10pct": float(100 * (ape <= 0.10).mean()),
        # Signed: > 0 means the model over-values the typical home in this set
        "median_pct_error": float(100 * np.median(pct_error)),
    }


def metrics_by_group(y_true, y_pred, groups, min_n=1):
    """regression_metrics for each group (e.g. price band or county).

    R² is dropped: within a narrow price band there is little variance to explain, so R²
    there is often negative and not meaningful. Use MdAPE / MAPE to compare groups.
    """
    frame = pd.DataFrame({"y": np.asarray(y_true, dtype=float), "p": np.asarray(y_pred, dtype=float),
                          "g": np.asarray(groups)})
    rows = [{"group": g, **regression_metrics(d["y"], d["p"])}
            for g, d in frame.groupby("g", observed=True) if len(d) >= max(min_n, 2)]
    return pd.DataFrame(rows).drop(columns="R2")


def price_bands(y_true, q=5):
    """Quintile labels of the actual price, e.g. 'Q1 ($190k–$560k)'."""
    bands = pd.qcut(np.asarray(y_true, dtype=float), q=q)
    labels = [f"Q{i + 1} (${b.left / 1e3:,.0f}k–${b.right / 1e3:,.0f}k)" for i, b in enumerate(bands.categories)]
    return bands.rename_categories(labels)


def paired_bootstrap_mdape(y_true, pred_a, pred_b, n_boot=2000, seed=42):
    """How much lower model B's MdAPE is than model A's, with a 95% bootstrap interval.

    Both models are scored on the same resampled homes each round, so the interval reflects
    the uncertainty in the difference itself (IDX Best Practices §09: the improvement over the
    baseline must be larger than noise). If the interval excludes 0, the gain is not noise.
    """
    y, a, b = (np.asarray(v, dtype=float) for v in (y_true, pred_a, pred_b))
    ape_a, ape_b = np.abs(a - y) / y, np.abs(b - y) / y
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    diffs = 100 * (np.median(ape_a[idx], axis=1) - np.median(ape_b[idx], axis=1))
    return {
        "mdape_a": float(100 * np.median(ape_a)),
        "mdape_b": float(100 * np.median(ape_b)),
        "improvement_pts": float(100 * (np.median(ape_a) - np.median(ape_b))),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
    }
