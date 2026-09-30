"""Save the key EDA / cleaning charts as PNGs in docs/figures/ for the README.

Notebook outputs are stripped on commit, so these PNGs are how results show up on GitHub.

Usage (from the repo root, after python src/cleaning.py):
    python src/make_figures.py
"""

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_PATH = os.path.join(REPO_ROOT, "data", "processed", "sfr_clean.parquet")
LOG_PATH = os.path.join(REPO_ROOT, "docs", "cleaning_log.csv")
FIG_DIR = os.path.join(REPO_ROOT, "docs", "figures")

SERIES = "#2a78d6"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_MUTED, "axes.titlecolor": TEXT,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": TEXT_MUTED, "ytick.color": TEXT_MUTED, "font.size": 10,
    "axes.axisbelow": True,
    "text.parse_math": False,  # labels contain "$"
})

def fmt_dollars(x):
    if x >= 1e6:
        return f"${x / 1e6:.2f}".rstrip("0").rstrip(".") + "M"
    return f"${x / 1e3:,.0f}k" if x else "$0"


dollars = mticker.FuncFormatter(lambda x, _: fmt_dollars(x))


def _save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(path, REPO_ROOT))


def cleaning_steps(log):
    steps = log[(log["rows_removed"] > 0) & (log["step"] != "not Residential / SingleFamilyResidence")]
    steps = steps.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.barh(steps["step"], steps["rows_removed"], color=SERIES, height=0.6)
    for y, v in enumerate(steps["rows_removed"]):
        ax.text(v + 4, y, f"{v:,}", va="center", color=TEXT, fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("single-family sales removed")
    sfr = log.loc[log["step"] == "not Residential / SingleFamilyResidence", "rows_after"].item()
    final = log["rows_after"].iloc[-1]
    ax.set_title(f"Cleaning keeps {final:,} of {sfr:,} single-family sales ({final / sfr:.1%})")
    ax.set_xlim(0, steps["rows_removed"].max() * 1.15)
    _save(fig, "cleaning_steps.png")


def price_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bins = np.logspace(np.log10(df["ClosePrice"].min()), np.log10(df["ClosePrice"].max()), 80)
    ax.hist(df["ClosePrice"], bins=bins, color=SERIES, edgecolor=SURFACE, linewidth=0.5)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(dollars)
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    med = df["ClosePrice"].median()
    ax.axvline(med, color=TEXT, linewidth=1, linestyle="--")
    top = ax.get_ylim()[1] * 1.12
    ax.set_ylim(0, top)
    ax.text(med * 1.08, top * 0.95, f"median {fmt_dollars(med)}", color=TEXT, fontsize=9, va="top")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_xlabel("close price (log scale)")
    ax.set_ylabel("sales")
    ax.set_title("Close prices are right-skewed, roughly log-normal")
    _save(fig, "price_distribution.png")


def median_price_by_month(df):
    m = df.groupby("close_month")["ClosePrice"].median()
    x = pd.PeriodIndex(m.index, freq="M").to_timestamp()
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(x, m.values, color=SERIES, linewidth=2, marker="o", markersize=4)
    ax.yaxis.set_major_formatter(dollars)
    ax.set_ylabel("median close price")
    ax.set_title("Median single-family close price by month")
    for i in (0, len(m) - 1):
        ax.annotate(fmt_dollars(m.iloc[i]), (x[i], m.iloc[i]), textcoords="offset points",
                    xytext=(0, 8), ha="center", color=TEXT, fontsize=9)
    _save(fig, "median_price_by_month.png")


def median_price_by_county(df, top=10):
    counties = df["CountyOrParish"].value_counts().head(top).index
    g = (df[df["CountyOrParish"].isin(counties)].groupby("CountyOrParish")["ClosePrice"]
         .agg(["median", "count"]).sort_values("median"))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(g.index, g["median"], color=SERIES, height=0.6)
    for y, (v, n) in enumerate(zip(g["median"], g["count"])):
        ax.text(v * 1.01, y, f"{fmt_dollars(v)}  ({n:,} sales)", va="center", color=TEXT, fontsize=9)
    ax.xaxis.set_major_formatter(dollars)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, g["median"].max() * 1.35)
    ax.set_xlabel("median close price")
    ax.set_title(f"Median close price, {top} counties with the most sales")
    _save(fig, "median_price_by_county.png")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    df = pd.read_parquet(CLEAN_PATH)
    log = pd.read_csv(LOG_PATH)
    cleaning_steps(log)
    price_distribution(df)
    median_price_by_month(df)
    median_price_by_county(df)


if __name__ == "__main__":
    main()
