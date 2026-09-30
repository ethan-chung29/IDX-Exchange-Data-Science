# IDX-Exchange-Data-Science
Automated Valuation Model (AVM) predicting California residential close prices from CRMLS sold data — IDX Exchange Data Science internship, Fall 2026.

## Goal

Predict `ClosePrice` for California single-family homes from listing features (size, beds/baths, age, lot, location, amenities), using 28 months of closed CRMLS sales (Jan 2024 – Apr 2026). The model targets `log(ClosePrice)`.

## Results so far

| Stage | Rows |
|---|---|
| Raw monthly files (all property types) | 615,707 |
| Residential single-family homes | 309,735 |
| After cleaning | **308,763** (99.7% kept) |

- **Duplicates:** 580 listings appear 2–4 times across monthly files, usually the same sale re-recorded with a corrected close date. The most recently updated record is kept.
- **Price typos:** 22 of the 31 sales above $50M are typos (e.g. $970M closing on a $975k list price). A close/list ratio band of [0.5, 2] removes them.
- **Coordinates:** `latfilled` / `lonfilled` in the `_filled` files are True/False flags meaning the coordinate was imputed. They are kept as `coords_imputed`.
- **Leakage:** per the IDX Best Practices doc, `ListPrice`, `OriginalListPrice`, `DaysOnMarket` and post-listing / post-close dates are excluded as features (`cleaning.LEAKAGE_COLS`). They stay in the clean table only for data-quality checks.

![Rows removed by each cleaning step](docs/figures/cleaning_steps.png)

![Distribution of close prices](docs/figures/price_distribution.png)

![Median close price by month](docs/figures/median_price_by_month.png)

![Median close price by county](docs/figures/median_price_by_county.png)

![Distributions of living area, bedrooms, bathrooms and lot size](docs/figures/feature_distributions.png)

## Repo layout

```
data/                     raw CRMLS CSVs + download script (git-ignored, never commit)
  processed/              sfr_clean.csv (+ .parquet), the cleaned modeling table
docs/
  data_dictionary.csv     every column: description, type, % missing
  cleaning_log.csv        rows removed / values nulled by each cleaning step
  figures/                charts shown above
notebooks/
  01_exploration.ipynb    exploratory data analysis and findings
  02_preprocessing.ipynb  cleaning rules and output checks (split + encoding next)
src/
  cleaning.py             cleaning pipeline (thresholds in RULES)
  make_figures.py         regenerates docs/figures/
```

## How to run

Put the monthly `CRMLSSold*.csv` files in `data/`, then from the repo root:

```bash
pip install -r requirements.txt
python src/cleaning.py        # writes data/processed/sfr_clean.csv (+ .parquet) and docs/cleaning_log.csv
python src/make_figures.py    # regenerates docs/figures/
```

Notebook outputs are stripped on commit (`nbstripout`), so run the notebooks locally to see their charts and tables.

## Next steps

- **Split by time:** test = latest month (2026-04). How many prior months to train on is tuned by experiment.
- **Outliers:** cut ClosePrice at the 0.5th / 99.5th percentile of the training set, then apply the same cutoffs to test.
- **Week 3 preprocessing:** categorical encoding and missing-value flags, all fit on training data only.
- Baseline model on `log_close_price`.
- Confirm with the team how coordinates were imputed in the `_filled` files.
