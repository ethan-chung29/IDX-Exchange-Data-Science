# IDX-Exchange-Data-Science
Automated Valuation Model (AVM) predicting California residential close prices from CRMLS sold data — IDX Exchange Data Science internship, Fall 2026.

## Goal

Predict `ClosePrice` for California single-family homes from listing features (size, beds/baths, age, lot, location, amenities), using 28 months of closed CRMLS sales (Jan 2024 – Apr 2026). The model targets `log(ClosePrice)`.

## Results so far

| Stage | Rows |
|---|---|
| Raw monthly files (all property types) | 615,707 |
| Residential single-family homes | 309,735 |
| After cleaning | **308,525** (99.6% kept) |
| Train (12 months, 2025-03 – 2026-02) / validation (2026-03) / test (2026-04), after outlier cut | 127,862 / 11,403 / 12,331 |

- **Impossible timelines:** 238 sales closed before their listing or purchase-contract date and are removed (`docs/cleaning_log.csv` logs count and % for every step).
- **Duplicates:** 580 listings appear 2–4 times across monthly files, usually the same sale re-recorded with a corrected close date. The most recently updated record is kept.
- **Price typos:** 22 of the 31 sales above $50M are typos (e.g. $970M closing on a $975k list price). A close/list ratio band of [0.5, 2] removes them.
- **Lot units:** 337 lot sizes under 10 "sq ft" (mostly San Diego and Riverside) are acres typed into the square-feet field and are converted; 46 ambiguous lots under 500 sq ft are set to missing.
- **Coordinates:** `latfilled` / `lonfilled` in the `_filled` files are True/False flags meaning the coordinate was imputed. They are kept as `coords_imputed`.
- **Features:** 33 input columns → 119 model features in the final model (size, rooms, age at sale, HOA/month, location, school districts, flooring, seasonality as month sine/cosine). See `docs/leakage_audit.csv`.
- **Leakage:** per the IDX Best Practices doc, `ListPrice`, `OriginalListPrice`, `DaysOnMarket` and post-listing / post-close dates are excluded as features (`cleaning.LEAKAGE_COLS`). They stay in the clean table only for data-quality checks.

### Final model and evaluation (Week 8)

Every model was retuned with a wider search on the 2026-03 validation month (`python src/models.py --tuned`), and all of them are scored on the same 12,328 homes of the 2026-04 test month. **Final model: LightGBM**, chosen on validation MdAPE alone (6.78%, essentially tied with XGBoost at 6.79%). It uses a 24-month window, 511 leaves and an absolute-error objective, which targets the median error directly.

| Test 2026-04 | R² | MdAPE | MAPE | MAE | Within 10% | Backtest MdAPE (2026-01 – 03) | Model file |
|---|---|---|---|---|---|---|---|
| Linear Regression (baseline) | 0.827 | 10.99% | 15.1% | $205k | 46% | 10.8–11.2% | 0.1 MB |
| Decision Tree | 0.845 | 9.32% | 13.8% | $189k | 53% | 9.0–9.5% | 0.7 MB |
| Random Forest | 0.897 | 7.52% | 11.1% | $153k | 61% | 7.4–7.7% | 1,106 MB |
| **LightGBM (final)** | **0.912** | **6.96%** | **10.5%** | **$142k** | **64%** | **6.8–7.0%** | 39 MB |
| XGBoost | 0.905 | 7.15% | 10.8% | $147k | 63% | 6.8–7.2% | 17 MB |

- LightGBM beats the baseline by 4.0 MdAPE points (95% paired-bootstrap interval 3.8–4.3) and the Random Forest by 0.56 (0.42–0.69). It is the lowest of every model in all four test months.
- 38% of test homes are valued within ±5%, 64% within ±10%, 87% within ±20%.
- Most accurate for $575k–$1.1M homes (MdAPE 5.3–6.0%). Luxury homes above $1.67M remain the weakest segment (MdAPE 10.2%, under-valued by 5.0%, vs 15.7% and −9.6% for the baseline).
- **Prediction intervals:** 80% / 90% ranges per band of predicted price, built from validation-month errors, contain 78–81% / 88–90% of test homes in every band (`docs/prediction_intervals.csv`).
- Two Week 8 feature ideas were tested on validation and not adopted: a market-trend feature (worse for every model) and the price level of the 10 nearest sales (cut Linear Regression's MdAPE from 10.8% to 9.3%, but did not help the boosted models). See `docs/feature_set_comparison_week8.csv`.

Full breakdown by price band, county, size, age and the largest misses: `notebooks/06_evaluation.ipynb`; every number in one table: `docs/metrics_summary.csv`.

![MdAPE by test month for every tuned model](docs/figures/model_mdape_by_month.png)

![Final model errors by price quintile](docs/figures/final_model_by_price_band.png)

### Gradient boosting (Week 7)

LightGBM and XGBoost on the `week6` features, with light tuning (training window × tree size × minimum leaf size) on the 2026-03 validation month only. Both picked a 12-month window.

| Test 2026-04 | R² | MdAPE | MAPE | MAE | Within 10% | Backtest MdAPE (2026-01 – 03) | Model file |
|---|---|---|---|---|---|---|---|
| Linear Regression | 0.827 | 11.0% | 15.1% | $205k | 46% | 10.8–11.2% | 0.1 MB |
| Decision Tree | 0.845 | 9.3% | 13.8% | $189k | 53% | 9.0–9.5% | 0.7 MB |
| Random Forest | 0.897 | 7.5% | 11.1% | $153k | 61% | 7.4–7.7% | 1,106 MB |
| **LightGBM** | **0.910** | **7.15%** | 10.7% | **$145k** | 63% | 6.9–7.3% | 19 MB |
| XGBoost | 0.908 | 7.16% | 10.7% | $146k | 63% | 6.9–7.2% | 35 MB |

Both boosted models beat the Random Forest by about 0.37 MdAPE points (95% paired-bootstrap interval 0.20–0.52) and are lower in all four test months. LightGBM and XGBoost were tied; the wider Week 8 search above improved LightGBM further. The biggest gain is on luxury homes: top-quintile under-valuation drops from −7.0% (Random Forest) to −4.0%. Details in `notebooks/05_advanced_models.ipynb`.

### Feature engineering (Week 6)

Added flooring types and a **school-district layer** (spatial join of each home to the [CA School District Areas 2024-25](https://data.ca.gov/dataset/california-school-district-areas-2024-25) boundaries; 99.1% of homes matched). Bed/bath ratio and sq ft per bedroom were tested but made both tree models worse on validation, so they are not in the final `week6` set. District demographic shares are deliberately not used (fair-housing risk).

| Test 2026-04 | Week 5 features | Week 6 features |
|---|---|---|
| Linear Regression | 11.23% MdAPE / R² 0.822 | **10.99% / 0.827** |
| Decision Tree | 9.30% / 0.858 | 9.32% / 0.845 |
| **Random Forest** | 7.60% / 0.896 | **7.52% / 0.897** (lower in all 4 test months) |

Full old-vs-new table: `docs/feature_set_comparison.csv` and `notebooks/04b_feature_engineering.ipynb`.

### Model comparison (Week 5)

Same features, pipeline and procedure for every model: training window and hyperparameters chosen on the 2026-03 validation month, then scored once on 2026-04. All three picked a 24-month window.

| Test 2026-04 | R² | MdAPE | MAPE | MAE | Within 10% | Backtest MdAPE (2026-01 – 03) |
|---|---|---|---|---|---|---|
| Linear Regression | 0.822 | 11.2% | 15.4% | $209k | 45% | 11.1–11.4% |
| Decision Tree | 0.858 | 9.3% | 13.6% | $185k | 53% | 9.1–9.5% |
| **Random Forest** | **0.896** | **7.6%** | **11.2%** | **$154k** | **61%** | **7.5–7.8%** |

Random Forest lowers MdAPE by 3.6 points versus the baseline (95% paired-bootstrap interval 3.4–3.9), and is best in every price quintile and county. Luxury homes remain the hardest (MdAPE 11.3% in the top quintile). Details in `notebooks/04_model_comparison.ipynb`.

### Baseline: Linear Regression (Week 4)

Trained on `log(ClosePrice)`; metrics in dollars. Window chosen on the 2026-03 validation month (24 months), then scored once on 2026-04.

| Test month | R² | MdAPE | MAPE | MAE | Within 10% |
|---|---|---|---|---|---|
| **2026-04 (test)** | **0.822** | **11.2%** | 15.4% | $209k | 45% |
| 2026-03 (backtest) | 0.817 | 11.1% | 15.1% | $201k | 46% |
| 2026-02 (backtest) | 0.803 | 11.4% | 15.4% | $208k | 45% |
| 2026-01 (backtest) | 0.788 | 11.3% | 15.7% | $200k | 45% |

The model over-values entry-level homes (median +4.9% in the lowest price quintile) and under-values luxury homes (−10.6%, MdAPE 15.9% in the top quintile). Full breakdown in `notebooks/03_baseline_model.ipynb` and `docs/metrics_baseline_linear_regression.csv`.

![Rows removed by each cleaning step](docs/figures/cleaning_steps.png)

![Distribution of close prices](docs/figures/price_distribution.png)

![Median close price by month](docs/figures/median_price_by_month.png)

![Median close price by county](docs/figures/median_price_by_county.png)

![Distributions of living area, bedrooms, bathrooms and lot size](docs/figures/feature_distributions.png)

## Repo layout

```
data/                     raw CRMLS CSVs + download script (git-ignored, never commit)
  external/               school-district boundary shapefile (download command below)
  processed/              sfr_clean.csv (+ .parquet), the cleaned modeling table
docs/
  data_dictionary.csv     every column: description, type, % missing
  cleaning_log.csv        rows removed / values nulled by each cleaning step
  split_summary.csv       rows and price cutoffs for each training-window length
  leakage_audit.csv       every column: used as a feature or excluded, and why
  metrics_<model>__<features>.csv  validation, test and backtest metrics per model and feature set
  feature_set_comparison.csv       old vs new feature sets (Week 6)
  feature_set_comparison_week8.csv market-trend and nearest-sales features, validation only (Week 8)
  metrics_summary.csv     every tuned model: validation, test and backtest, by price band and county (Week 8)
  prediction_intervals.csv  final model's 80% / 90% price ranges per predicted-price band, with test coverage
  figures/                charts shown above
models/                   trained model files (git-ignored; rebuild with python src/models.py)
notebooks/
  01_exploration.ipynb    exploratory data analysis and findings
  02_preprocessing.ipynb  cleaning rules, train/validation/test split, preprocessing pipeline
  03_baseline_model.ipynb Linear Regression baseline, backtest and error breakdown
  04_model_comparison.ipynb  Linear Regression vs Decision Tree vs Random Forest (Week 5 features)
  04b_feature_engineering.ipynb  new features, school-district layer, old vs new feature sets
  05_advanced_models.ipynb   LightGBM and XGBoost with light tuning, compared with every earlier model
  06_evaluation.ipynb     final evaluation: model choice, price bands, regions, segments, largest misses, intervals
src/
  cleaning.py             cleaning pipeline (thresholds in RULES)
  preprocessing.py        time-based train/validation/test split + train-only ClosePrice percentile cut
  features.py             feature sets (week5 / week6 ...), leakage audit, scikit-learn preprocessing pipeline
  geo.py                  school-district spatial join (CA School District Areas 2024-25)
  models.py               model registry; window + hyperparameter selection on validation, test, rolling backtest
  evaluation.py           R², MAPE, MdAPE, MAE, RMSE and breakdowns by price band / county
  evaluate_final.py       picks the final model on validation, writes metrics_summary.csv, intervals and figures
  make_figures.py         regenerates docs/figures/
```

## How to run

Put the monthly `CRMLSSold*.csv` files in `data/`, then from the repo root:

```bash
brew install libomp           # macOS only: OpenMP runtime needed by LightGBM and XGBoost
pip install -r requirements.txt
python src/cleaning.py        # writes data/processed/sfr_clean.csv (+ .parquet) and docs/cleaning_log.csv
python src/make_figures.py    # regenerates docs/figures/
python src/preprocessing.py --train-months 12   # writes data/processed/{train,val,test}.csv and docs/split_summary.csv
mkdir -p data/external && curl -L -o data/external/ca_school_district_areas_2024_25.zip \
  "https://gis.data.ca.gov/api/download/v1/items/b0e3b936426a47ce9d9a2e77e2bb86cc/shapefile?layers=0"
python src/geo.py             # school-district join -> data/processed/school_districts.parquet
python src/features.py        # writes docs/leakage_audit.csv
python src/models.py          # trains all five models on the week6 features (~35 min), writes docs/metrics_*.csv and models/
python src/models.py lightgbm xgboost                 # only the Week 7 boosting models (~25 min)
python src/models.py --tuned  # Week 8 wider search for every model (~2 hours; keep the Mac awake)
python src/evaluate_final.py  # final model, docs/metrics_summary.csv, docs/prediction_intervals.csv, figures
python src/models.py --features week5 random_forest   # a single model / another feature set
```

Notebook outputs are stripped on commit (`nbstripout`), so run the notebooks locally to see their charts and tables.

## Data snapshot

- **Source:** CRMLS closed sales, 28 monthly files (`CRMLSSold202401` – `CRMLSSold202604`), downloaded 2026-09-29 through the IDX proxy (`data/crmls_sold.py`; key from the `IDX_PROXY_KEY` environment variable).
- **Coverage:** sales closed January 2024 – April 2026; California single-family homes only.
- **School districts:** CA School District Areas 2024-25 (data.ca.gov), downloaded 2026-10-08.
- Seeds are fixed (`random_state=42`) and every package version is pinned in `requirements.txt`.

## Limitations

- **The 2026-03 backtest month is also the validation month**, so it is not an independent check; 2026-01 and 2026-02 are.
- **Cleaning rules that use the sale price** (close/list ratio, price per sq ft, minimum price) also apply to the test month. They remove 359 sales (0.12%), mostly typos, so error on unfiltered data would be slightly higher.
- **One market, one period:** accuracy outside CRMLS coverage or in a sharply changing market is unknown.
- **No condition or renovation data.** The worst 5% of predictions are mostly over-valued older homes, likely fixer-uppers.
- **School-district boundaries are the 2024-25 snapshot**, applied to 2024–2026 sales; district demographics are deliberately excluded (fair-housing risk).
- **Coordinates in the `_filled` files** were imputed by an unknown method (open question for the team).

## Next steps

- **Split + outlier cut (done):** test = 2026-04, validation = 2026-03; ClosePrice cut at the training set's 0.5th / 99.5th percentile and applied to validation and test. Window lengths compared in `docs/split_summary.csv`.
- **Preprocessing pipeline (done):** median imputation + missing-value flags, one-hot for county/levels, cross-fitted target encoding for ZIP/city/MLS area/school district, scaling; all fit on training data only (`src/features.py`).
- **Training window:** picked per model on the **validation** month (24 months for the baseline); test is used once at the end.
- **Baseline (done):** Linear Regression, see results above.
- **Week 5 (done):** Decision Tree and Random Forest, see the comparison above.
- **Week 6 (done):** flooring types + school-district layer, see above.
- **Week 7 (done):** LightGBM and XGBoost with light tuning, see above.
- **Week 8 (done):** wider tuning, final evaluation and prediction intervals; LightGBM is the final model.
- **Week 9 (optional):** Streamlit app: saved LightGBM model + school-district lookup + 80% price range.
- Confirm with the team how coordinates were imputed in the `_filled` files.
