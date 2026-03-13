# CLAUDE.md — divvy-bike-rebalancing

This file instructs Claude Code on how to set up and maintain this project.
Read this file before doing anything else in this repository.

---

## Project Overview

This project forecasts per-station bike inventory starting values for the Divvy
bike-sharing system and uses those forecasts to optimize overnight rebalancing
via minimum-cost network flow. The original work was done in a single Google
Colab notebook and is being migrated here into a clean, reproducible structure.

**Stack:** Python, DuckDB, LightGBM, NetworkX, Pandas, Matplotlib

---

## Folder Structure to Create

If any of these folders do not exist, create them (with a `.gitkeep` so they
appear in git):

```
data/
  raw/          # original parquet file goes here — gitignored
  processed/    # intermediate outputs from feature engineering
notebooks/      # numbered Jupyter notebooks, one per pipeline stage
src/            # reusable Python modules extracted from notebooks
reports/
  figures/      # saved matplotlib plots
```

---

## Files to Create on Setup

### `requirements.txt`
```
duckdb
pandas
numpy
lightgbm
scikit-learn
networkx
matplotlib
jupyter
```

### `notebooks/` — create these four empty notebooks with the section headers below

| File | Purpose |
|------|---------|
| `01_eda.ipynb` | Connect DuckDB, schema inspection, null counts, categorical distributions |
| `02_feature_engineering.ipynb` | Hybrid progressive densification, cumulative net flow, inventory bounds, NA handling, lag features, 7-day rolling averages, train/test split |
| `03_modeling.ipynb` | LightGBM midpoint model, coverage/efficiency metrics, station-level KPI scatter, cumulative KPI trend over time |
| `04_rebalancing_optimization.ipynb` | Station geometry + KNN edge graph, min-cost flow rebalancing pipeline, post-OR KPI evaluation, CSV exports |

### `src/` — create these module stubs with docstrings only (no code yet)

| File | Contains |
|------|---------|
| `features.py` | `build_station_day_calendar()`, `compute_inventory_bounds()`, `add_lag_features()`, `add_rolling_features()` |
| `models.py` | `train_lgbm()`, `evaluate_coverage()`, `coverage_summary()` |
| `rebalancing.py` | `haversine_m()`, `build_knn_edges()`, `adjust_to_fixed_fleet_int()`, `run_rebalancing_pipeline()` |
| `utils.py` | shared helpers e.g. `connect_duckdb()`, `load_parquet_view()`, `get_data_path()` |

---

## Data Storage & Path Convention

The raw data file (`divvy.parquet`) is stored locally on the user's machine and
is gitignored. It is never committed to the repository.

The expected location is `data/raw/divvy.parquet` relative to the project root,
but since the file lives locally and may be in a different location on different
machines, **every notebook prompts the user for the path at runtime** using the
pattern in `src/utils.py`.

### Runtime path prompt pattern

`src/utils.py` should implement a `get_data_path()` function like this:

```python
import os

def get_data_path(default: str = "../data/raw/divvy.parquet") -> str:
    """
    Prompt the user for the path to the raw parquet file.
    Press Enter to accept the default.
    """
    path = input(f"Enter path to divvy.parquet [{default}]: ").strip()
    if not path:
        path = default
    if not os.path.exists(path):
        raise FileNotFoundError(f"No file found at: {path}")
    return path
```

Every notebook's first code cell should call this before connecting DuckDB:

```python
from src.utils import get_data_path, connect_duckdb
file_path = get_data_path()
con = connect_duckdb(file_path)
```

### Path replacements from original Colab code

| Old (Colab) | New (local) |
|-------------|-------------|
| `/content/drive/MyDrive/Colab_datasets/divvy.parquet` | result of `get_data_path()` |
| `/content/drive/MyDrive/Divvy_exports/` | `../reports/` |
| `drive.mount('/content/drive')` | remove entirely — not needed locally |

All paths in `src/` modules must be passed in as arguments, never hardcoded.

---

## Notebook Structure Convention

Each notebook should follow this pattern:

1. **Markdown cell** — section title and 2–3 sentence description of what this
   notebook does and what it assumes as input
2. **Markdown cell** — Assumptions & dependencies (what data/views must exist,
   what prior notebook must have been run)
3. Code cells with a markdown cell before each logical block explaining *why*,
   not just *what*
4. **Markdown cell at the end** — Summary of outputs produced and what the next
   notebook expects

Do not use `# ====` banner comments inside code cells. Those become markdown cells.

---

## Variable Naming Convention

Use full descriptive names throughout. Key renames from the original Colab code:

| Original | Renamed |
|----------|---------|
| `min_cumflow` | `min_cumulative_flow` |
| `max_cumflow` | `max_cumulative_flow` |

Apply this consistently across all notebooks and `src/` modules.

---

## Key Concepts & Pipeline Assumptions

These must be accurately reflected in notebook markdown cells and `src/` docstrings.
Every implementation detail below is explicit and must be followed exactly —
do not infer or substitute alternatives.

### The Core Problem

Each station has an inventory curve for each day. It starts at some unknown value
at the beginning of the day, then moves up and down as bikes arrive and depart.
The shape of the curve is fully determined by observed trips. The only unknown is
the y-intercept — where the curve starts.

There exists a range of valid starting values `[L, U]` such that if the curve
starts anywhere in that range, it never goes below 0 or above `station_capacity_day`
at any point during the day. The goal is to predict that starting value and use
it to set each station's inventory through overnight rebalancing.

---

### Stage 1: Station×Day Calendar (Hybrid Progressive Densification)

The calendar is built by giving each station a row for every day from its **own
first observed date** (not the global dataset minimum) up to the dataset-wide
maximum date. A station's first observed date is the minimum of its first
appearance as `from_station_id` in `starttime` or as `to_station_id` in `stoptime`.

This avoids inflating zero-flow days for stations that didn't exist yet, which
would corrupt rolling averages and lag features.

---

### Stage 2: Station Capacity Derivation

`station_capacity_day` is derived per station per day as follows:

1. From departure records: `MAX(dpcapacity_start)` grouped by `from_station_id`
   and `DATE(starttime)` → call this `cap_s`
2. From arrival records: `MAX(dpcapacity_end)` grouped by `to_station_id` and
   `DATE(stoptime)` → call this `cap_e`
3. These two are joined with a **FULL OUTER JOIN** on `station_id` and `trip_date`
4. The final capacity is `GREATEST(COALESCE(cap_s, 0), COALESCE(cap_e, 0))`

Taking the GREATEST of both sources ensures we capture the most complete capacity
reading regardless of whether the station appeared as a departure or arrival on
that day. Cast the result to DOUBLE.

After merging onto the station×day calendar, **forward fill `station_capacity_day`
within each station group**. Capacity is assumed constant until a new observation
says otherwise. Physical dock expansions are rare.

---

### Stage 3: Cumulative Net Flow

For each station×day, all trips are grouped into **hourly buckets** using
`EXTRACT(hour FROM starttime)` for departures and `EXTRACT(hour FROM stoptime)`
for arrivals — never individual trip timestamps.

Departures and arrivals per hour are joined with a **FULL OUTER JOIN** on
`station_id`, `trip_date`, and `hour`. This is critical — a LEFT JOIN on
departures would silently drop hours where only arrivals occurred, producing
wrong cumulative flow values.

For each hour:
```
hourly_net_flow = trips_arrived - trips_departed
```

The cumulative sum of `hourly_net_flow` across hours within each station×day
gives the shape of the inventory curve relative to its starting point. From the
**cumulative sum column** (not the raw hourly flow) we extract:
- `min_cumulative_flow` — the minimum value of the cumulative sum across all hours
- `max_cumulative_flow` — the maximum value of the cumulative sum across all hours

Days with zero trips get `min_cumulative_flow = 0` and `max_cumulative_flow = 0`.

**Critical:** Always use hourly bucketing, never individual trip timestamps. Using
trip-level timestamps produces a finer-grained cumulative sum with different
min/max values, which changes `[L, U]` and breaks consistency with the original
model training.

---

### Stage 4: Inventory Bounds [L, U]

Derived from the constraint that inventory can never go below 0 or above
`station_capacity_day` at any point during the day:

```
L = clip(-min_cumulative_flow, 0, station_capacity_day)
U = clip(station_capacity_day - max_cumulative_flow, 0, station_capacity_day)
```

- **L:** If the curve drops by `-min_cumulative_flow` during the day, the starting
  inventory must be at least that large or the curve would go negative.
- **U:** If the curve rises by `max_cumulative_flow` during the day, the starting
  inventory must be at most `station_capacity_day - max_cumulative_flow` or the
  curve would exceed capacity.

**Inversion repair:** If `U < L` after this computation, set `L = 0` and
`U = station_capacity_day`. This is a conservative fallback that treats the full
capacity range as feasible rather than discarding the row.

**Prediction target:**
```
s_true = (L + U) / 2
```
The model predicts the midpoint of `[L, U]`. A prediction is **covered** if the
rounded predicted value `s_hat_r` falls within `[L, U]`.

---

### Stage 5: NA Handling and Forward Fills

These decisions must be preserved exactly. Each fill has a specific rationale:

**`min_cumulative_flow` and `max_cumulative_flow` — fill with 0:**
A zero-trip day means the curve never moved. Filling with 0 gives `L = 0` and
`U = station_capacity_day` — the full range is feasible when we have no flow data.

**`temperature` and `events` — forward fill within station group:**
Weather is only recorded on days with trips. Forward fill carries the last known
reading forward. Weather doesn't become unknown just because nobody rode a bike.

**Rolling 7-day averages — 3-tier fallback for `temperature_roll7`,
`min_start_inventory_roll7`, `max_start_inventory_roll7`:**
1. Station-level 7-day rolling mean — primary
2. City-wide 7-day rolling mean by date — fallback. All stations are in the same
   city (Chicago), so city-wide averages are geographically meaningful proxies.
3. Global dataset mean — last resort. Reasonable for the same reason — all
   stations are in the same city, so temperature variation across the dataset
   is limited.

**`trips_departed_roll7` and `trips_arrived_roll7` — fill with 0:**
No trips in the past 7 days means zero activity. Zero is the correct value.

**Lag features — drop first day per station:**
After shifting 1 day within each station group, the first day per station has
all `_prev` columns as NaN. Drop these rows entirely — there is no valid prior
context for a station's first day, and filling with anything would introduce
fabricated data. One row lost per station is negligible.

---

### Stage 6: Feature Engineering

All features use only information available the day before the prediction date.
Lag features are computed first by shifting 1 day within each station group.
Rolling features are then computed on the raw (unshifted) columns. Do not apply
a shift before computing rolling features.

**Lag features (`_prev`) — shift 1 day within station group:**
- `min_start_inventory_prev` — from `L` (renamed `min_start_inventory`)
- `max_start_inventory_prev` — from `U` (renamed `max_start_inventory`)
- `station_capacity_day_prev`
- `temperature_prev`
- `events_prev`
- `trips_departed_prev`
- `trips_arrived_prev`

**Rolling 7-day averages (`_roll7`) — computed on raw columns, not shifted:**
- `trips_departed_roll7`
- `trips_arrived_roll7`
- `temperature_roll7`
- `min_start_inventory_roll7` — rolling mean of `min_start_inventory`
- `max_start_inventory_roll7` — rolling mean of `max_start_inventory`

**Station identity:**
- `station_id` as a categorical feature

Total: 13 features.

---

### Stage 7: Train/Test Split and Model

- Train: `trip_date < 2017-10-01`
- Test: `trip_date >= 2017-10-01`
- After splitting, restrict training set to only stations that appear in the
  test set. This prevents LightGBM categorical encoding issues and keeps training
  focused on stations we actually need to predict.

**Before fitting, cast these columns to `category` dtype:**
```python
df["station_id"] = df["station_id"].astype("category")
df["events_prev"] = df["events_prev"].astype("category")
```
Pass `categorical_feature=["station_id", "events_prev"]` to `model.fit()`.

**LightGBM hyperparameters — use exactly these, do not use defaults:**
```python
LGBMRegressor(
    n_estimators=250,
    learning_rate=0.08,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

**Predictions:**
- `s_hat` — raw continuous prediction from `model.predict()`
- `s_hat_r` — `s_hat` rounded to the nearest integer

**Primary metric — coverage (uses `s_hat_r`):**
```
covered = 1 if s_hat_r >= L and s_hat_r <= U else 0
```

**Secondary metric — conditional efficiency (uses `s_hat_r`, covered rows only):**
```
efficiency = 1 - |s_hat_r - s_true| / (U - L)
```
Uncovered rows are NaN for efficiency, not 0.

**Tertiary metric — RMSE (uses unrounded `s_hat`, not `s_hat_r`):**
```python
rmse = np.sqrt(mean_squared_error(df["s_true"], df["s_hat"]))
```

**KPI scatter cutoff date:** The scatter plot has no fixed cutoff date. It uses
the full test period through the last date in the data. In the Plotly dashboard,
the slider minimum is the first test date and the slider maximum is the last date
in `postOR_station_kpi.csv` — do not hardcode any artificial cap like 2017-11-30.

---

### Stage 8: Rebalancing Optimization

**Station geometry:**
Station coordinates are derived as follows — use both departure and arrival
records, not just one:
```sql
SELECT
  COALESCE(from_station_id, to_station_id) AS station_id,
  AVG(COALESCE(latitude_start, latitude_end)) AS lat,
  AVG(COALESCE(longitude_start, longitude_end)) AS lon
FROM divvy
WHERE COALESCE(latitude_start, latitude_end) IS NOT NULL
GROUP BY 1
```

**Fixed fleet assumption:** Total bikes across all stations is held constant.
Rebalancing redistributes bikes; it does not add or remove them.

**Fleet size:** Computed as the integer sum of `s_hat_r` across all stations
on the first test day:
```python
fleet_size = int(pred_df.loc[pred_df["trip_date"] == dates[0], "s_hat_r"].sum())
```

**Fleet adjustment:** For each day, clip `s_hat_r` to `[0, station_capacity_day]`,
then randomly add or remove single bikes from eligible stations until the total
equals `fleet_size`. Adding: choose randomly from stations below capacity.
Removing: choose randomly from stations above 0.

**Min-cost flow network:**
- Nodes: stations
- Edges: each station connected to its k=8 nearest neighbors by Haversine distance
  using the station coordinates derived above
- Edge weights: distance in meters (integer rounded)
- Edge capacity: effectively infinite (use a large constant e.g. 10^9)
- Node supply/demand: `s_target_tomorrow - s_target_today`

**Post-OR evaluation:** Same coverage and conditional efficiency metrics as
Stage 7 but using `s_target` instead of `s_hat_r`. RMSE is not recomputed
post-OR.

---

## .gitignore Additions

Make sure these are in `.gitignore` if not already present:

```
data/raw/
*.parquet
*.csv
reports/*.csv
.ipynb_checkpoints/
__pycache__/
*.pyc
.env
```

---

## What NOT to Do

- Do not commit the raw parquet file — it is too large for GitHub
- Do not hardcode file paths in `src/` modules
- Do not put the Google Drive mount cell in any notebook — this is a local project now
- Do not duplicate logic between notebooks and `src/` — notebooks import from `src/`
  once functions are stable
- Do not use individual trip timestamps for cumulative flow — always use hourly buckets
- Do not use a LEFT JOIN when computing hourly net flow — use a FULL OUTER JOIN
- Do not apply a shift before computing rolling features — lags and rolling are
  computed independently on the raw columns
- Do not use LightGBM default hyperparameters — use the exact values specified above
- Do not use `s_hat_r` for RMSE — use unrounded `s_hat`
- Do not use short variable names like `min_cumflow` — use `min_cumulative_flow`
