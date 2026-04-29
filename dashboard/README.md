# FortuneTellers Streamlit Dashboard

A two-page Streamlit interface layered on top of the existing
`fortunetellers/` package (data cleaning, weekly demand aggregation,
product-level feature engineering, product clustering, cluster-wise model
training, cluster-specific best-model selection, and the forecasting
pipeline).

- **AI Forecasting Chatbot** — `app.py`. Type a natural-language request such as
  *"Forecast product 85123A in the UK for the next 12 weeks"*. The bot
  parses the StockCode, country and horizon, looks up the product's cluster,
  reads the cluster's saved best model from the artifacts, and runs the
  existing forecasting pipeline to return a chart, table and structured
  business briefing.
- **Model / Cluster Comparison Dashboard** — `dashboard.py` plus
  `pages/1_Model_Comparison.py`. Pick a SKU, country and horizon and explore
  the cluster, the saved model selection, the candidate-model comparison and
  the forecast.

The existing `fortunetellers/` package is **imported as-is** and never
modified.

---

## 1. Installation

### 1.1. Project layout

The project root is `IEOR4578_Assignment-Deliverable-main/`. The dashboard
folder lives next to the existing `fortunetellers/` package and its
artifacts:

```
IEOR4578_Assignment-Deliverable-main/
├── fortunetellers/                 # existing package (do not modify)
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── modeling.py
│   ├── pipeline.py
│   └── agent.py
├── fortunetellers_artifacts/       # training / clustering outputs
│   ├── clustering/
│   │   ├── product_features_clustered.csv
│   │   └── cluster_metadata.json
│   ├── modeling/
│   │   ├── cluster_model_selection_summary.csv
│   │   ├── cluster_candidate_metrics.csv
│   │   └── best_model_params.json
│   └── pipeline/                   # cached forecast JSONs
├── data/
│   └── raw/
│       └── online_retail_II.xlsx   # UCI Online Retail II source
├── requirements.txt                # base project dependencies
└── dashboard/                      # ← this folder
    ├── app.py
    ├── dashboard.py
    ├── agent_tools.py
    ├── pages/
    │   └── 1_Model_Comparison.py
    ├── requirements.txt
    └── README.md (this file)
```

### 1.2. Create a virtual environment (recommended)

```bash
cd IEOR4578_Assignment-Deliverable-main
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

### 1.3. Install dependencies

The dashboard shares dependencies with the existing package, so install
both requirement files:

```bash
pip install -r requirements.txt
pip install -r dashboard/requirements.txt
```

`dashboard/requirements.txt` ships:

- `streamlit`, `plotly` — UI and charts
- `pandas`, `numpy`, `scipy`, `scikit-learn` — data and modelling
- `lightgbm` — for `LGBM_Tuned` / `LGBM_Default` inference
- `openpyxl` — to read `online_retail_II.xlsx`
- `holidays` — for calendar features
- `joblib` — model serialisation (where needed)
- `anthropic` (optional) — used only to polish chatbot prose; no model
  selection happens through the LLM.

### 1.4. Provide the source data (one-time)

`fortunetellers.pipeline.ForecastingPipeline` needs cleaned transactions to
run. Any **one** of the following is enough:

1. **Recommended.** Place the original Excel at `data/raw/online_retail_II.xlsx`.
2. Or set the environment variable: `export FORTUNETELLERS_RAW_EXCEL=/abs/path/online_retail_II.xlsx`.
3. Or pre-build the cleaned CSVs at
   `fortunetellers_artifacts/processed/total_retail_cleaned.csv` and
   `fortunetellers_artifacts/processed/all_products_cleaned.csv`.

If none of those is present the dashboard still launches: it falls back to
saved forecast JSONs (e.g. `fortunetellers_artifacts/pipeline/forecast_85123A_United_Kingdom.json`)
and shows a clear, actionable banner explaining what's missing.

---

## 2. How to run

### 2.1. Launch chatbot + dashboard together (typical)

```bash
cd IEOR4578_Assignment-Deliverable-main
streamlit run dashboard/app.py
```

Streamlit auto-discovers `dashboard/pages/`, so:

- Sidebar item 1 → **AI Forecasting Chatbot** (`app.py`)
- Sidebar item 2 → **Model Comparison** (`pages/1_Model_Comparison.py`,
  which calls `dashboard.render()`)

The browser usually opens at `http://localhost:8501` automatically.

### 2.2. Launch only the Model Comparison page

```bash
streamlit run dashboard/dashboard.py
```

### 2.3. What happens on the first turn

The very first chatbot message ("Forecast product 85123A in the UK for the
next 12 weeks") triggers a one-off ~30-60 second spinner while the
pipeline:

1. Reads `data/raw/online_retail_II.xlsx` and writes
   `fortunetellers_artifacts/processed/total_retail_cleaned.csv` and
   `all_products_cleaned.csv` (one-time).
2. Builds the panel for the product's cluster and trains the cluster's
   saved best model.
3. Generates a 12-week forecast.

Subsequent turns reuse the cached cleaned CSV and the cached cluster
model, so they return in seconds. Asking about a product in a different
cluster triggers a short re-train for that cluster only.

### 2.4. Chatbot sidebar options

| Option | Description |
| --- | --- |
| LLM API key (optional) | Anthropic API key. When provided, Claude Haiku rewrites the briefing prose only. **Model selection never goes through the LLM** — artifacts always decide. Leave blank to use the deterministic regex parser. |
| Default country | Used when the query does not name a country (default: `United Kingdom`). |
| Default forecast horizon (weeks) | Used when the query does not name a horizon (default: 12). |
| Clear chat | Reset the conversation. |

### 2.5. Dashboard sidebar options

- **Product / SKU** — every StockCode in `product_features_clustered.csv`.
- **Country** — countries that have rows for the selected product (with a
  resilient fallback if the live dataset isn't loaded).
- **Cluster filter** — restrict the candidate-model table to one cluster,
  or follow the selected product's cluster.
- **Forecast horizon** — in weeks.
- **Candidate model filter** — narrow the comparison to specific models
  (`RF_Default`, `LGBM_Tuned`, `DeepAR`, …).
- **Run forecast** — required to populate section 4 (forecast chart).

---

## 3. End-to-end mechanism — from a sentence to a forecast

Each step below corresponds to one function in
`dashboard/agent_tools.py`. The Streamlit UI calls only these functions;
modelling internals stay inside `fortunetellers/`.

### Step 0. User input

Example: `Forecast product 85123A in the UK for the next 12 weeks`.

### Step 1. Natural-language parsing — `parse_forecast_request(query, default_country, default_horizon)`

Deterministic, regex-based — works without any LLM.

- **Product ID.** First tries explicit prefixes
  (`product|sku|stockcode|stock\s*code|item\s*[:#-]?\s*([A-Za-z0-9]+)`).
  If nothing matches, it scans 4–12 character alphanumeric tokens and
  keeps only those that exist in
  `product_features_clustered.csv`. Returns a friendly error dict if it
  cannot find anything.
- **Country.** Aliases dictionary (`uk → United Kingdom`,
  `all/global/worldwide → ALL`, …) applied longest-first; falls back to
  `default_country`.
- **Horizon.** `\d+\s*week` is taken literally; `\d+\s*month` is
  converted to weeks (×4); otherwise `default_horizon` (12).

Result: `{ok: True, product_id: "85123A", country: "United Kingdom", horizon_weeks: 12}`.

### Step 2. Cluster lookup — `lookup_product_cluster(product_id)`

`fortunetellers_artifacts/clustering/product_features_clustered.csv` is
loaded once (LRU-cached) and indexed by StockCode. Two columns drive the
rest of the pipeline:

- `cluster` — integer (e.g. `2`).
- `cluster_label` — string (e.g. `Steady regulars`).

Several representative features (`mean_weekly_sales`, `cv`,
`pct_zero_weeks`, `q4_pct`, `cancel_rate`, …) are returned alongside so
section 1 of the dashboard can show them. The cluster was assigned at
training time (KMeans + rule-based handling for sporadic / intermittent
SKUs) and is **not** recomputed at request time. The mapping comes from
`cluster_metadata.json`:

| cluster | label |
| ---: | --- |
| -2 | Truly sporadic |
| -1 | Intermittent (Croston) |
| 0 | High cancellation risk |
| 1 | Volatile mid-range |
| 2 | Steady regulars |

### Step 3. Best-model lookup — `get_best_model(product_id)`

This is where **the LLM never gets a vote.**

1. Filter `cluster_model_selection_summary.csv` by the cluster id to
   read `selected_model`, `valid_mape_selected`, `test_mape_selected`,
   `test_mape_lgbm_baseline`, and the `n_train/valid/test` counts.
2. Find the matching `cluster_configs[]` entry in
   `best_model_params.json` to get the saved hyper-parameters.
3. Pull every candidate row for the cluster from
   `cluster_candidate_metrics.csv` and sort by validation MAPE so the UI
   can render a leaderboard.

Example return for `85123A` (cluster 2):

```json
{
  "selected_model": "RF_Default",
  "selected_params": {"n_estimators": 300, "max_depth": 20, "...": "..."},
  "test_mape_selected": 75.18,
  "valid_mape_selected": 82.63,
  "candidate_models": [
    {"candidate": "RF_Default",          "valid_mape": 82.63, "n_valid": 5202},
    {"candidate": "RF_C2_BEST",          "valid_mape": 82.66, "n_valid": 5202},
    {"candidate": "LGBM_Tuned",          "valid_mape": 86.53, "n_valid": 5202},
    {"candidate": "LGBM_Default",        "valid_mape": 88.87, "n_valid": 5202},
    {"candidate": "AggregateMLP_Disagg", "valid_mape": 618.29, "n_valid": 5202}
  ],
  "model_explanation": "Random Forest with the project's default hyper-parameters..."
}
```

The Model Comparison page renders this leaderboard in section 5 and
highlights the saved best-model row in amber.

### Step 4. Forecasting — `run_forecast(product_id, country, horizon_weeks)`

Internally this calls
`fortunetellers.pipeline.ForecastingPipeline.forecast_product(...)` —
the existing pipeline, **not a re-implementation**.

The pipeline:

1. Loads the cleaned transaction table once (process-wide singleton).
2. Confirms the product → cluster mapping (same as step 2).
3. Trains the cluster's saved best model:
   - `RF_Default` / `RF_C2_BEST` → `RandomForestRegressor` over
     lag / rolling / calendar / product features.
   - `LGBM_Default` / `LGBM_Tuned` → LightGBM over the same features.
   - `AggregateMLP_Disagg` → small MLP forecasts the cluster's aggregate
     demand and disaggregates to the SKU using its recent share.
   - `CrostonSBA` → Syntetos-Boylan approximation for intermittent demand.
4. Builds a future panel for the product (with the country filter
   applied) and recursively predicts `horizon` weeks ahead.
5. Returns a `ForecastOutput` dataclass.

`agent_tools.run_forecast` adds:

- `forecast_horizon_weeks` and `forecast_summary` (mean / peak / total).
- A graceful fallback: if the pipeline cannot initialise (raw data
  missing) but
  `fortunetellers_artifacts/pipeline/forecast_<pid>_<country>.json`
  exists, the cached payload is hydrated and a `warning` field is set so
  the UI can show an amber banner.
- An in-memory store keyed by product so `get_last_forecast(product_id)`
  can return the most recent forecast for that SKU.

### Step 5. Building the assistant payload — `build_assistant_payload(parsed, forecast, best_model)`

Combines parsed query + forecast + selected-model rationale into a
single dict the chatbot UI consumes:

- `headline` — e.g. *"12-week forecast for product 85123A in United Kingdom"*.
- `rationale` — 4–5 sentences explaining the cluster, the artifact-driven
  model choice, the test MAPE, and which model was used.
- `mape_explanation` — plain-English MAPE primer.
- `fields` — flat dict for the chat bubble: product, description,
  country, cluster, cluster_label, selected_model, test_mape_selected,
  forecast horizon, mean / peak / total forecast, warning_flag,
  recommendation.

### Step 6. UI rendering — `app.py`

- The chat bubble shows the markdown briefing built from `fields`,
  followed by the rationale and the MAPE primer.
- Five metric cards: Mean / Peak / Total forecast, Selected MAPE, Recent
  12-week average sales.
- Two Plotly charts: (a) Recent actuals + Forecast with a dotted slate
  bridge between the last actual and the first forecast week, and (b) a
  bar chart of recent actual sales.
- Forecast table with `Week`, `Week start`, `Forecast sales`.
- Optional: if an Anthropic API key is supplied, Claude Haiku rewrites
  only the prose. The system prompt forbids it from changing any number,
  model name or cluster label.

### Why is there a dotted line between actual and forecast?

`recent_12_weeks` comes from `groupby("Week").sum()` on the product's
sales, which **drops weeks with zero sales**. The line therefore ends at
the last week the product actually sold. The forecast, however, starts
from the dataset's global last week + 1. For intermittent products there
can be a few empty weeks in between. The dotted slate-gray "Bridge
(no-sale weeks)" trace makes that explicit so the chart looks continuous
without misrepresenting reality.

---

## 4. Model selection — at a glance

| Stage | LLM influence | Artifact influence |
| --- | --- | --- |
| Natural language → product / country / horizon | None (regex) | None |
| Product → cluster | None | `product_features_clustered.csv` |
| Cluster → model to use | **None** | `best_model_params.json` + `cluster_model_selection_summary.csv` |
| Candidate model leaderboard | None | `cluster_candidate_metrics.csv` |
| Forecast numbers | None | `ForecastingPipeline.forecast_product` |
| Final reply prose | Optional (rewrite only) | — |

In short, **the model and the numbers come 100% from the saved
artifacts**; the LLM, when present, only smooths the language.

---

## 5. Common errors and how to fix them

| Symptom | Cause | Fix |
| --- | --- | --- |
| `Cannot run the forecasting pipeline because the cleaned transaction data is missing AND the raw Excel is missing.` | Neither the cleaned CSVs nor the raw Excel are on disk. | Place the Excel at `data/raw/online_retail_II.xlsx` or set `FORTUNETELLERS_RAW_EXCEL`. |
| `Product '...' was not found in product_features_clustered.csv` | StockCode does not appear in the clustering artifact. | Double-check the StockCode against the `StockCode` column of `product_features_clustered.csv`. |
| `No sales data for product X in Y. Available countries: ...` | The product never sold in that country. | Pick one of the countries listed in the message, or use `ALL`. |
| First Streamlit call takes 30–60 seconds | Excel cleaning + first cluster training. | Normal. Subsequent calls are fast. |
| `ModuleNotFoundError: lightgbm/holidays/openpyxl` | `dashboard/requirements.txt` not installed. | `pip install -r dashboard/requirements.txt`. |
| Visible gap between actual and forecast | The product is intermittent — recent zero weeks aren't shown in the actual line. | The dotted slate bridge in the chart already communicates this. See section 3 above. |

---

## 6. Changing artifact paths

Only one location needs editing in `dashboard/agent_tools.py`:

```python
def _project_paths() -> ProjectPaths:
    return ProjectPaths(artifact_root=_PROJECT_ROOT / "fortunetellers_artifacts")
```

Point `artifact_root` at any other absolute path and every loader
(`_load_clustered_features`, `_load_selection_summary`,
`_load_candidate_metrics`, `_load_best_model_params`,
`_load_cluster_metadata`) follows it. After replacing files on disk,
either call `agent_tools.reload_artifacts()` or restart the Streamlit
process so the LRU caches and Streamlit's `@st.cache_data` rebuild on
the next request.
