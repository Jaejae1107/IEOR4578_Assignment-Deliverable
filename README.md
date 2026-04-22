# FortuneTellers Retail Forecasting Pipeline

This repository packages the FortuneTellers deliverable into a reproducible workflow that goes beyond the original notebook. It includes:

- feature engineering for product-level demand segmentation
- cluster-wise model training and model selection
- a forecasting pipeline that routes a SKU to the right cluster model
- an agent-style query wrapper that answers natural-language forecast requests

The code was verified locally on April 22, 2026 using the UCI `Online Retail II` Excel file and the generated artifacts saved in this repo.

## Project Goal

The project objective is to improve retail demand forecasting by avoiding a one-model-fits-all approach. Instead, products are segmented by demand behavior first, then modeled with cluster-specific forecasting logic.

The current implementation supports the following workflow:

1. clean and standardize the raw transaction data
2. engineer product features from weekly demand behavior
3. segment products into demand clusters
4. train and compare candidate models for each cluster
5. route a SKU query into the selected cluster model
6. generate a 12-week forecast plus an operational recommendation

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── data/
│   ├── README.md
│   └── raw/
├── fortunetellers/
│   ├── __init__.py
│   ├── agent.py
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── modeling.py
│   └── pipeline.py
├── scripts/
│   ├── run_fortunetellers_agent.py
│   ├── run_fortunetellers_feature_engineering.py
│   ├── run_fortunetellers_forecast_pipeline.py
│   ├── run_fortunetellers_modeling.py
│   └── run_fortunetellers_modeling_summary.py
└── fortunetellers_artifacts/
    ├── agent/
    ├── clustering/
    ├── modeling/
    └── pipeline/
```

## Data Requirement

The raw Excel file is not committed to this repository. To run the full pipeline from scratch, place the UCI dataset at:

`data/raw/online_retail_II.xlsx`

You can also override the location with:

- the CLI flag `--raw-excel`
- the environment variable `FORTUNETELLERS_RAW_EXCEL`

If the file is not present in `data/raw`, the scripts also look for `~/Downloads/online_retail_II.xlsx` as a convenience fallback.

## Environment Setup

Create a virtual environment and install the required libraries:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Optional LightGBM Setup on macOS

The code is resilient to missing native LightGBM. If `lightgbm` fails to import, the project falls back to histogram gradient boosting classes with the same interface for the key modeling steps.

If you want the native LightGBM path on macOS:

```bash
brew install libomp
pip install lightgbm
```

## End-to-End Run Order

Run the project from the repository root.

### 1. Feature Engineering

```bash
python scripts/run_fortunetellers_feature_engineering.py --rebuild
```

What this step produces:

- cleaned transaction CSVs
- product feature table
- scaled clustering table
- cluster assignments and metadata

Primary outputs:

- `fortunetellers_artifacts/clustering/product_features_clustered.csv`
- `fortunetellers_artifacts/clustering/cluster_metadata.json`

### 2. Cluster-Wise Modeling

```bash
python scripts/run_fortunetellers_modeling.py
python scripts/run_fortunetellers_modeling_summary.py
```

What this step produces:

- model selection summary for each cluster
- candidate comparison metrics
- tuned LightGBM trial log
- saved model parameter payload for downstream routing
- markdown summary for reporting

Primary outputs:

- `fortunetellers_artifacts/modeling/cluster_model_selection_summary.csv`
- `fortunetellers_artifacts/modeling/cluster_candidate_metrics.csv`
- `fortunetellers_artifacts/modeling/best_model_params.json`
- `fortunetellers_artifacts/modeling/cluster_modeling_summary.md`

### 3. Forecasting Pipeline

```bash
python scripts/run_fortunetellers_forecast_pipeline.py --product-id 85123A --country "United Kingdom"
```

What this step does:

- loads the saved cluster artifacts
- finds the target product's cluster
- loads the selected model recipe for that cluster
- builds the product history panel
- recursively predicts the next 12 weeks
- writes a machine-readable forecast JSON

Primary output:

- `fortunetellers_artifacts/pipeline/forecast_85123A_United_Kingdom.json`

### 4. Agent-Style Query Wrapper

```bash
python scripts/run_fortunetellers_agent.py --query "What is the 12-week forecast for product 85123A in United Kingdom?"
```

Example queries:

```bash
python scripts/run_fortunetellers_agent.py --query "Give me the forecast for SKU 85123A in UK"
python scripts/run_fortunetellers_agent.py --query "Forecast product 85123A for all countries"
```

What this step does:

- parses a product ID from natural language
- resolves country aliases such as `UK`
- calls the forecasting pipeline
- saves both the forecast JSON and an agent response JSON
- prints a business-facing answer to the terminal

Primary output:

- `fortunetellers_artifacts/agent/agent_response_85123A_United_Kingdom.json`

## Methodology

## 1. Data Cleaning

Implemented in `fortunetellers/data.py`.

The raw workbook contains two yearly sheets. The preprocessing logic:

- parses both yearly tabs
- removes the repeated `536365` overlap from the first sheet
- concatenates both years into one retail table
- standardizes column names
- removes duplicate rows
- filters out zero-price rows
- removes non-product entries such as postage, gift vouchers, manual adjustments, and fee rows
- creates weekly periods and sales values

The training and test boundary is defined by the final 12 weekly periods of the full data range.

## 2. Feature Engineering

Implemented in `fortunetellers/features.py`.

Product-level features are engineered from weekly transaction behavior, including:

- mean and median weekly sales
- total sales
- average transaction size
- weekly sales standard deviation and coefficient of variation
- spike ratio
- percentage of zero-sales weeks
- average inter-demand interval
- longest zero-sales streak
- seasonal concentration
- quarter mix such as `q4_pct` and `q1_pct`
- trend features including `trend_log_diff`
- price behavior
- customer concentration and repeat behavior
- country concentration
- cancellation rate

Products are first bucketed into:

- `Truly sporadic`
- `Intermittent (Croston)`
- `Active products`

Only active products are passed into KMeans clustering. The clustering inputs are:

- `cv`
- `seasonal_conc`
- `q4_pct`
- `log_mean_price`
- `cancel_rate`

The project evaluates `k = 3, 4, 5` and selects the best `k` using silhouette score.

## 3. Cluster Labeling

After KMeans assigns the active-product clusters, each cluster receives a business label using rule-based summaries of the median feature profile.

Possible active-cluster labels include:

- `Steady regulars`
- `Volatile mid-range`
- `High cancellation risk`
- `Christmas / seasonal`
- `Steady mid-range`

Important note:

Cluster numbers can change across reruns because KMeans label IDs are arbitrary. For reporting and presentation, use the cluster label more prominently than the numeric cluster ID.

## 4. Modeling Strategy

Implemented in `fortunetellers/modeling.py`.

The project compares candidate models at the cluster level rather than fitting one global model for all products. The current code supports:

- `RF_Default`
- `RF_C2_BEST`
- `LGBM_Default`
- `LGBM_Tuned`
- `CrostonSBA`
- residual-corrected variants when relevant

Supporting logic includes:

- signed-log transformation for net sales
- time-based train/validation splitting
- weekly lag and rolling features
- return-event modeling for residual correction
- Croston-style handling for intermittent demand
- fallback LightGBM-compatible estimators when native LightGBM is unavailable

The selected model for each cluster is saved into `best_model_params.json`, which becomes the routing contract for the downstream pipeline.

## 5. Forecasting Pipeline

Implemented in `fortunetellers/pipeline.py`.

The forecasting pipeline is intentionally designed as a deployable layer on top of the modeling artifacts. Given a SKU and a country filter, it:

- locates the SKU in the product feature table
- retrieves the cluster assignment
- reads the selected model config for that cluster
- rebuilds the weekly panel for the relevant training population
- recursively generates future lagged features
- predicts the next 12 weekly sales values
- attaches a warning flag and recommendation based on the demand segment

## 6. Agent Wrapper

Implemented in `fortunetellers/agent.py`.

The agent wrapper is a lightweight natural-language interface, not a full LLM agent runtime. It exists so the project can demonstrate an interactive decision-support layer before a meeting or class demo.

The wrapper currently supports:

- SKU extraction from natural-language prompts
- country alias resolution
- formatted business reply generation
- JSON logging for downstream UI or integration work

## Latest Verified Results

The following results were generated and verified on April 22, 2026.

### Feature Engineering

- Best active-product cluster count: `k = 3`
- Silhouette scores:
  - `k = 3`: `0.4767`
  - `k = 4`: `0.3933`
  - `k = 5`: `0.3681`

Cluster counts from the verified rerun:

- `-2`: `Truly sporadic` (`1384`)
- `-1`: `Intermittent (Croston)` (`1711`)
- `0`: `High cancellation risk` (`82`)
- `1`: `Steady regulars` (`1294`)
- `2`: `Volatile mid-range` (`218`)

### Selected Model by Cluster

| Cluster | Label | Selected Model | Valid MAPE | Test MAPE |
| --- | --- | --- | --- | --- |
| -2 | Truly sporadic | LGBM_Default | 83.19 | 93.69 |
| -1 | Intermittent (Croston) | RF_Default | 78.17 | 73.85 |
| 0 | High cancellation risk | RF_Default | 92.30 | 78.57 |
| 1 | Steady regulars | RF_Default | 83.66 | 77.15 |
| 2 | Volatile mid-range | RF_Default | 73.78 | 69.31 |

Interpretation:

- `RF_Default` won in 4 of the 5 clusters
- `High cancellation risk` improved substantially relative to the LightGBM baseline
- `Truly sporadic` remains the weakest segment and should stay in the low-confidence bucket

### Verified Pipeline Example

Query:

```bash
python scripts/run_fortunetellers_agent.py --query "What is the 12-week forecast for product 85123A in United Kingdom?"
```

Verified result:

- Product: `85123A`
- Description: `WHITE HANGING HEART T-LIGHT HOLDER`
- Segment: `Steady regulars`
- Selected model: `RF_Default`
- Test MAPE: `77.14897784431318`
- Recommendation: `Automation-ready; use forecast for replenishment planning.`
- Warning level: `normal`

Sample forecast preview:

| Week Start | Forecast Sales |
| --- | --- |
| 2011-12-12 | 1333.75 |
| 2011-12-19 | 405.06 |
| 2011-12-26 | 0.36 |
| 2012-01-02 | 716.17 |

The same agent wrapper was also verified with:

- `UK` alias parsing
- `ALL` country aggregation

## Reproducibility Notes

- The repo does not ship the raw Excel file.
- The repo ignores the large processed CSV outputs because they can be regenerated from the raw workbook.
- Smaller downstream artifacts are committed so reviewers can inspect the latest rerun outputs without executing the full pipeline immediately.
- Native LightGBM is optional. If it is unavailable, the code uses a compatible fallback to keep the workflow runnable.

## Known Limitations

- Cluster numbering is not stable across reruns.
- Forecast quality for intermittent and truly sporadic demand remains limited.
- The current agent layer is a rule-based wrapper, not a production conversational service.
- The code currently assumes weekly forecasting only.
- The forecast recommendation logic is intentionally simple and heuristic.

## Recommended Next Steps

- add unit tests around parsing, routing, and forecast serialization
- expose the pipeline through a small web or notebook demo
- connect the agent output JSON to `n8n` or a chat interface
- persist trained model objects if runtime latency becomes important
- compare alternative intermittent-demand models for the weakest clusters

## Key Files

- `fortunetellers/config.py`: project paths and shared constants
- `fortunetellers/data.py`: data loading and cleaning
- `fortunetellers/features.py`: feature engineering and clustering
- `fortunetellers/modeling.py`: cluster-wise training and selection
- `fortunetellers/pipeline.py`: forecasting pipeline
- `fortunetellers/agent.py`: natural-language wrapper
- `scripts/run_fortunetellers_feature_engineering.py`: feature engineering entrypoint
- `scripts/run_fortunetellers_modeling.py`: modeling entrypoint
- `scripts/run_fortunetellers_forecast_pipeline.py`: pipeline entrypoint
- `scripts/run_fortunetellers_agent.py`: agent entrypoint
- `scripts/run_fortunetellers_modeling_summary.py`: reporting helper

## Submission Checklist

- feature engineering runnable from script
- cluster-wise modeling runnable from script
- forecasting pipeline runnable from script
- agent-style forecast interaction runnable from script
- latest rerun artifacts included for inspection
- setup and reproduction steps documented in detail
