# FortuneTellers Cluster Modeling Summary

This report summarizes the selected model for each demand cluster and the strongest candidate comparisons from the current rerun.

## Selected Models

| Cluster | Label | Selected Model | Valid MAPE | Test MAPE | LGBM Baseline | Delta vs LGBM |
| --- | --- | --- | --- | --- | --- | --- |
| -2 | Truly sporadic | RF_Default | 82.41 | 79.41 | 90.44 | -11.03 |
| -1 | Intermittent (Croston) | RF_Default | 77.54 | 70.95 | 73.92 | -2.98 |
| 0 | High cancellation risk | RF_Default | 102.45 | 80.61 | 100.93 | -20.31 |
| 1 | Steady regulars | RF_Default | 83.61 | 75.04 | 80.32 | -5.28 |
| 2 | Volatile mid-range | RF_C2_BEST | 72.76 | 68.15 | 77.87 | -9.72 |

## Cluster-by-Cluster Notes

### Cluster -2 - Truly sporadic

- Selected model: `RF_Default`
- Held-out test MAPE: `79.41`
- Relative result: `RF_Default` beat the LGBM baseline by 11.03 points.

| Candidate | Valid MAPE | Valid N |
| --- | --- | --- |
| RF_Default | 82.41 | 1071 |
| LGBM_Default | 85.36 | 1071 |
| AggregateMLP_Disagg | 111.21 | 1071 |

### Cluster -1 - Intermittent (Croston)

- Selected model: `RF_Default`
- Held-out test MAPE: `70.95`
- Relative result: `RF_Default` beat the LGBM baseline by 2.98 points.

| Candidate | Valid MAPE | Valid N |
| --- | --- | --- |
| RF_Default | 77.54 | 3223 |
| LGBM_Default | 83.17 | 3223 |
| AggregateMLP_Disagg | 433.86 | 3223 |

### Cluster 0 - High cancellation risk

- Selected model: `RF_Default`
- Held-out test MAPE: `80.61`
- Relative result: `RF_Default` beat the LGBM baseline by 20.31 points.

| Candidate | Valid MAPE | Valid N |
| --- | --- | --- |
| RF_Default | 102.45 | 325 |
| LGBM_Tuned | 116.73 | 325 |
| LGBM_Default | 128.42 | 325 |
| AggregateMLP_Disagg | 449.96 | 325 |

### Cluster 1 - Steady regulars

- Selected model: `RF_Default`
- Held-out test MAPE: `75.04`
- Relative result: `RF_Default` beat the LGBM baseline by 5.28 points.

| Candidate | Valid MAPE | Valid N |
| --- | --- | --- |
| RF_Default | 83.61 | 5683 |
| LGBM_Tuned | 87.04 | 5683 |
| LGBM_Default | 88.48 | 5683 |
| AggregateMLP_Disagg | 704.88 | 5683 |

### Cluster 2 - Volatile mid-range

- Selected model: `RF_C2_BEST`
- Held-out test MAPE: `68.15`
- Relative result: `RF_C2_BEST` beat the LGBM baseline by 9.72 points.

| Candidate | Valid MAPE | Valid N |
| --- | --- | --- |
| RF_C2_BEST | 72.76 | 760 |
| RF_Default | 72.84 | 760 |
| LGBM_Default | 79.16 | 760 |
| LGBM_Tuned | 79.26 | 760 |
| AggregateMLP_Disagg | 442.38 | 760 |

## Key Takeaways

- `RF_Default` won 4 of the 5 clusters in this rerun, which suggests the simpler tree ensemble is the most stable default choice right now.
- `High cancellation risk` saw the biggest gain over the LGBM baseline, which supports keeping a conservative deployment story for that segment.
- `Truly sporadic` remains weak regardless of model choice, so this segment should stay in the low-confidence / manual-review bucket.
- For presentation, use the cluster label more prominently than the cluster number because the numeric IDs can shift across reruns.
