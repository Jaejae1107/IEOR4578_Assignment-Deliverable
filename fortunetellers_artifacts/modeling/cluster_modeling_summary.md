# FortuneTellers Cluster Modeling Summary

This report summarizes the selected model for each demand cluster and the strongest candidate comparisons from the current rerun.

## Selected Models

| Cluster | Label | Selected Model | Valid MAPE | Test MAPE | LGBM Baseline | Delta vs LGBM |
| --- | --- | --- | --- | --- | --- | --- |
| -2 | Truly sporadic | LGBM_Default | 83.19 | 93.69 | 93.69 | 0.00 |
| -1 | Intermittent (Croston) | RF_Default | 78.17 | 73.85 | 76.49 | -2.65 |
| 0 | High cancellation risk | RF_Default | 92.30 | 78.57 | 103.43 | -24.86 |
| 1 | Steady regulars | RF_Default | 83.66 | 77.15 | 82.51 | -5.36 |
| 2 | Volatile mid-range | RF_Default | 73.78 | 69.31 | 78.05 | -8.74 |

## Cluster-by-Cluster Notes

### Cluster -2 - Truly sporadic

- Selected model: `LGBM_Default`
- Held-out test MAPE: `93.69`
- Relative result: `LGBM_Default` matched the LGBM baseline.

| Candidate | Valid MAPE | Valid N |
| --- | --- | --- |
| LGBM_Default | 83.19 | 1077 |
| RF_Default | 83.66 | 1077 |
| AggregateMLP_Disagg | 105.26 | 1077 |

### Cluster -1 - Intermittent (Croston)

- Selected model: `RF_Default`
- Held-out test MAPE: `73.85`
- Relative result: `RF_Default` beat the LGBM baseline by 2.65 points.

| Candidate | Valid MAPE | Valid N |
| --- | --- | --- |
| RF_Default | 78.17 | 3240 |
| LGBM_Default | 82.66 | 3240 |
| AggregateMLP_Disagg | 337.78 | 3240 |

### Cluster 0 - High cancellation risk

- Selected model: `RF_Default`
- Held-out test MAPE: `78.57`
- Relative result: `RF_Default` beat the LGBM baseline by 24.86 points.

| Candidate | Valid MAPE | Valid N |
| --- | --- | --- |
| RF_Default | 92.30 | 325 |
| LGBM_Tuned | 104.81 | 325 |
| LGBM_Default | 108.05 | 325 |
| AggregateMLP_Disagg | 518.83 | 325 |

### Cluster 1 - Steady regulars

- Selected model: `RF_Default`
- Held-out test MAPE: `77.15`
- Relative result: `RF_Default` beat the LGBM baseline by 5.36 points.

| Candidate | Valid MAPE | Valid N |
| --- | --- | --- |
| RF_Default | 83.66 | 5694 |
| LGBM_Tuned | 87.55 | 5694 |
| LGBM_Default | 87.94 | 5694 |
| AggregateMLP_Disagg | 672.00 | 5694 |

### Cluster 2 - Volatile mid-range

- Selected model: `RF_Default`
- Held-out test MAPE: `69.31`
- Relative result: `RF_Default` beat the LGBM baseline by 8.74 points.

| Candidate | Valid MAPE | Valid N |
| --- | --- | --- |
| RF_Default | 73.78 | 763 |
| RF_C2_BEST | 74.00 | 763 |
| LGBM_Tuned | 79.33 | 763 |
| LGBM_Default | 80.07 | 763 |
| AggregateMLP_Disagg | 415.91 | 763 |

## Key Takeaways

- `RF_Default` won 4 of the 5 clusters in this rerun, which suggests the simpler tree ensemble is the most stable default choice right now.
- `High cancellation risk` saw the biggest gain over the LGBM baseline, which supports keeping a conservative deployment story for that segment.
- `Truly sporadic` remains weak regardless of model choice, so this segment should stay in the low-confidence / manual-review bucket.
- For presentation, use the cluster label more prominently than the cluster number because the numeric IDs can shift across reruns.
