# FortuneTellers Original vs Current Comparison

This report compares the original notebook benchmark against the current rerun after restoring the original clustering split.

Notes:
- Original benchmark comes from the source notebook's unified pipeline output.
- Current rerun uses the restored original cluster definitions and a pooled RandomForest benchmark per cluster.
- Current MAPE is therefore useful for directional comparison, not a perfect apples-to-apples replacement for every original candidate family.

## Cluster Alignment Check

| Cluster | Original Label | Current Label | Original Count | Current Count | Label Match | Count Match |
| --- | --- | --- | --- | --- | --- | --- |
| -2 | Truly sporadic | Truly sporadic | 1328 | 1328 | Yes | Yes |
| -1 | Intermittent (Croston) | Intermittent (Croston) | 1627 | 1627 | Yes | Yes |
| 0 | High cancellation risk | High cancellation risk | 92 | 92 | Yes | Yes |
| 1 | Volatile mid-range | Volatile mid-range | 228 | 228 | Yes | Yes |
| 2 | Steady regulars | Steady regulars | 1281 | 1281 | Yes | Yes |

## Model Comparison

| Cluster | Original Model | Original Test MAPE | Current Model | Current Test MAPE | Delta (Current - Original) |
| --- | --- | --- | --- | --- | --- |
| -2 | RF_Default | 89.04 | RF_Default | 72.30 | -16.74 |
| -1 | RF_Default | 76.34 | RF_Default | 71.63 | -4.71 |
| 0 | RF_Default | 82.89 | RF_Default | 78.50 | -4.39 |
| 1 | RF_Default | 75.06 | RF_Default | 69.81 | -5.26 |
| 2 | RF_C2_BEST | 81.40 | RF_C2_BEST | 74.72 | -6.68 |
