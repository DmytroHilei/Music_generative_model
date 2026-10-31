# Head-to-head bootstrap of D: cont (2026-10-30 21:04, 10,000 draws, seed 0)

D is a sum of absolute gaps: it stops at 0 and a gap near 0 can only grow under resampling, so the draws lean above D (mean draw - D) and D need not be central in their 95%. That range is the spread over the prompts, not a confidence interval for the true D, and its low end is no lower bound. A difference of two Ds keeps what the two inflations do not cancel (mean draw - difference), so read it with that column; the share of draws closer is no p-value. Only the files this prompt set drew from are covered.

## First 1000 new notes, 32 of 32 prompts

| system | D | 95% of draws | mean draw - D |
|---|---|---|---|
| amt | 0.289 | 0.110 to 0.553 | +0.016 |
| f1 | 0.218 | 0.147 to 0.375 | +0.033 |
| long_450m | 0.121 | 0.053 to 0.280 | +0.024 |
| moonbeam | 0.703 | 0.319 to 1.287 | +0.033 |
| moonbeam_t06 | 1.316 | 0.381 to 3.130 | +0.026 |
| skytnt_large | 0.357 | 0.232 to 0.515 | +0.009 |
| skytnt_medium | 0.314 | 0.189 to 0.542 | +0.030 |

| pair | D difference | 95% of draws | mean draw - difference | share of draws closer |
|---|---|---|---|---|
| f1 vs amt | -0.071 | -0.290 to +0.148 | +0.017 | 0.667 |
| f1 vs moonbeam | -0.485 | -1.025 to -0.073 | -0.000 | 0.993 |
| f1 vs moonbeam_t06 | -1.098 | -2.866 to -0.150 | +0.007 | 1.000 |
| f1 vs skytnt_large | -0.140 | -0.278 to +0.046 | +0.024 | 0.918 |
| f1 vs skytnt_medium | -0.096 | -0.274 to +0.078 | +0.003 | 0.853 |
| long_450m vs amt | -0.168 | -0.405 to +0.054 | +0.008 | 0.926 |
| long_450m vs moonbeam | -0.582 | -1.168 to -0.138 | -0.010 | 0.999 |
| long_450m vs moonbeam_t06 | -1.195 | -2.998 to -0.241 | -0.003 | 1.000 |
| long_450m vs skytnt_large | -0.236 | -0.389 to -0.056 | +0.014 | 0.997 |
| long_450m vs skytnt_medium | -0.193 | -0.384 to -0.046 | -0.006 | 0.998 |

## First 200 new notes, 32 of 32 prompts

| system | D | 95% of draws | mean draw - D |
|---|---|---|---|
| amt | 0.239 | 0.102 to 0.427 | +0.002 |
| f1 | 0.198 | 0.085 to 0.361 | +0.017 |
| long_450m | 0.090 | 0.055 to 0.196 | +0.026 |
| moonbeam | 0.322 | 0.136 to 0.648 | +0.043 |
| moonbeam_t06 | 0.525 | 0.241 to 0.957 | +0.020 |
| skytnt_large | 0.218 | 0.160 to 0.372 | +0.033 |
| skytnt_medium | 0.184 | 0.139 to 0.375 | +0.051 |

| pair | D difference | 95% of draws | mean draw - difference | share of draws closer |
|---|---|---|---|---|
| f1 vs amt | -0.041 | -0.304 to +0.211 | +0.016 | 0.549 |
| f1 vs moonbeam | -0.124 | -0.464 to +0.136 | -0.026 | 0.826 |
| f1 vs moonbeam_t06 | -0.327 | -0.748 to +0.025 | -0.003 | 0.966 |
| f1 vs skytnt_large | -0.020 | -0.220 to +0.145 | -0.015 | 0.641 |
| f1 vs skytnt_medium | +0.014 | -0.217 to +0.164 | -0.033 | 0.568 |
| long_450m vs amt | -0.149 | -0.302 to +0.018 | +0.024 | 0.954 |
| long_450m vs moonbeam | -0.232 | -0.526 to -0.028 | -0.017 | 0.991 |
| long_450m vs moonbeam_t06 | -0.435 | -0.836 to -0.134 | +0.006 | 1.000 |
| long_450m vs skytnt_large | -0.128 | -0.243 to -0.041 | -0.007 | 0.998 |
| long_450m vs skytnt_medium | -0.094 | -0.244 to -0.022 | -0.025 | 0.993 |
