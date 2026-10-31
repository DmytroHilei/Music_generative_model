# Head-to-head bootstrap of D: piano96 (2026-10-31 14:57, 10,000 draws, seed 0)

D is a sum of absolute gaps: it stops at 0 and a gap near 0 can only grow under resampling, so the draws lean above D (mean draw - D) and D need not be central in their 95%. That range is the spread over the prompts, not a confidence interval for the true D, and its low end is no lower bound. A difference of two Ds keeps what the two inflations do not cancel (mean draw - difference), so read it with that column; the share of draws closer is no p-value. Only the files this prompt set drew from are covered.

## piano96, first 1000 new notes, 96 of 96 prompts

| system | D | 95% of draws | mean draw - D |
|---|---|---|---|
| aria | 0.074 | 0.020 to 0.141 | -0.000 |
| long_450m | 0.035 | 0.011 to 0.062 | -0.000 |
| piano | 0.182 | 0.157 to 0.209 | +0.000 |
| ua_2048_v2 | 0.012 | 0.001 to 0.034 | +0.002 |

| pair | D difference | 95% of draws | mean draw - difference | share of draws closer |
|---|---|---|---|---|
| long_450m vs aria | -0.039 | -0.099 to +0.011 | +0.000 | 0.926 |
| piano vs aria | +0.108 | +0.043 to +0.167 | +0.000 | 0.001 |
| ua_2048_v2 vs aria | -0.062 | -0.127 to -0.004 | +0.002 | 0.983 |

## piano96, first 200 new notes, 96 of 96 prompts

| system | D | 95% of draws | mean draw - D |
|---|---|---|---|
| aria | 0.044 | 0.016 to 0.075 | +0.000 |
| long_450m | 0.016 | 0.001 to 0.034 | +0.000 |
| piano | 0.081 | 0.053 to 0.111 | +0.000 |
| ua_2048_v2 | 0.003 | 0.000 to 0.027 | +0.006 |

| pair | D difference | 95% of draws | mean draw - difference | share of draws closer |
|---|---|---|---|---|
| long_450m vs aria | -0.028 | -0.055 to -0.004 | +0.000 | 0.990 |
| piano vs aria | +0.037 | -0.005 to +0.082 | +0.000 | 0.046 |
| ua_2048_v2 vs aria | -0.040 | -0.068 to -0.001 | +0.006 | 0.977 |
