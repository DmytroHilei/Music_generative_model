# Head-to-head bootstrap of D: piano (2026-10-30 21:04, 10,000 draws, seed 0)

D is a sum of absolute gaps: it stops at 0 and a gap near 0 can only grow under resampling, so the draws lean above D (mean draw - D) and D need not be central in their 95%. That range is the spread over the prompts, not a confidence interval for the true D, and its low end is no lower bound. A difference of two Ds keeps what the two inflations do not cancel (mean draw - difference), so read it with that column; the share of draws closer is no p-value. Only the files this prompt set drew from are covered.

## First 1000 new notes, 32 of 32 prompts

| system | D | 95% of draws | mean draw - D |
|---|---|---|---|
| aria | 0.096 | 0.007 to 0.232 | +0.000 |
| long_450m | 0.040 | 0.011 to 0.071 | +0.000 |
| piano | 0.177 | 0.136 to 0.217 | +0.000 |
| ua_2048_v2 | 0.018 | 0.001 to 0.048 | +0.002 |

| pair | D difference | 95% of draws | mean draw - difference | share of draws closer |
|---|---|---|---|---|
| long_450m vs aria | -0.056 | -0.181 to +0.025 | -0.000 | 0.842 |
| piano vs aria | +0.081 | -0.051 to +0.177 | -0.000 | 0.099 |
| ua_2048_v2 vs aria | -0.078 | -0.215 to +0.021 | +0.002 | 0.908 |

## First 200 new notes, 32 of 32 prompts

| system | D | 95% of draws | mean draw - D |
|---|---|---|---|
| aria | 0.067 | 0.012 to 0.140 | -0.000 |
| long_450m | 0.019 | 0.002 to 0.041 | +0.000 |
| piano | 0.078 | 0.038 to 0.123 | +0.000 |
| ua_2048_v2 | 0.006 | 0.001 to 0.040 | +0.008 |

| pair | D difference | 95% of draws | mean draw - difference | share of draws closer |
|---|---|---|---|---|
| long_450m vs aria | -0.047 | -0.117 to +0.002 | +0.001 | 0.966 |
| piano vs aria | +0.012 | -0.058 to +0.073 | +0.001 | 0.342 |
| ua_2048_v2 vs aria | -0.060 | -0.124 to +0.004 | +0.009 | 0.961 |
