# Head-to-head bootstrap of D: cont96 (2026-10-31 16:32, 10,000 draws, seed 0)

D is a sum of absolute gaps: it stops at 0 and a gap near 0 can only grow under resampling, so the draws lean above D (mean draw - D) and D need not be central in their 95%. That range is the spread over the prompts, not a confidence interval for the true D, and its low end is no lower bound. A difference of two Ds keeps what the two inflations do not cancel (mean draw - difference), so read it with that column; the share of draws closer is no p-value. Only the files this prompt set drew from are covered.

## cont96, first 1000 new notes, 96 of 96 prompts

| system | D | 95% of draws | mean draw - D |
|---|---|---|---|
| amt | 0.097 | 0.062 to 0.214 | +0.032 |
| f1 | 0.179 | 0.138 to 0.288 | +0.028 |
| long_450m | 0.144 | 0.101 to 0.232 | +0.020 |
| skytnt_large | 0.561 | 0.453 to 0.676 | +0.004 |
| skytnt_medium | 0.280 | 0.214 to 0.380 | +0.011 |

| pair | D difference | 95% of draws | mean draw - difference | share of draws closer |
|---|---|---|---|---|
| f1 vs amt | +0.081 | -0.027 to +0.173 | -0.004 | 0.063 |
| f1 vs skytnt_large | -0.382 | -0.486 to -0.229 | +0.024 | 1.000 |
| f1 vs skytnt_medium | -0.102 | -0.194 to +0.027 | +0.017 | 0.931 |
| long_450m vs amt | +0.046 | -0.078 to +0.137 | -0.012 | 0.257 |
| long_450m vs skytnt_large | -0.417 | -0.514 to -0.289 | +0.016 | 1.000 |
| long_450m vs skytnt_medium | -0.136 | -0.207 to -0.058 | +0.009 | 1.000 |

## cont96 no_drums, first 1000 new notes, 65 of 65 prompts

| system | D | 95% of draws | mean draw - D |
|---|---|---|---|
| amt | 0.094 | 0.055 to 0.244 | +0.039 |
| f1 | 0.200 | 0.157 to 0.336 | +0.040 |
| long_450m | 0.137 | 0.091 to 0.242 | +0.026 |
| skytnt_large | 0.485 | 0.370 to 0.620 | +0.008 |
| skytnt_medium | 0.286 | 0.209 to 0.417 | +0.016 |

| pair | D difference | 95% of draws | mean draw - difference | share of draws closer |
|---|---|---|---|---|
| f1 vs amt | +0.105 | -0.014 to +0.221 | +0.001 | 0.041 |
| f1 vs skytnt_large | -0.285 | -0.400 to -0.102 | +0.032 | 1.000 |
| f1 vs skytnt_medium | -0.087 | -0.202 to +0.067 | +0.024 | 0.821 |
| long_450m vs amt | +0.042 | -0.094 to +0.138 | -0.013 | 0.298 |
| long_450m vs skytnt_large | -0.348 | -0.467 to -0.198 | +0.019 | 1.000 |
| long_450m vs skytnt_medium | -0.150 | -0.242 to -0.054 | +0.010 | 1.000 |

## cont96 with_drums, first 1000 new notes, 31 of 31 prompts

| system | D | 95% of draws | mean draw - D |
|---|---|---|---|
| amt | 0.371 | 0.155 to 0.643 | +0.001 |
| f1 | 0.199 | 0.125 to 0.447 | +0.056 |
| long_450m | 0.246 | 0.116 to 0.391 | +0.003 |
| skytnt_large | 0.846 | 0.606 to 1.088 | -0.004 |
| skytnt_medium | 0.316 | 0.222 to 0.472 | +0.023 |

| pair | D difference | 95% of draws | mean draw - difference | share of draws closer |
|---|---|---|---|---|
| f1 vs amt | -0.172 | -0.374 to +0.148 | +0.055 | 0.807 |
| f1 vs skytnt_large | -0.647 | -0.881 to -0.271 | +0.060 | 1.000 |
| f1 vs skytnt_medium | -0.117 | -0.267 to +0.127 | +0.033 | 0.803 |
| long_450m vs amt | -0.125 | -0.448 to +0.177 | +0.002 | 0.773 |
| long_450m vs skytnt_large | -0.600 | -0.808 to -0.385 | +0.006 | 1.000 |
| long_450m vs skytnt_medium | -0.071 | -0.195 to +0.010 | -0.021 | 0.961 |

## cont96 published, first 1000 new notes, 32 of 32 prompts

| system | D | 95% of draws | mean draw - D |
|---|---|---|---|
| amt | 0.289 | 0.110 to 0.553 | +0.016 |
| f1 | 0.218 | 0.147 to 0.375 | +0.033 |
| long_450m | 0.121 | 0.053 to 0.280 | +0.024 |
| skytnt_large | 0.357 | 0.232 to 0.515 | +0.009 |
| skytnt_medium | 0.314 | 0.189 to 0.542 | +0.030 |

| pair | D difference | 95% of draws | mean draw - difference | share of draws closer |
|---|---|---|---|---|
| f1 vs amt | -0.071 | -0.290 to +0.148 | +0.017 | 0.667 |
| f1 vs skytnt_large | -0.140 | -0.278 to +0.046 | +0.024 | 0.918 |
| f1 vs skytnt_medium | -0.096 | -0.274 to +0.078 | +0.003 | 0.853 |
| long_450m vs amt | -0.168 | -0.405 to +0.054 | +0.008 | 0.926 |
| long_450m vs skytnt_large | -0.236 | -0.389 to -0.056 | +0.014 | 0.997 |
| long_450m vs skytnt_medium | -0.193 | -0.384 to -0.046 | -0.006 | 0.998 |

## cont96 new, first 1000 new notes, 64 of 64 prompts

| system | D | 95% of draws | mean draw - D |
|---|---|---|---|
| amt | 0.224 | 0.102 to 0.398 | +0.016 |
| f1 | 0.159 | 0.119 to 0.315 | +0.047 |
| long_450m | 0.181 | 0.119 to 0.292 | +0.021 |
| skytnt_large | 0.693 | 0.555 to 0.852 | +0.006 |
| skytnt_medium | 0.295 | 0.233 to 0.406 | +0.018 |

| pair | D difference | 95% of draws | mean draw - difference | share of draws closer |
|---|---|---|---|---|
| f1 vs amt | -0.064 | -0.214 to +0.146 | +0.032 | 0.640 |
| f1 vs skytnt_large | -0.534 | -0.661 to -0.316 | +0.041 | 1.000 |
| f1 vs skytnt_medium | -0.136 | -0.232 to +0.025 | +0.029 | 0.946 |
| long_450m vs amt | -0.042 | -0.231 to +0.143 | +0.005 | 0.640 |
| long_450m vs skytnt_large | -0.512 | -0.640 to -0.362 | +0.015 | 1.000 |
| long_450m vs skytnt_medium | -0.114 | -0.175 to -0.048 | +0.002 | 1.000 |

## cont96, first 200 new notes, 96 of 96 prompts

| system | D | 95% of draws | mean draw - D |
|---|---|---|---|
| amt | 0.099 | 0.045 to 0.215 | +0.015 |
| f1 | 0.205 | 0.135 to 0.318 | +0.020 |
| long_450m | 0.084 | 0.053 to 0.159 | +0.016 |
| skytnt_large | 0.219 | 0.162 to 0.306 | +0.011 |
| skytnt_medium | 0.161 | 0.132 to 0.257 | +0.024 |

| pair | D difference | 95% of draws | mean draw - difference | share of draws closer |
|---|---|---|---|---|
| f1 vs amt | +0.106 | -0.043 to +0.242 | +0.005 | 0.075 |
| f1 vs skytnt_large | -0.014 | -0.127 to +0.112 | +0.009 | 0.538 |
| f1 vs skytnt_medium | +0.044 | -0.084 to +0.150 | -0.004 | 0.242 |
| long_450m vs amt | -0.015 | -0.110 to +0.066 | +0.001 | 0.587 |
| long_450m vs skytnt_large | -0.135 | -0.204 to -0.061 | +0.005 | 1.000 |
| long_450m vs skytnt_medium | -0.076 | -0.139 to -0.037 | -0.008 | 1.000 |

## cont96 no_drums, first 200 new notes, 65 of 65 prompts

| system | D | 95% of draws | mean draw - D |
|---|---|---|---|
| amt | 0.080 | 0.034 to 0.207 | +0.020 |
| f1 | 0.204 | 0.115 to 0.315 | +0.009 |
| long_450m | 0.089 | 0.048 to 0.164 | +0.012 |
| skytnt_large | 0.214 | 0.158 to 0.318 | +0.018 |
| skytnt_medium | 0.175 | 0.137 to 0.289 | +0.027 |

| pair | D difference | 95% of draws | mean draw - difference | share of draws closer |
|---|---|---|---|---|
| f1 vs amt | +0.124 | -0.055 to +0.251 | -0.011 | 0.085 |
| f1 vs skytnt_large | -0.010 | -0.149 to +0.110 | -0.008 | 0.609 |
| f1 vs skytnt_medium | +0.029 | -0.127 to +0.137 | -0.017 | 0.416 |
| long_450m vs amt | +0.008 | -0.105 to +0.084 | -0.009 | 0.468 |
| long_450m vs skytnt_large | -0.125 | -0.216 to -0.051 | -0.006 | 1.000 |
| long_450m vs skytnt_medium | -0.086 | -0.173 to -0.038 | -0.015 | 1.000 |

## cont96 with_drums, first 200 new notes, 31 of 31 prompts

| system | D | 95% of draws | mean draw - D |
|---|---|---|---|
| amt | 0.157 | 0.108 to 0.450 | +0.082 |
| f1 | 0.336 | 0.160 to 0.609 | +0.025 |
| long_450m | 0.141 | 0.077 to 0.303 | +0.036 |
| skytnt_large | 0.271 | 0.163 to 0.415 | +0.013 |
| skytnt_medium | 0.185 | 0.134 to 0.346 | +0.036 |

| pair | D difference | 95% of draws | mean draw - difference | share of draws closer |
|---|---|---|---|---|
| f1 vs amt | +0.179 | -0.205 to +0.422 | -0.058 | 0.221 |
| f1 vs skytnt_large | +0.065 | -0.201 to +0.378 | +0.012 | 0.312 |
| f1 vs skytnt_medium | +0.150 | -0.116 to +0.407 | -0.012 | 0.153 |
| long_450m vs amt | -0.016 | -0.274 to +0.092 | -0.046 | 0.728 |
| long_450m vs skytnt_large | -0.129 | -0.216 to -0.004 | +0.024 | 0.980 |
| long_450m vs skytnt_medium | -0.044 | -0.124 to +0.035 | -0.000 | 0.856 |

## cont96 published, first 200 new notes, 32 of 32 prompts

| system | D | 95% of draws | mean draw - D |
|---|---|---|---|
| amt | 0.239 | 0.102 to 0.427 | +0.002 |
| f1 | 0.198 | 0.085 to 0.361 | +0.017 |
| long_450m | 0.090 | 0.055 to 0.196 | +0.026 |
| skytnt_large | 0.218 | 0.160 to 0.372 | +0.033 |
| skytnt_medium | 0.184 | 0.139 to 0.375 | +0.051 |

| pair | D difference | 95% of draws | mean draw - difference | share of draws closer |
|---|---|---|---|---|
| f1 vs amt | -0.041 | -0.304 to +0.211 | +0.016 | 0.549 |
| f1 vs skytnt_large | -0.020 | -0.220 to +0.145 | -0.015 | 0.641 |
| f1 vs skytnt_medium | +0.014 | -0.217 to +0.164 | -0.033 | 0.568 |
| long_450m vs amt | -0.149 | -0.302 to +0.018 | +0.024 | 0.954 |
| long_450m vs skytnt_large | -0.128 | -0.243 to -0.041 | -0.007 | 0.998 |
| long_450m vs skytnt_medium | -0.094 | -0.244 to -0.022 | -0.025 | 0.993 |

## cont96 new, first 200 new notes, 64 of 64 prompts

| system | D | 95% of draws | mean draw - D |
|---|---|---|---|
| amt | 0.047 | 0.045 to 0.196 | +0.058 |
| f1 | 0.215 | 0.128 to 0.364 | +0.030 |
| long_450m | 0.098 | 0.052 to 0.192 | +0.015 |
| skytnt_large | 0.264 | 0.194 to 0.354 | +0.006 |
| skytnt_medium | 0.155 | 0.119 to 0.256 | +0.024 |

| pair | D difference | 95% of draws | mean draw - difference | share of draws closer |
|---|---|---|---|---|
| f1 vs amt | +0.168 | -0.007 to +0.278 | -0.028 | 0.032 |
| f1 vs skytnt_large | -0.049 | -0.179 to +0.122 | +0.023 | 0.633 |
| f1 vs skytnt_medium | +0.060 | -0.084 to +0.198 | +0.005 | 0.180 |
| long_450m vs amt | +0.052 | -0.098 to +0.103 | -0.042 | 0.406 |
| long_450m vs skytnt_large | -0.166 | -0.230 to -0.086 | +0.009 | 1.000 |
| long_450m vs skytnt_medium | -0.057 | -0.116 to -0.014 | -0.009 | 0.995 |
