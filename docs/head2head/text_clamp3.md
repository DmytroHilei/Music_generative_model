# CLaMP 3 scores, text (2026-10-31 01:52)

C2 checkpoint, every file through read_midi -> tokens_to_midi. 10000 bootstrap draws over prompts (seed 1234), 95% percentile interval. Up to 100 other captions per prompt. first = the file's first segment of 512 patches, the same input size for all rows; global = the authors' whole-file feature, which favours longer files (module docstring).

## Length

| system | files | patches | segments | files over one segment |
|---|---|---|---|---|
| caption | 431 | 305 | 1.32 | 0.325 |
| f1 | 432 | 225 | 1.07 | 0.072 |
| f1_retrieve | 432 | 254 | 1.09 | 0.093 |
| midi_llm | 431 | 266 | 1.00 | 0.000 |
| s50 | 432 | 350 | 1.45 | 0.451 |
| text2midi | 432 | 193 | 1.00 | 0.000 |

## en, first segment

| system | own cosine | contrast | MRR |
|---|---|---|---|
| caption | 0.124 (0.087 to 0.165) | 0.061 (0.025 to 0.101) | 0.322 (0.203 to 0.470) |
| f1 | 0.167 (0.127 to 0.208) | 0.065 (0.028 to 0.102) | 0.321 (0.219 to 0.443) |
| f1_retrieve | 0.136 (0.098 to 0.178) | 0.052 (0.016 to 0.091) | 0.297 (0.192 to 0.429) |
| midi_llm | 0.080 (0.051 to 0.110) | 0.021 (-0.012 to 0.055) | 0.239 (0.164 to 0.340) |
| s50 | 0.142 (0.099 to 0.188) | 0.071 (0.031 to 0.113) | 0.338 (0.215 to 0.488) |
| text2midi | 0.091 (0.060 to 0.124) | 0.030 (-0.004 to 0.066) | 0.257 (0.166 to 0.379) |

en, first segment, own cosine: share of draws where the row beats the column

| | caption | f1 | f1_retrieve | midi_llm | s50 | text2midi |
|---|---|---|---|---|---|---|
| caption |  | 0.000 | 0.004 | 0.998 | 0.022 | 0.996 |
| f1 | 1.000 |  | 0.998 | 1.000 | 0.997 | 1.000 |
| f1_retrieve | 0.996 | 0.002 |  | 1.000 | 0.236 | 1.000 |
| midi_llm | 0.002 | 0.000 | 0.000 |  | 0.000 | 0.151 |
| s50 | 0.978 | 0.003 | 0.763 | 1.000 |  | 1.000 |
| text2midi | 0.004 | 0.000 | 0.000 | 0.849 | 0.000 |  |

en, first segment, contrast: share of draws where the row beats the column

| | caption | f1 | f1_retrieve | midi_llm | s50 | text2midi |
|---|---|---|---|---|---|---|
| caption |  | 0.281 | 0.946 | 1.000 | 0.026 | 1.000 |
| f1 | 0.719 |  | 0.970 | 1.000 | 0.243 | 1.000 |
| f1_retrieve | 0.054 | 0.030 |  | 0.998 | 0.005 | 0.976 |
| midi_llm | 0.000 | 0.000 | 0.002 |  | 0.000 | 0.093 |
| s50 | 0.974 | 0.757 | 0.995 | 1.000 |  | 1.000 |
| text2midi | 0.000 | 0.000 | 0.024 | 0.907 | 0.000 |  |

en, first segment, MRR: share of draws where the row beats the column

| | caption | f1 | f1_retrieve | midi_llm | s50 | text2midi |
|---|---|---|---|---|---|---|
| caption |  | 0.470 | 0.949 | 1.000 | 0.079 | 0.995 |
| f1 | 0.529 |  | 0.909 | 1.000 | 0.236 | 1.000 |
| f1_retrieve | 0.051 | 0.091 |  | 0.994 | 0.018 | 0.896 |
| midi_llm | 0.000 | 0.000 | 0.006 |  | 0.000 | 0.120 |
| s50 | 0.921 | 0.764 | 0.982 | 1.000 |  | 1.000 |
| text2midi | 0.005 | 0.000 | 0.104 | 0.880 | 0.000 |  |

## en, whole file (the authors' feature)

| system | own cosine | contrast | MRR |
|---|---|---|---|
| caption | 0.125 (0.088 to 0.166) | 0.061 (0.025 to 0.101) | 0.321 (0.203 to 0.469) |
| f1 | 0.168 (0.128 to 0.208) | 0.065 (0.028 to 0.102) | 0.321 (0.219 to 0.444) |
| f1_retrieve | 0.136 (0.098 to 0.178) | 0.052 (0.016 to 0.091) | 0.297 (0.192 to 0.429) |
| midi_llm | 0.080 (0.051 to 0.110) | 0.021 (-0.012 to 0.055) | 0.239 (0.164 to 0.340) |
| s50 | 0.144 (0.100 to 0.190) | 0.071 (0.031 to 0.114) | 0.338 (0.214 to 0.489) |
| text2midi | 0.091 (0.060 to 0.124) | 0.030 (-0.004 to 0.066) | 0.257 (0.166 to 0.379) |

## uk, first segment

| system | own cosine | contrast | MRR |
|---|---|---|---|
| caption | 0.077 (0.046 to 0.109) | 0.030 (-0.004 to 0.065) | 0.389 (0.249 to 0.546) |
| f1 | 0.099 (0.068 to 0.128) | 0.030 (-0.004 to 0.061) | 0.389 (0.251 to 0.533) |
| f1_retrieve | 0.079 (0.048 to 0.108) | 0.023 (-0.010 to 0.054) | 0.356 (0.227 to 0.505) |
| midi_llm | 0.040 (0.009 to 0.068) | 0.001 (-0.033 to 0.032) | 0.259 (0.163 to 0.375) |
| s50 | 0.086 (0.049 to 0.122) | 0.033 (-0.005 to 0.072) | 0.396 (0.231 to 0.577) |
| text2midi | 0.039 (0.009 to 0.069) | 0.001 (-0.028 to 0.030) | 0.279 (0.168 to 0.412) |

uk, first segment, own cosine: share of draws where the row beats the column

| | caption | f1 | f1_retrieve | midi_llm | s50 | text2midi |
|---|---|---|---|---|---|---|
| caption |  | 0.021 | 0.406 | 1.000 | 0.079 | 1.000 |
| f1 | 0.979 |  | 0.999 | 1.000 | 0.919 | 1.000 |
| f1_retrieve | 0.594 | 0.001 |  | 1.000 | 0.191 | 1.000 |
| midi_llm | 0.000 | 0.000 | 0.000 |  | 0.000 | 0.535 |
| s50 | 0.921 | 0.081 | 0.809 | 1.000 |  | 1.000 |
| text2midi | 0.001 | 0.000 | 0.000 | 0.465 | 0.000 |  |

uk, first segment, contrast: share of draws where the row beats the column

| | caption | f1 | f1_retrieve | midi_llm | s50 | text2midi |
|---|---|---|---|---|---|---|
| caption |  | 0.528 | 0.972 | 0.998 | 0.284 | 0.989 |
| f1 | 0.472 |  | 0.897 | 1.000 | 0.325 | 0.997 |
| f1_retrieve | 0.028 | 0.103 |  | 0.985 | 0.084 | 0.965 |
| midi_llm | 0.002 | 0.000 | 0.015 |  | 0.000 | 0.467 |
| s50 | 0.716 | 0.675 | 0.916 | 1.000 |  | 0.998 |
| text2midi | 0.011 | 0.003 | 0.035 | 0.533 | 0.002 |  |

uk, first segment, MRR: share of draws where the row beats the column

| | caption | f1 | f1_retrieve | midi_llm | s50 | text2midi |
|---|---|---|---|---|---|---|
| caption |  | 0.496 | 0.991 | 0.996 | 0.447 | 0.954 |
| f1 | 0.504 |  | 0.792 | 1.000 | 0.429 | 0.971 |
| f1_retrieve | 0.009 | 0.208 |  | 0.963 | 0.143 | 0.862 |
| midi_llm | 0.004 | 0.000 | 0.037 |  | 0.008 | 0.328 |
| s50 | 0.553 | 0.571 | 0.857 | 0.992 |  | 0.927 |
| text2midi | 0.046 | 0.029 | 0.138 | 0.672 | 0.073 |  |

## uk, whole file (the authors' feature)

| system | own cosine | contrast | MRR |
|---|---|---|---|
| caption | 0.078 (0.046 to 0.110) | 0.031 (-0.004 to 0.065) | 0.389 (0.249 to 0.546) |
| f1 | 0.099 (0.069 to 0.129) | 0.030 (-0.004 to 0.061) | 0.389 (0.251 to 0.533) |
| f1_retrieve | 0.080 (0.049 to 0.109) | 0.023 (-0.010 to 0.054) | 0.354 (0.225 to 0.503) |
| midi_llm | 0.040 (0.009 to 0.068) | 0.001 (-0.033 to 0.032) | 0.259 (0.163 to 0.375) |
| s50 | 0.088 (0.051 to 0.124) | 0.034 (-0.005 to 0.073) | 0.398 (0.231 to 0.582) |
| text2midi | 0.039 (0.009 to 0.069) | 0.001 (-0.028 to 0.030) | 0.279 (0.168 to 0.412) |
