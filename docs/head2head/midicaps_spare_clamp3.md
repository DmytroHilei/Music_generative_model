# CLaMP 3 scores, midicaps_spare (2026-11-04 20:50)

C2 checkpoint, every file through read_midi -> tokens_to_midi. 10000 bootstrap draws over prompts (seed 1234), 95% percentile interval. Up to 100 other captions per prompt. first = the file's first segment of 512 patches, the same input size for all rows; global = the authors' whole-file feature, which favours longer files (module docstring).

## Length

| system | files | patches | segments | files over one segment |
|---|---|---|---|---|
| f1 | 100 | 287 | 1.10 | 0.100 |
| f1_cfg4 | 100 | 318 | 1.15 | 0.150 |
| f1_free | 100 | 271 | 1.09 | 0.090 |
| f1_retrieve | 100 | 291 | 1.06 | 0.060 |
| f1_topp09 | 100 | 515 | 1.48 | 0.480 |
| real | 100 | 1295 | 3.11 | 0.630 |

## en, first segment

| system | own cosine | contrast | MRR |
|---|---|---|---|
| f1 | 0.027 (0.014 to 0.041) | 0.042 (0.028 to 0.056) | 0.118 (0.077 to 0.165) |
| f1_cfg4 | 0.031 (0.016 to 0.046) | 0.044 (0.028 to 0.060) | 0.123 (0.087 to 0.164) |
| f1_free | 0.024 (0.011 to 0.037) | 0.033 (0.020 to 0.046) | 0.090 (0.058 to 0.130) |
| f1_retrieve | 0.041 (0.026 to 0.055) | 0.056 (0.041 to 0.071) | 0.150 (0.104 to 0.202) |
| f1_topp09 | 0.039 (0.022 to 0.056) | 0.063 (0.048 to 0.079) | 0.141 (0.100 to 0.186) |
| real | 0.132 (0.117 to 0.148) | 0.160 (0.144 to 0.175) | 0.454 (0.381 to 0.530) |
| real_shuffled | -0.024 (-0.039 to -0.010) | 0.002 (-0.012 to 0.015) | 0.042 (0.030 to 0.057) |

en, first segment, own cosine: share of draws where the row beats the column

| | f1 | f1_cfg4 | f1_free | f1_retrieve | f1_topp09 | real | real_shuffled |
|---|---|---|---|---|---|---|---|
| f1 |  | 0.302 | 0.813 | 0.030 | 0.062 | 0.000 | 1.000 |
| f1_cfg4 | 0.698 |  | 0.844 | 0.111 | 0.154 | 0.000 | 1.000 |
| f1_free | 0.187 | 0.156 |  | 0.008 | 0.034 | 0.000 | 1.000 |
| f1_retrieve | 0.970 | 0.889 | 0.992 |  | 0.562 | 0.000 | 1.000 |
| f1_topp09 | 0.938 | 0.846 | 0.966 | 0.438 |  | 0.000 | 1.000 |
| real | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |  | 1.000 |
| real_shuffled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |

en, first segment, contrast: share of draws where the row beats the column

| | f1 | f1_cfg4 | f1_free | f1_retrieve | f1_topp09 | real | real_shuffled |
|---|---|---|---|---|---|---|---|
| f1 |  | 0.389 | 0.996 | 0.020 | 0.003 | 0.000 | 1.000 |
| f1_cfg4 | 0.611 |  | 0.942 | 0.067 | 0.012 | 0.000 | 1.000 |
| f1_free | 0.004 | 0.058 |  | 0.000 | 0.000 | 0.000 | 1.000 |
| f1_retrieve | 0.980 | 0.933 | 1.000 |  | 0.198 | 0.000 | 1.000 |
| f1_topp09 | 0.997 | 0.988 | 1.000 | 0.802 |  | 0.000 | 1.000 |
| real | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |  | 1.000 |
| real_shuffled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |

en, first segment, MRR: share of draws where the row beats the column

| | f1 | f1_cfg4 | f1_free | f1_retrieve | f1_topp09 | real | real_shuffled |
|---|---|---|---|---|---|---|---|
| f1 |  | 0.412 | 0.987 | 0.111 | 0.184 | 0.000 | 1.000 |
| f1_cfg4 | 0.588 |  | 0.915 | 0.168 | 0.241 | 0.000 | 1.000 |
| f1_free | 0.013 | 0.085 |  | 0.003 | 0.025 | 0.000 | 0.999 |
| f1_retrieve | 0.889 | 0.832 | 0.997 |  | 0.612 | 0.000 | 1.000 |
| f1_topp09 | 0.816 | 0.759 | 0.975 | 0.389 |  | 0.000 | 1.000 |
| real | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |  | 1.000 |
| real_shuffled | 0.000 | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 |  |

## en, whole file (the authors' feature)

| system | own cosine | contrast | MRR |
|---|---|---|---|
| f1 | 0.028 (0.014 to 0.042) | 0.042 (0.028 to 0.056) | 0.119 (0.078 to 0.166) |
| f1_cfg4 | 0.031 (0.016 to 0.047) | 0.044 (0.028 to 0.060) | 0.122 (0.086 to 0.163) |
| f1_free | 0.024 (0.011 to 0.037) | 0.033 (0.020 to 0.046) | 0.091 (0.058 to 0.130) |
| f1_retrieve | 0.041 (0.026 to 0.055) | 0.056 (0.041 to 0.071) | 0.150 (0.104 to 0.202) |
| f1_topp09 | 0.045 (0.028 to 0.061) | 0.064 (0.049 to 0.080) | 0.144 (0.101 to 0.192) |
| real | 0.144 (0.130 to 0.159) | 0.167 (0.151 to 0.182) | 0.473 (0.399 to 0.552) |
| real_shuffled | -0.018 (-0.033 to -0.004) | 0.002 (-0.011 to 0.016) | 0.042 (0.030 to 0.056) |
