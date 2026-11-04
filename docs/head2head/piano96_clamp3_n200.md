# CLaMP 3 continuation scores, piano96, first 200 new notes (2026-11-04 17:55)

C2 checkpoint, every file cut after its prompt and normalised (h2h_clamp3.py). Target = the embedding of the prompt's real continuation: own = cosine to it, contrast = own minus the mean cosine to 95 other prompts' real continuations, MRR = its reciprocal rank among them. prompt = the prompt itself (128 notes), real_shuffled = another prompt's real continuation (floor). 10000 bootstrap draws over the 96 prompts, 95% percentile interval. first = the first segment of 512 patches; global = the authors' whole-file feature.

## Length

| system | files | patches | segments | files over one segment |
|---|---|---|---|---|
| aria | 96 | 87 | 1.00 | 0.000 |
| long_450m | 96 | 87 | 1.00 | 0.000 |
| piano | 96 | 87 | 1.00 | 0.000 |
| prompt | 96 | 59 | 1.00 | 0.000 |
| ua_2048_v2 | 96 | 87 | 1.00 | 0.000 |
| real | 96 | 88 | 1.00 | 0.000 |

## piano96, first segment

| system | own cosine | contrast | MRR |
|---|---|---|---|
| aria | 0.754 (0.734 to 0.772) | 0.208 (0.191 to 0.224) | 0.682 (0.602 to 0.758) |
| long_450m | 0.751 (0.733 to 0.769) | 0.181 (0.164 to 0.198) | 0.545 (0.461 to 0.627) |
| piano | 0.718 (0.698 to 0.737) | 0.170 (0.153 to 0.188) | 0.500 (0.419 to 0.581) |
| prompt | 0.770 (0.755 to 0.786) | 0.206 (0.189 to 0.223) | 0.694 (0.614 to 0.773) |
| ua_2048_v2 | 0.730 (0.713 to 0.746) | 0.154 (0.137 to 0.170) | 0.482 (0.401 to 0.563) |
| real_shuffled | 0.571 (0.548 to 0.593) | 0.003 (-0.017 to 0.022) | 0.047 (0.034 to 0.064) |

piano96, first segment, own cosine: share of draws where the row beats the column

| | aria | long_450m | piano | prompt | ua_2048_v2 | real_shuffled |
|---|---|---|---|---|---|---|
| aria |  | 0.578 | 1.000 | 0.065 | 0.976 | 1.000 |
| long_450m | 0.422 |  | 0.999 | 0.033 | 0.974 | 1.000 |
| piano | 0.001 | 0.001 |  | 0.000 | 0.140 | 1.000 |
| prompt | 0.935 | 0.967 | 1.000 |  | 1.000 | 1.000 |
| ua_2048_v2 | 0.024 | 0.026 | 0.860 | 0.000 |  | 1.000 |
| real_shuffled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |

piano96, first segment, contrast: share of draws where the row beats the column

| | aria | long_450m | piano | prompt | ua_2048_v2 | real_shuffled |
|---|---|---|---|---|---|---|
| aria |  | 0.992 | 1.000 | 0.553 | 1.000 | 1.000 |
| long_450m | 0.008 |  | 0.870 | 0.007 | 0.996 | 1.000 |
| piano | 0.000 | 0.130 |  | 0.000 | 0.955 | 1.000 |
| prompt | 0.447 | 0.993 | 1.000 |  | 1.000 | 1.000 |
| ua_2048_v2 | 0.000 | 0.004 | 0.045 | 0.000 |  | 1.000 |
| real_shuffled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |

## piano96, whole file (the authors' feature)

| system | own cosine | contrast | MRR |
|---|---|---|---|
| aria | 0.754 (0.734 to 0.772) | 0.208 (0.191 to 0.224) | 0.682 (0.602 to 0.758) |
| long_450m | 0.751 (0.733 to 0.769) | 0.181 (0.164 to 0.198) | 0.545 (0.461 to 0.627) |
| piano | 0.718 (0.698 to 0.737) | 0.170 (0.153 to 0.188) | 0.500 (0.419 to 0.581) |
| prompt | 0.770 (0.755 to 0.786) | 0.206 (0.189 to 0.223) | 0.694 (0.614 to 0.773) |
| ua_2048_v2 | 0.730 (0.713 to 0.746) | 0.154 (0.137 to 0.170) | 0.482 (0.401 to 0.563) |
| real_shuffled | 0.571 (0.548 to 0.593) | 0.003 (-0.017 to 0.022) | 0.047 (0.034 to 0.064) |

## Frechet to the real continuations, first segment (lower is better; side number)

N = 96 files per row against the 96 real continuations, raw features: at this N the covariances are rank N - 1 of 768. 300 draws over files, each row's draws centred on its value.

| system | Frechet |
|---|---|
| aria | 78.336 (70.822 to 89.514) |
| long_450m | 64.934 (58.529 to 72.493) |
| piano | 88.715 (78.317 to 100.515) |
| prompt | 75.336 (69.329 to 83.154) |
| ua_2048_v2 | 72.834 (66.847 to 80.123) |

Frechet, first segment: share of draws where the row beats the column

| | aria | long_450m | piano | prompt | ua_2048_v2 |
|---|---|---|---|---|---|
| aria |  | 0.000 | 0.977 | 0.203 | 0.087 |
| long_450m | 1.000 |  | 1.000 | 1.000 | 0.997 |
| piano | 0.023 | 0.000 |  | 0.000 | 0.000 |
| prompt | 0.797 | 0.000 | 1.000 |  | 0.170 |
| ua_2048_v2 | 0.913 | 0.003 | 1.000 | 0.830 |  |

## Frechet to the real continuations, whole file (the authors' feature) (lower is better; side number)

N = 96 files per row against the 96 real continuations, raw features: at this N the covariances are rank N - 1 of 768. 300 draws over files, each row's draws centred on its value.

| system | Frechet |
|---|---|
| aria | 78.336 (70.822 to 89.514) |
| long_450m | 64.934 (58.529 to 72.493) |
| piano | 88.715 (78.317 to 100.515) |
| prompt | 75.336 (69.329 to 83.154) |
| ua_2048_v2 | 72.834 (66.847 to 80.123) |

Frechet, whole file (the authors' feature): share of draws where the row beats the column

| | aria | long_450m | piano | prompt | ua_2048_v2 |
|---|---|---|---|---|---|
| aria |  | 0.000 | 0.977 | 0.203 | 0.087 |
| long_450m | 1.000 |  | 1.000 | 1.000 | 0.997 |
| piano | 0.023 | 0.000 |  | 0.000 | 0.000 |
| prompt | 0.797 | 0.000 | 1.000 |  | 0.170 |
| ua_2048_v2 | 0.913 | 0.003 | 1.000 | 0.830 |  |
