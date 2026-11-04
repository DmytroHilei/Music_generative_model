# CLaMP 3 continuation scores, piano96, first 1000 new notes (2026-11-04 17:53)

C2 checkpoint, every file cut after its prompt and normalised (h2h_clamp3.py). Target = the embedding of the prompt's real continuation: own = cosine to it, contrast = own minus the mean cosine to 95 other prompts' real continuations, MRR = its reciprocal rank among them. prompt = the prompt itself (128 notes), real_shuffled = another prompt's real continuation (floor). 10000 bootstrap draws over the 96 prompts, 95% percentile interval. first = the first segment of 512 patches; global = the authors' whole-file feature.

## Length

| system | files | patches | segments | files over one segment |
|---|---|---|---|---|
| aria | 96 | 389 | 1.00 | 0.000 |
| long_450m | 96 | 333 | 1.00 | 0.000 |
| piano | 96 | 216 | 1.00 | 0.000 |
| prompt | 96 | 59 | 1.00 | 0.000 |
| ua_2048_v2 | 96 | 369 | 1.00 | 0.000 |
| real | 96 | 408 | 1.00 | 0.000 |

## piano96, first segment

| system | own cosine | contrast | MRR |
|---|---|---|---|
| aria | 0.658 (0.632 to 0.685) | 0.216 (0.194 to 0.238) | 0.578 (0.495 to 0.663) |
| long_450m | 0.689 (0.667 to 0.710) | 0.211 (0.189 to 0.232) | 0.547 (0.467 to 0.627) |
| piano | 0.604 (0.580 to 0.627) | 0.154 (0.134 to 0.175) | 0.428 (0.347 to 0.511) |
| prompt | 0.576 (0.561 to 0.591) | 0.165 (0.149 to 0.181) | 0.519 (0.440 to 0.598) |
| ua_2048_v2 | 0.649 (0.630 to 0.668) | 0.152 (0.133 to 0.171) | 0.424 (0.344 to 0.505) |
| real_shuffled | 0.492 (0.470 to 0.515) | 0.009 (-0.012 to 0.031) | 0.052 (0.037 to 0.071) |

piano96, first segment, own cosine: share of draws where the row beats the column

| | aria | long_450m | piano | prompt | ua_2048_v2 | real_shuffled |
|---|---|---|---|---|---|---|
| aria |  | 0.012 | 1.000 | 1.000 | 0.737 | 1.000 |
| long_450m | 0.988 |  | 1.000 | 1.000 | 0.998 | 1.000 |
| piano | 0.000 | 0.000 |  | 0.986 | 0.000 | 1.000 |
| prompt | 0.000 | 0.000 | 0.014 |  | 0.000 | 1.000 |
| ua_2048_v2 | 0.264 | 0.002 | 1.000 | 1.000 |  | 1.000 |
| real_shuffled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |

piano96, first segment, contrast: share of draws where the row beats the column

| | aria | long_450m | piano | prompt | ua_2048_v2 | real_shuffled |
|---|---|---|---|---|---|---|
| aria |  | 0.666 | 1.000 | 1.000 | 1.000 | 1.000 |
| long_450m | 0.334 |  | 1.000 | 1.000 | 1.000 | 1.000 |
| piano | 0.000 | 0.000 |  | 0.159 | 0.570 | 1.000 |
| prompt | 0.000 | 0.000 | 0.841 |  | 0.862 | 1.000 |
| ua_2048_v2 | 0.000 | 0.000 | 0.430 | 0.138 |  | 1.000 |
| real_shuffled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |

## piano96, whole file (the authors' feature)

| system | own cosine | contrast | MRR |
|---|---|---|---|
| aria | 0.658 (0.632 to 0.685) | 0.216 (0.194 to 0.238) | 0.578 (0.495 to 0.663) |
| long_450m | 0.689 (0.667 to 0.710) | 0.211 (0.189 to 0.232) | 0.547 (0.467 to 0.627) |
| piano | 0.604 (0.580 to 0.627) | 0.154 (0.134 to 0.175) | 0.428 (0.347 to 0.511) |
| prompt | 0.576 (0.561 to 0.591) | 0.165 (0.149 to 0.181) | 0.519 (0.440 to 0.598) |
| ua_2048_v2 | 0.649 (0.630 to 0.668) | 0.152 (0.133 to 0.171) | 0.424 (0.344 to 0.505) |
| real_shuffled | 0.492 (0.470 to 0.515) | 0.009 (-0.012 to 0.031) | 0.052 (0.037 to 0.071) |

## Frechet to the real continuations, first segment (lower is better; side number)

N = 96 files per row against the 96 real continuations, raw features: at this N the covariances are rank N - 1 of 768. 300 draws over files, each row's draws centred on its value.

| system | Frechet |
|---|---|
| aria | 122.399 (110.176 to 137.103) |
| long_450m | 91.081 (83.321 to 100.804) |
| piano | 155.264 (141.297 to 170.831) |
| prompt | 219.194 (211.691 to 227.749) |
| ua_2048_v2 | 103.311 (95.572 to 112.898) |

Frechet, first segment: share of draws where the row beats the column

| | aria | long_450m | piano | prompt | ua_2048_v2 |
|---|---|---|---|---|---|
| aria |  | 0.000 | 1.000 | 1.000 | 0.000 |
| long_450m | 1.000 |  | 1.000 | 1.000 | 1.000 |
| piano | 0.000 | 0.000 |  | 1.000 | 0.000 |
| prompt | 0.000 | 0.000 | 0.000 |  | 0.000 |
| ua_2048_v2 | 1.000 | 0.000 | 1.000 | 1.000 |  |

## Frechet to the real continuations, whole file (the authors' feature) (lower is better; side number)

N = 96 files per row against the 96 real continuations, raw features: at this N the covariances are rank N - 1 of 768. 300 draws over files, each row's draws centred on its value.

| system | Frechet |
|---|---|
| aria | 122.399 (110.176 to 137.103) |
| long_450m | 91.081 (83.321 to 100.804) |
| piano | 155.264 (141.297 to 170.831) |
| prompt | 219.194 (211.691 to 227.749) |
| ua_2048_v2 | 103.311 (95.572 to 112.898) |

Frechet, whole file (the authors' feature): share of draws where the row beats the column

| | aria | long_450m | piano | prompt | ua_2048_v2 |
|---|---|---|---|---|---|
| aria |  | 0.000 | 1.000 | 1.000 | 0.000 |
| long_450m | 1.000 |  | 1.000 | 1.000 | 1.000 |
| piano | 0.000 | 0.000 |  | 1.000 | 0.000 |
| prompt | 0.000 | 0.000 | 0.000 |  | 0.000 |
| ua_2048_v2 | 1.000 | 0.000 | 1.000 | 1.000 |  |
