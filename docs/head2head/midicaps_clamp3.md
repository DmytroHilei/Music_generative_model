# CLaMP 3 scores, midicaps (2026-10-31 09:32)

C2 checkpoint, every file through read_midi -> tokens_to_midi. 10000 bootstrap draws over prompts (seed 1234), 95% percentile interval. Up to 100 other captions per prompt. first = the file's first segment of 512 patches, the same input size for all rows; global = the authors' whole-file feature, which favours longer files (module docstring).

## Length

| system | files | patches | segments | files over one segment |
|---|---|---|---|---|
| f1 | 495 | 293 | 1.12 | 0.121 |
| midi_llm | 495 | 267 | 1.00 | 0.000 |
| real | 495 | 1491 | 3.43 | 0.695 |
| text2midi | 492 | 188 | 1.00 | 0.000 |

## en, first segment

| system | own cosine | contrast | MRR |
|---|---|---|---|
| f1 | 0.043 (0.036 to 0.049) | 0.046 (0.040 to 0.053) | 0.122 (0.104 to 0.142) |
| midi_llm | 0.090 (0.083 to 0.097) | 0.105 (0.098 to 0.113) | 0.276 (0.246 to 0.307) |
| real | 0.126 (0.120 to 0.133) | 0.154 (0.147 to 0.160) | 0.469 (0.435 to 0.504) |
| real_shuffled | -0.025 (-0.031 to -0.018) | 0.003 (-0.003 to 0.009) | 0.052 (0.043 to 0.063) |
| text2midi | 0.063 (0.057 to 0.069) | 0.066 (0.060 to 0.072) | 0.156 (0.135 to 0.178) |

en, first segment, own cosine: share of draws where the row beats the column

| | f1 | midi_llm | real | real_shuffled | text2midi |
|---|---|---|---|---|---|
| f1 |  | 0.000 | 0.000 | 1.000 | 0.000 |
| midi_llm | 1.000 |  | 0.000 | 1.000 | 1.000 |
| real | 1.000 | 1.000 |  | 1.000 | 1.000 |
| real_shuffled | 0.000 | 0.000 | 0.000 |  | 0.000 |
| text2midi | 1.000 | 0.000 | 0.000 | 1.000 |  |

en, first segment, contrast: share of draws where the row beats the column

| | f1 | midi_llm | real | real_shuffled | text2midi |
|---|---|---|---|---|---|
| f1 |  | 0.000 | 0.000 | 1.000 | 0.000 |
| midi_llm | 1.000 |  | 0.000 | 1.000 | 1.000 |
| real | 1.000 | 1.000 |  | 1.000 | 1.000 |
| real_shuffled | 0.000 | 0.000 | 0.000 |  | 0.000 |
| text2midi | 1.000 | 0.000 | 0.000 | 1.000 |  |

en, first segment, MRR: share of draws where the row beats the column

| | f1 | midi_llm | real | real_shuffled | text2midi |
|---|---|---|---|---|---|
| f1 |  | 0.000 | 0.000 | 1.000 | 0.002 |
| midi_llm | 1.000 |  | 0.000 | 1.000 | 1.000 |
| real | 1.000 | 1.000 |  | 1.000 | 1.000 |
| real_shuffled | 0.000 | 0.000 | 0.000 |  | 0.000 |
| text2midi | 0.998 | 0.000 | 0.000 | 1.000 |  |

## en, whole file (the authors' feature)

| system | own cosine | contrast | MRR |
|---|---|---|---|
| f1 | 0.043 (0.037 to 0.049) | 0.046 (0.040 to 0.052) | 0.121 (0.103 to 0.141) |
| midi_llm | 0.090 (0.083 to 0.097) | 0.105 (0.098 to 0.113) | 0.276 (0.246 to 0.307) |
| real | 0.140 (0.134 to 0.146) | 0.161 (0.155 to 0.167) | 0.485 (0.451 to 0.519) |
| real_shuffled | -0.017 (-0.023 to -0.010) | 0.005 (-0.002 to 0.011) | 0.052 (0.043 to 0.063) |
| text2midi | 0.063 (0.057 to 0.069) | 0.066 (0.060 to 0.072) | 0.156 (0.135 to 0.178) |

## Frechet to the real pool, first segment (lower is better)

N = 492 files per row, 860 reference files outside real/, on the raw features. 1000 draws over files, each row's draws centred on its value (a resample repeats files, which alone raises the distance). real = the floor at this N.

| system | Frechet |
|---|---|
| f1 | 192.626 (186.150 to 198.738) |
| midi_llm | 111.958 (108.244 to 116.107) |
| real | 32.774 (29.963 to 35.598) |
| text2midi | 153.602 (149.425 to 158.249) |

Frechet, first segment: share of draws where the row beats the column

| | f1 | midi_llm | real | text2midi |
|---|---|---|---|---|
| f1 |  | 0.000 | 0.000 | 0.000 |
| midi_llm | 1.000 |  | 0.000 | 1.000 |
| real | 1.000 | 1.000 |  | 1.000 |
| text2midi | 1.000 | 0.000 | 0.000 |  |

## Frechet to the real pool, whole file (the authors' feature) (lower is better)

N = 492 files per row, 860 reference files outside real/, on the raw features. 1000 draws over files, each row's draws centred on its value (a resample repeats files, which alone raises the distance). real = the floor at this N.

| system | Frechet |
|---|---|
| f1 | 188.230 (182.133 to 194.052) |
| midi_llm | 118.596 (114.924 to 122.664) |
| real | 29.427 (26.995 to 31.923) |
| text2midi | 155.472 (151.350 to 159.955) |

Frechet, whole file (the authors' feature): share of draws where the row beats the column

| | f1 | midi_llm | real | text2midi |
|---|---|---|---|---|
| f1 |  | 0.000 | 0.000 | 0.000 |
| midi_llm | 1.000 |  | 0.000 | 1.000 |
| real | 1.000 | 1.000 |  | 1.000 |
| text2midi | 1.000 | 0.000 | 0.000 |  |
