# MidiCaps test set (2026-10-31 09:22)

495 captions, one file each, first 40 s, tempo and key estimated from the notes (h2h_midicaps.py). 10000 bootstrap draws over prompts, 95% percentile interval. A missing file scores 0. real = the same estimators on the true files (ceiling); real_shuffled = against another prompt's targets (floor). Caption match uses our caption builder's thresholds.

| system | files | TB | TBT | CK | CKD | instrument recall | family recall | instrument Jaccard | caption match (ours) |
|---|---|---|---|---|---|---|---|---|---|
| f1 | 495 | 0.244 (0.206 to 0.283) | 0.620 (0.576 to 0.661) | 0.042 (0.026 to 0.061) | 0.075 (0.053 to 0.099) | 0.425 (0.397 to 0.454) | 0.728 (0.703 to 0.753) | 0.250 (0.227 to 0.274) | 0.526 (0.510 to 0.542) |
| midi_llm | 495 | 0.400 (0.358 to 0.442) | 0.640 (0.598 to 0.683) | 0.317 (0.277 to 0.360) | 0.420 (0.376 to 0.465) | 0.664 (0.639 to 0.689) | 0.812 (0.792 to 0.831) | 0.554 (0.527 to 0.581) | 0.609 (0.592 to 0.625) |
| real | 495 | 0.608 (0.566 to 0.651) | 0.739 (0.701 to 0.778) | 0.586 (0.543 to 0.628) | 0.636 (0.594 to 0.679) | 0.809 (0.787 to 0.831) | 0.856 (0.836 to 0.874) | 0.671 (0.645 to 0.697) | 0.711 (0.694 to 0.729) |
| text2midi | 492 | 0.345 (0.303 to 0.388) | 0.640 (0.598 to 0.681) | 0.309 (0.269 to 0.349) | 0.378 (0.335 to 0.420) | 0.426 (0.397 to 0.454) | 0.685 (0.659 to 0.710) | 0.221 (0.206 to 0.237) | 0.557 (0.539 to 0.575) |
| real_shuffled | 495 | 0.248 (0.212 to 0.287) | 0.600 (0.558 to 0.642) | 0.051 (0.032 to 0.071) | 0.099 (0.073 to 0.127) | 0.167 (0.145 to 0.190) | 0.392 (0.361 to 0.422) | 0.093 (0.079 to 0.108) | 0.388 (0.372 to 0.405) |

| system | seconds | notes | nps | runaway | copies |
|---|---|---|---|---|---|
| f1 | 40.4 | 546 | 13.6 | 0.0357 | 0.0015 |
| midi_llm | 28.8 | 569 | 25.3 | 0.148 | 0.0211 |
| real | 37.9 | 660 | 16.9 | 0.0593 | 0.1 |
| text2midi | 22.2 | 433 | 22.4 | 0.0722 | 0.00824 |
| real_shuffled | 37.9 | 660 | 16.9 | 0.0593 | 0.1 |

TB: share of draws where the row beats the column

| | f1 | midi_llm | real | text2midi | real_shuffled |
|---|---|---|---|---|---|
| f1 |  | 0.000 | 0.000 | 0.000 | 0.421 |
| midi_llm | 1.000 |  | 0.000 | 0.982 | 1.000 |
| real | 1.000 | 1.000 |  | 1.000 | 1.000 |
| text2midi | 1.000 | 0.015 | 0.000 |  | 1.000 |
| real_shuffled | 0.546 | 0.000 | 0.000 | 0.000 |  |

TBT: share of draws where the row beats the column

| | f1 | midi_llm | real | text2midi | real_shuffled |
|---|---|---|---|---|---|
| f1 |  | 0.213 | 0.000 | 0.216 | 0.733 |
| midi_llm | 0.763 |  | 0.000 | 0.475 | 0.908 |
| real | 1.000 | 1.000 |  | 1.000 | 1.000 |
| text2midi | 0.760 | 0.489 | 0.000 |  | 0.912 |
| real_shuffled | 0.244 | 0.081 | 0.000 | 0.077 |  |

CK: share of draws where the row beats the column

| | f1 | midi_llm | real | text2midi | real_shuffled |
|---|---|---|---|---|---|
| f1 |  | 0.000 | 0.000 | 0.000 | 0.250 |
| midi_llm | 1.000 |  | 0.000 | 0.605 | 1.000 |
| real | 1.000 | 1.000 |  | 1.000 | 1.000 |
| text2midi | 1.000 | 0.366 | 0.000 |  | 1.000 |
| real_shuffled | 0.700 | 0.000 | 0.000 | 0.000 |  |

CKD: share of draws where the row beats the column

| | f1 | midi_llm | real | text2midi | real_shuffled |
|---|---|---|---|---|---|
| f1 |  | 0.000 | 0.000 | 0.000 | 0.084 |
| midi_llm | 1.000 |  | 0.000 | 0.928 | 1.000 |
| real | 1.000 | 1.000 |  | 1.000 | 1.000 |
| text2midi | 1.000 | 0.064 | 0.000 |  | 1.000 |
| real_shuffled | 0.897 | 0.000 | 0.000 | 0.000 |  |

instrument recall: share of draws where the row beats the column

| | f1 | midi_llm | real | text2midi | real_shuffled |
|---|---|---|---|---|---|
| f1 |  | 0.000 | 0.000 | 0.498 | 1.000 |
| midi_llm | 1.000 |  | 0.000 | 1.000 | 1.000 |
| real | 1.000 | 1.000 |  | 1.000 | 1.000 |
| text2midi | 0.502 | 0.000 | 0.000 |  | 1.000 |
| real_shuffled | 0.000 | 0.000 | 0.000 | 0.000 |  |

family recall: share of draws where the row beats the column

| | f1 | midi_llm | real | text2midi | real_shuffled |
|---|---|---|---|---|---|
| f1 |  | 0.000 | 0.000 | 0.997 | 1.000 |
| midi_llm | 1.000 |  | 0.000 | 1.000 | 1.000 |
| real | 1.000 | 1.000 |  | 1.000 | 1.000 |
| text2midi | 0.003 | 0.000 | 0.000 |  | 1.000 |
| real_shuffled | 0.000 | 0.000 | 0.000 | 0.000 |  |

instrument Jaccard: share of draws where the row beats the column

| | f1 | midi_llm | real | text2midi | real_shuffled |
|---|---|---|---|---|---|
| f1 |  | 0.000 | 0.000 | 0.979 | 1.000 |
| midi_llm | 1.000 |  | 0.000 | 1.000 | 1.000 |
| real | 1.000 | 1.000 |  | 1.000 | 1.000 |
| text2midi | 0.021 | 0.000 | 0.000 |  | 1.000 |
| real_shuffled | 0.000 | 0.000 | 0.000 | 0.000 |  |

caption match (ours): share of draws where the row beats the column

| | f1 | midi_llm | real | text2midi | real_shuffled |
|---|---|---|---|---|---|
| f1 |  | 0.000 | 0.000 | 0.002 | 1.000 |
| midi_llm | 1.000 |  | 0.000 | 1.000 | 1.000 |
| real | 1.000 | 1.000 |  | 1.000 | 1.000 |
| text2midi | 0.998 | 0.000 | 0.000 |  | 1.000 |
| real_shuffled | 0.000 | 0.000 | 0.000 | 0.000 |  |
