# CLaMP 3 continuation scores, cont96, first 1000 new notes (2026-11-04 17:47)

C2 checkpoint, every file cut after its prompt and normalised (h2h_clamp3.py). Target = the embedding of the prompt's real continuation: own = cosine to it, contrast = own minus the mean cosine to 95 other prompts' real continuations, MRR = its reciprocal rank among them. prompt = the prompt itself (128 notes), real_shuffled = another prompt's real continuation (floor). 10000 bootstrap draws over the 96 prompts, 95% percentile interval. first = the first segment of 512 patches; global = the authors' whole-file feature.

## Length

| system | files | patches | segments | files over one segment |
|---|---|---|---|---|
| amt | 96 | 399 | 1.00 | 0.000 |
| f1 | 96 | 325 | 1.00 | 0.000 |
| long_450m | 96 | 365 | 1.00 | 0.000 |
| prompt | 96 | 62 | 1.00 | 0.000 |
| skytnt_large | 96 | 376 | 1.00 | 0.000 |
| skytnt_medium | 96 | 367 | 1.00 | 0.000 |
| real | 96 | 370 | 1.00 | 0.000 |

## cont96, first segment

| system | own cosine | contrast | MRR |
|---|---|---|---|
| amt | 0.648 (0.627 to 0.668) | 0.247 (0.227 to 0.267) | 0.640 (0.560 to 0.718) |
| f1 | 0.597 (0.573 to 0.621) | 0.213 (0.192 to 0.235) | 0.563 (0.484 to 0.643) |
| long_450m | 0.715 (0.694 to 0.736) | 0.317 (0.295 to 0.339) | 0.811 (0.746 to 0.872) |
| prompt | 0.645 (0.626 to 0.664) | 0.258 (0.239 to 0.279) | 0.669 (0.595 to 0.740) |
| skytnt_large | 0.583 (0.560 to 0.605) | 0.200 (0.178 to 0.222) | 0.537 (0.455 to 0.621) |
| skytnt_medium | 0.648 (0.626 to 0.670) | 0.260 (0.239 to 0.281) | 0.666 (0.586 to 0.741) |
| real_shuffled | 0.385 (0.364 to 0.408) | -0.014 (-0.034 to 0.007) | 0.043 (0.030 to 0.060) |

cont96, first segment, own cosine: share of draws where the row beats the column

| | amt | f1 | long_450m | prompt | skytnt_large | skytnt_medium | real_shuffled |
|---|---|---|---|---|---|---|---|
| amt |  | 0.999 | 0.000 | 0.587 | 1.000 | 0.512 | 1.000 |
| f1 | 0.001 |  | 0.000 | 0.000 | 0.859 | 0.000 | 1.000 |
| long_450m | 1.000 | 1.000 |  | 1.000 | 1.000 | 1.000 | 1.000 |
| prompt | 0.413 | 1.000 | 0.000 |  | 1.000 | 0.411 | 1.000 |
| skytnt_large | 0.000 | 0.141 | 0.000 | 0.000 |  | 0.000 | 1.000 |
| skytnt_medium | 0.488 | 1.000 | 0.000 | 0.589 | 1.000 |  | 1.000 |
| real_shuffled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |

cont96, first segment, contrast: share of draws where the row beats the column

| | amt | f1 | long_450m | prompt | skytnt_large | skytnt_medium | real_shuffled |
|---|---|---|---|---|---|---|---|
| amt |  | 0.992 | 0.000 | 0.207 | 1.000 | 0.168 | 1.000 |
| f1 | 0.008 |  | 0.000 | 0.000 | 0.865 | 0.000 | 1.000 |
| long_450m | 1.000 | 1.000 |  | 1.000 | 1.000 | 1.000 | 1.000 |
| prompt | 0.793 | 1.000 | 0.000 |  | 1.000 | 0.435 | 1.000 |
| skytnt_large | 0.000 | 0.135 | 0.000 | 0.000 |  | 0.000 | 1.000 |
| skytnt_medium | 0.832 | 1.000 | 0.000 | 0.565 | 1.000 |  | 1.000 |
| real_shuffled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |

## cont96 no_drums, first segment

| system | own cosine | contrast | MRR |
|---|---|---|---|
| amt | 0.655 (0.630 to 0.678) | 0.258 (0.235 to 0.281) | 0.643 (0.552 to 0.735) |
| f1 | 0.576 (0.549 to 0.604) | 0.205 (0.180 to 0.231) | 0.529 (0.430 to 0.628) |
| long_450m | 0.715 (0.692 to 0.739) | 0.321 (0.295 to 0.347) | 0.838 (0.760 to 0.908) |
| prompt | 0.640 (0.617 to 0.664) | 0.259 (0.233 to 0.283) | 0.645 (0.556 to 0.729) |
| skytnt_large | 0.572 (0.546 to 0.598) | 0.202 (0.175 to 0.228) | 0.549 (0.449 to 0.648) |
| skytnt_medium | 0.630 (0.604 to 0.655) | 0.250 (0.225 to 0.275) | 0.645 (0.548 to 0.738) |
| real_shuffled | 0.378 (0.352 to 0.406) | -0.024 (-0.048 to 0.002) | 0.039 (0.026 to 0.058) |

cont96 no_drums, first segment, own cosine: share of draws where the row beats the column

| | amt | f1 | long_450m | prompt | skytnt_large | skytnt_medium | real_shuffled |
|---|---|---|---|---|---|---|---|
| amt |  | 1.000 | 0.000 | 0.805 | 1.000 | 0.960 | 1.000 |
| f1 | 0.000 |  | 0.000 | 0.000 | 0.603 | 0.002 | 1.000 |
| long_450m | 1.000 | 1.000 |  | 1.000 | 1.000 | 1.000 | 1.000 |
| prompt | 0.195 | 1.000 | 0.000 |  | 1.000 | 0.783 | 1.000 |
| skytnt_large | 0.000 | 0.397 | 0.000 | 0.000 |  | 0.000 | 1.000 |
| skytnt_medium | 0.040 | 0.998 | 0.000 | 0.217 | 1.000 |  | 1.000 |
| real_shuffled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |

cont96 no_drums, first segment, contrast: share of draws where the row beats the column

| | amt | f1 | long_450m | prompt | skytnt_large | skytnt_medium | real_shuffled |
|---|---|---|---|---|---|---|---|
| amt |  | 0.999 | 0.000 | 0.494 | 1.000 | 0.698 | 1.000 |
| f1 | 0.001 |  | 0.000 | 0.000 | 0.591 | 0.003 | 1.000 |
| long_450m | 1.000 | 1.000 |  | 1.000 | 1.000 | 1.000 | 1.000 |
| prompt | 0.506 | 1.000 | 0.000 |  | 1.000 | 0.726 | 1.000 |
| skytnt_large | 0.000 | 0.409 | 0.000 | 0.000 |  | 0.000 | 1.000 |
| skytnt_medium | 0.302 | 0.997 | 0.000 | 0.274 | 1.000 |  | 1.000 |
| real_shuffled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |

## cont96 with_drums, first segment

| system | own cosine | contrast | MRR |
|---|---|---|---|
| amt | 0.634 (0.594 to 0.671) | 0.224 (0.189 to 0.260) | 0.632 (0.482 to 0.780) |
| f1 | 0.640 (0.596 to 0.682) | 0.230 (0.191 to 0.270) | 0.635 (0.493 to 0.775) |
| long_450m | 0.715 (0.676 to 0.757) | 0.307 (0.268 to 0.348) | 0.755 (0.637 to 0.864) |
| prompt | 0.655 (0.623 to 0.687) | 0.258 (0.225 to 0.293) | 0.718 (0.586 to 0.843) |
| skytnt_large | 0.605 (0.562 to 0.647) | 0.196 (0.155 to 0.238) | 0.513 (0.365 to 0.664) |
| skytnt_medium | 0.686 (0.645 to 0.726) | 0.281 (0.241 to 0.321) | 0.711 (0.574 to 0.840) |
| real_shuffled | 0.400 (0.363 to 0.439) | 0.007 (-0.025 to 0.040) | 0.051 (0.027 to 0.088) |

cont96 with_drums, first segment, own cosine: share of draws where the row beats the column

| | amt | f1 | long_450m | prompt | skytnt_large | skytnt_medium | real_shuffled |
|---|---|---|---|---|---|---|---|
| amt |  | 0.405 | 0.000 | 0.191 | 0.841 | 0.035 | 1.000 |
| f1 | 0.595 |  | 0.000 | 0.274 | 0.931 | 0.009 | 1.000 |
| long_450m | 1.000 | 1.000 |  | 1.000 | 1.000 | 0.963 | 1.000 |
| prompt | 0.809 | 0.726 | 0.000 |  | 0.998 | 0.066 | 1.000 |
| skytnt_large | 0.159 | 0.069 | 0.000 | 0.002 |  | 0.000 | 1.000 |
| skytnt_medium | 0.965 | 0.991 | 0.037 | 0.934 | 1.000 |  | 1.000 |
| real_shuffled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |

cont96 with_drums, first segment, contrast: share of draws where the row beats the column

| | amt | f1 | long_450m | prompt | skytnt_large | skytnt_medium | real_shuffled |
|---|---|---|---|---|---|---|---|
| amt |  | 0.404 | 0.000 | 0.075 | 0.851 | 0.020 | 1.000 |
| f1 | 0.596 |  | 0.000 | 0.111 | 0.946 | 0.002 | 1.000 |
| long_450m | 1.000 | 1.000 |  | 0.996 | 1.000 | 0.935 | 1.000 |
| prompt | 0.925 | 0.889 | 0.004 |  | 0.999 | 0.108 | 1.000 |
| skytnt_large | 0.149 | 0.054 | 0.000 | 0.001 |  | 0.000 | 1.000 |
| skytnt_medium | 0.980 | 0.998 | 0.065 | 0.892 | 1.000 |  | 1.000 |
| real_shuffled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |

## cont96 published, first segment

| system | own cosine | contrast | MRR |
|---|---|---|---|
| amt | 0.649 (0.618 to 0.681) | 0.259 (0.228 to 0.291) | 0.604 (0.478 to 0.733) |
| f1 | 0.573 (0.536 to 0.611) | 0.207 (0.172 to 0.245) | 0.543 (0.404 to 0.681) |
| long_450m | 0.712 (0.682 to 0.742) | 0.321 (0.286 to 0.357) | 0.863 (0.762 to 0.948) |
| prompt | 0.656 (0.621 to 0.692) | 0.280 (0.246 to 0.314) | 0.696 (0.581 to 0.804) |
| skytnt_large | 0.583 (0.541 to 0.623) | 0.207 (0.170 to 0.245) | 0.526 (0.390 to 0.662) |
| skytnt_medium | 0.633 (0.601 to 0.665) | 0.260 (0.234 to 0.286) | 0.679 (0.560 to 0.796) |
| real_shuffled | 0.376 (0.338 to 0.414) | -0.027 (-0.060 to 0.007) | 0.032 (0.022 to 0.043) |

cont96 published, first segment, own cosine: share of draws where the row beats the column

| | amt | f1 | long_450m | prompt | skytnt_large | skytnt_medium | real_shuffled |
|---|---|---|---|---|---|---|---|
| amt |  | 1.000 | 0.000 | 0.392 | 1.000 | 0.830 | 1.000 |
| f1 | 0.000 |  | 0.000 | 0.000 | 0.330 | 0.013 | 1.000 |
| long_450m | 1.000 | 1.000 |  | 1.000 | 1.000 | 1.000 | 1.000 |
| prompt | 0.608 | 1.000 | 0.001 |  | 1.000 | 0.870 | 1.000 |
| skytnt_large | 0.000 | 0.670 | 0.000 | 0.000 |  | 0.003 | 1.000 |
| skytnt_medium | 0.170 | 0.987 | 0.000 | 0.130 | 0.998 |  | 1.000 |
| real_shuffled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |

cont96 published, first segment, contrast: share of draws where the row beats the column

| | amt | f1 | long_450m | prompt | skytnt_large | skytnt_medium | real_shuffled |
|---|---|---|---|---|---|---|---|
| amt |  | 0.987 | 0.001 | 0.160 | 0.997 | 0.514 | 1.000 |
| f1 | 0.013 |  | 0.000 | 0.000 | 0.495 | 0.017 | 1.000 |
| long_450m | 0.999 | 1.000 |  | 0.996 | 1.000 | 0.998 | 1.000 |
| prompt | 0.840 | 1.000 | 0.004 |  | 1.000 | 0.849 | 1.000 |
| skytnt_large | 0.003 | 0.505 | 0.000 | 0.000 |  | 0.003 | 1.000 |
| skytnt_medium | 0.486 | 0.983 | 0.002 | 0.151 | 0.997 |  | 1.000 |
| real_shuffled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |

## cont96 new, first segment

| system | own cosine | contrast | MRR |
|---|---|---|---|
| amt | 0.647 (0.620 to 0.673) | 0.241 (0.215 to 0.266) | 0.657 (0.557 to 0.755) |
| f1 | 0.608 (0.577 to 0.640) | 0.216 (0.189 to 0.244) | 0.573 (0.473 to 0.675) |
| long_450m | 0.717 (0.689 to 0.744) | 0.314 (0.285 to 0.342) | 0.785 (0.699 to 0.865) |
| prompt | 0.640 (0.617 to 0.662) | 0.248 (0.223 to 0.272) | 0.655 (0.559 to 0.750) |
| skytnt_large | 0.583 (0.556 to 0.610) | 0.197 (0.169 to 0.224) | 0.543 (0.440 to 0.648) |
| skytnt_medium | 0.655 (0.625 to 0.684) | 0.261 (0.230 to 0.290) | 0.659 (0.556 to 0.758) |
| real_shuffled | 0.390 (0.363 to 0.418) | -0.008 (-0.033 to 0.018) | 0.049 (0.030 to 0.073) |

cont96 new, first segment, own cosine: share of draws where the row beats the column

| | amt | f1 | long_450m | prompt | skytnt_large | skytnt_medium | real_shuffled |
|---|---|---|---|---|---|---|---|
| amt |  | 0.976 | 0.000 | 0.676 | 1.000 | 0.348 | 1.000 |
| f1 | 0.024 |  | 0.000 | 0.041 | 0.947 | 0.001 | 1.000 |
| long_450m | 1.000 | 1.000 |  | 1.000 | 1.000 | 1.000 | 1.000 |
| prompt | 0.324 | 0.960 | 0.000 |  | 1.000 | 0.125 | 1.000 |
| skytnt_large | 0.000 | 0.053 | 0.000 | 0.000 |  | 0.000 | 1.000 |
| skytnt_medium | 0.652 | 0.999 | 0.000 | 0.875 | 1.000 |  | 1.000 |
| real_shuffled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |

cont96 new, first segment, contrast: share of draws where the row beats the column

| | amt | f1 | long_450m | prompt | skytnt_large | skytnt_medium | real_shuffled |
|---|---|---|---|---|---|---|---|
| amt |  | 0.922 | 0.000 | 0.359 | 0.992 | 0.143 | 1.000 |
| f1 | 0.078 |  | 0.000 | 0.031 | 0.919 | 0.000 | 1.000 |
| long_450m | 1.000 | 1.000 |  | 1.000 | 1.000 | 1.000 | 1.000 |
| prompt | 0.641 | 0.969 | 0.000 |  | 1.000 | 0.162 | 1.000 |
| skytnt_large | 0.008 | 0.081 | 0.000 | 0.000 |  | 0.000 | 1.000 |
| skytnt_medium | 0.857 | 1.000 | 0.000 | 0.838 | 1.000 |  | 1.000 |
| real_shuffled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |

## cont96, whole file (the authors' feature)

| system | own cosine | contrast | MRR |
|---|---|---|---|
| amt | 0.648 (0.627 to 0.668) | 0.247 (0.227 to 0.267) | 0.640 (0.560 to 0.718) |
| f1 | 0.597 (0.573 to 0.621) | 0.213 (0.192 to 0.235) | 0.563 (0.484 to 0.643) |
| long_450m | 0.715 (0.694 to 0.736) | 0.317 (0.295 to 0.339) | 0.811 (0.746 to 0.872) |
| prompt | 0.645 (0.626 to 0.664) | 0.258 (0.239 to 0.279) | 0.669 (0.595 to 0.740) |
| skytnt_large | 0.583 (0.560 to 0.605) | 0.200 (0.178 to 0.222) | 0.537 (0.455 to 0.621) |
| skytnt_medium | 0.648 (0.626 to 0.670) | 0.260 (0.239 to 0.281) | 0.666 (0.586 to 0.741) |
| real_shuffled | 0.385 (0.364 to 0.408) | -0.014 (-0.034 to 0.007) | 0.043 (0.030 to 0.060) |

## cont96 no_drums, whole file (the authors' feature)

| system | own cosine | contrast | MRR |
|---|---|---|---|
| amt | 0.655 (0.630 to 0.678) | 0.258 (0.235 to 0.281) | 0.643 (0.552 to 0.735) |
| f1 | 0.576 (0.549 to 0.604) | 0.205 (0.180 to 0.231) | 0.529 (0.430 to 0.628) |
| long_450m | 0.715 (0.692 to 0.739) | 0.321 (0.295 to 0.347) | 0.838 (0.760 to 0.908) |
| prompt | 0.640 (0.617 to 0.664) | 0.259 (0.233 to 0.283) | 0.645 (0.556 to 0.729) |
| skytnt_large | 0.572 (0.546 to 0.598) | 0.202 (0.175 to 0.228) | 0.549 (0.449 to 0.648) |
| skytnt_medium | 0.630 (0.604 to 0.655) | 0.250 (0.225 to 0.275) | 0.645 (0.548 to 0.738) |
| real_shuffled | 0.378 (0.352 to 0.406) | -0.024 (-0.048 to 0.002) | 0.039 (0.026 to 0.058) |

## cont96 with_drums, whole file (the authors' feature)

| system | own cosine | contrast | MRR |
|---|---|---|---|
| amt | 0.634 (0.594 to 0.671) | 0.224 (0.189 to 0.260) | 0.632 (0.482 to 0.780) |
| f1 | 0.640 (0.596 to 0.682) | 0.230 (0.191 to 0.270) | 0.635 (0.493 to 0.775) |
| long_450m | 0.715 (0.676 to 0.757) | 0.307 (0.268 to 0.348) | 0.755 (0.637 to 0.864) |
| prompt | 0.655 (0.623 to 0.687) | 0.258 (0.225 to 0.293) | 0.718 (0.586 to 0.843) |
| skytnt_large | 0.605 (0.562 to 0.647) | 0.196 (0.155 to 0.238) | 0.513 (0.365 to 0.664) |
| skytnt_medium | 0.686 (0.645 to 0.726) | 0.281 (0.241 to 0.321) | 0.711 (0.574 to 0.840) |
| real_shuffled | 0.400 (0.363 to 0.439) | 0.007 (-0.025 to 0.040) | 0.051 (0.027 to 0.088) |

## cont96 published, whole file (the authors' feature)

| system | own cosine | contrast | MRR |
|---|---|---|---|
| amt | 0.649 (0.618 to 0.681) | 0.259 (0.228 to 0.291) | 0.604 (0.478 to 0.733) |
| f1 | 0.573 (0.536 to 0.611) | 0.207 (0.172 to 0.245) | 0.543 (0.404 to 0.681) |
| long_450m | 0.712 (0.682 to 0.742) | 0.321 (0.286 to 0.357) | 0.863 (0.762 to 0.948) |
| prompt | 0.656 (0.621 to 0.692) | 0.280 (0.246 to 0.314) | 0.696 (0.581 to 0.804) |
| skytnt_large | 0.583 (0.541 to 0.623) | 0.207 (0.170 to 0.245) | 0.526 (0.390 to 0.662) |
| skytnt_medium | 0.633 (0.601 to 0.665) | 0.260 (0.234 to 0.286) | 0.679 (0.560 to 0.796) |
| real_shuffled | 0.376 (0.338 to 0.414) | -0.027 (-0.060 to 0.007) | 0.032 (0.022 to 0.043) |

## cont96 new, whole file (the authors' feature)

| system | own cosine | contrast | MRR |
|---|---|---|---|
| amt | 0.647 (0.620 to 0.673) | 0.241 (0.215 to 0.266) | 0.657 (0.557 to 0.755) |
| f1 | 0.608 (0.577 to 0.640) | 0.216 (0.189 to 0.244) | 0.573 (0.473 to 0.675) |
| long_450m | 0.717 (0.689 to 0.744) | 0.314 (0.285 to 0.342) | 0.785 (0.699 to 0.865) |
| prompt | 0.640 (0.617 to 0.662) | 0.248 (0.223 to 0.272) | 0.655 (0.559 to 0.750) |
| skytnt_large | 0.583 (0.556 to 0.610) | 0.197 (0.169 to 0.224) | 0.543 (0.440 to 0.648) |
| skytnt_medium | 0.655 (0.625 to 0.684) | 0.261 (0.230 to 0.290) | 0.659 (0.556 to 0.758) |
| real_shuffled | 0.390 (0.363 to 0.418) | -0.008 (-0.033 to 0.018) | 0.049 (0.030 to 0.073) |

## Frechet to the real continuations, first segment (lower is better; side number)

N = 96 files per row against the 96 real continuations, raw features: at this N the covariances are rank N - 1 of 768. 300 draws over files, each row's draws centred on its value.

| system | Frechet |
|---|---|
| amt | 130.260 (121.193 to 140.669) |
| f1 | 181.237 (168.545 to 194.084) |
| long_450m | 110.090 (100.119 to 121.045) |
| prompt | 188.269 (178.770 to 198.874) |
| skytnt_large | 176.977 (163.885 to 189.359) |
| skytnt_medium | 140.181 (128.531 to 152.392) |

Frechet, first segment: share of draws where the row beats the column

| | amt | f1 | long_450m | prompt | skytnt_large | skytnt_medium |
|---|---|---|---|---|---|---|
| amt |  | 1.000 | 0.000 | 1.000 | 1.000 | 0.993 |
| f1 | 0.000 |  | 0.000 | 0.853 | 0.290 | 0.000 |
| long_450m | 1.000 | 1.000 |  | 1.000 | 1.000 | 1.000 |
| prompt | 0.000 | 0.147 | 0.000 |  | 0.013 | 0.000 |
| skytnt_large | 0.000 | 0.710 | 0.000 | 0.987 |  | 0.000 |
| skytnt_medium | 0.007 | 1.000 | 0.000 | 1.000 | 1.000 |  |

## Frechet to the real continuations, whole file (the authors' feature) (lower is better; side number)

N = 96 files per row against the 96 real continuations, raw features: at this N the covariances are rank N - 1 of 768. 300 draws over files, each row's draws centred on its value.

| system | Frechet |
|---|---|
| amt | 130.260 (121.193 to 140.669) |
| f1 | 181.237 (168.545 to 194.084) |
| long_450m | 110.090 (100.119 to 121.045) |
| prompt | 188.269 (178.770 to 198.874) |
| skytnt_large | 176.977 (163.885 to 189.359) |
| skytnt_medium | 140.181 (128.531 to 152.392) |

Frechet, whole file (the authors' feature): share of draws where the row beats the column

| | amt | f1 | long_450m | prompt | skytnt_large | skytnt_medium |
|---|---|---|---|---|---|---|
| amt |  | 1.000 | 0.000 | 1.000 | 1.000 | 0.993 |
| f1 | 0.000 |  | 0.000 | 0.853 | 0.290 | 0.000 |
| long_450m | 1.000 | 1.000 |  | 1.000 | 1.000 | 1.000 |
| prompt | 0.000 | 0.147 | 0.000 |  | 0.013 | 0.000 |
| skytnt_large | 0.000 | 0.710 | 0.000 | 0.987 |  | 0.000 |
| skytnt_medium | 0.007 | 1.000 | 0.000 | 1.000 | 1.000 |  |
