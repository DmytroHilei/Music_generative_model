# CLaMP 3 continuation scores, cont96, first 200 new notes (2026-11-04 17:50)

C2 checkpoint, every file cut after its prompt and normalised (h2h_clamp3.py). Target = the embedding of the prompt's real continuation: own = cosine to it, contrast = own minus the mean cosine to 95 other prompts' real continuations, MRR = its reciprocal rank among them. prompt = the prompt itself (128 notes), real_shuffled = another prompt's real continuation (floor). 10000 bootstrap draws over the 96 prompts, 95% percentile interval. first = the first segment of 512 patches; global = the authors' whole-file feature.

## Length

| system | files | patches | segments | files over one segment |
|---|---|---|---|---|
| amt | 96 | 88 | 1.00 | 0.000 |
| f1 | 96 | 88 | 1.00 | 0.000 |
| long_450m | 96 | 88 | 1.00 | 0.000 |
| prompt | 96 | 62 | 1.00 | 0.000 |
| skytnt_large | 96 | 89 | 1.00 | 0.000 |
| skytnt_medium | 96 | 89 | 1.00 | 0.000 |
| real | 96 | 88 | 1.00 | 0.000 |

## cont96, first segment

| system | own cosine | contrast | MRR |
|---|---|---|---|
| amt | 0.751 (0.732 to 0.769) | 0.261 (0.243 to 0.278) | 0.799 (0.728 to 0.866) |
| f1 | 0.748 (0.728 to 0.768) | 0.254 (0.236 to 0.273) | 0.747 (0.672 to 0.817) |
| long_450m | 0.799 (0.782 to 0.815) | 0.305 (0.287 to 0.322) | 0.899 (0.847 to 0.944) |
| prompt | 0.782 (0.767 to 0.796) | 0.292 (0.276 to 0.309) | 0.891 (0.839 to 0.938) |
| skytnt_large | 0.725 (0.705 to 0.744) | 0.235 (0.215 to 0.255) | 0.697 (0.619 to 0.773) |
| skytnt_medium | 0.763 (0.745 to 0.781) | 0.275 (0.258 to 0.291) | 0.831 (0.766 to 0.891) |
| real_shuffled | 0.489 (0.466 to 0.511) | -0.010 (-0.029 to 0.009) | 0.042 (0.031 to 0.056) |

cont96, first segment, own cosine: share of draws where the row beats the column

| | amt | f1 | long_450m | prompt | skytnt_large | skytnt_medium | real_shuffled |
|---|---|---|---|---|---|---|---|
| amt |  | 0.594 | 0.000 | 0.000 | 0.988 | 0.109 | 1.000 |
| f1 | 0.406 |  | 0.000 | 0.001 | 0.987 | 0.054 | 1.000 |
| long_450m | 1.000 | 1.000 |  | 0.983 | 1.000 | 1.000 | 1.000 |
| prompt | 1.000 | 0.999 | 0.017 |  | 1.000 | 0.982 | 1.000 |
| skytnt_large | 0.012 | 0.013 | 0.000 | 0.000 |  | 0.000 | 1.000 |
| skytnt_medium | 0.891 | 0.946 | 0.000 | 0.018 | 1.000 |  | 1.000 |
| real_shuffled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |

cont96, first segment, contrast: share of draws where the row beats the column

| | amt | f1 | long_450m | prompt | skytnt_large | skytnt_medium | real_shuffled |
|---|---|---|---|---|---|---|---|
| amt |  | 0.720 | 0.000 | 0.000 | 0.989 | 0.081 | 1.000 |
| f1 | 0.280 |  | 0.000 | 0.000 | 0.967 | 0.015 | 1.000 |
| long_450m | 1.000 | 1.000 |  | 0.922 | 1.000 | 1.000 | 1.000 |
| prompt | 1.000 | 1.000 | 0.078 |  | 1.000 | 0.986 | 1.000 |
| skytnt_large | 0.011 | 0.033 | 0.000 | 0.000 |  | 0.000 | 1.000 |
| skytnt_medium | 0.919 | 0.985 | 0.000 | 0.014 | 1.000 |  | 1.000 |
| real_shuffled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |

## cont96 no_drums, first segment

| system | own cosine | contrast | MRR |
|---|---|---|---|
| amt | 0.758 (0.738 to 0.778) | 0.273 (0.254 to 0.291) | 0.854 (0.776 to 0.922) |
| f1 | 0.742 (0.717 to 0.767) | 0.260 (0.237 to 0.282) | 0.744 (0.653 to 0.831) |
| long_450m | 0.801 (0.781 to 0.820) | 0.317 (0.295 to 0.338) | 0.894 (0.832 to 0.949) |
| prompt | 0.783 (0.764 to 0.800) | 0.301 (0.281 to 0.320) | 0.897 (0.836 to 0.949) |
| skytnt_large | 0.719 (0.693 to 0.745) | 0.245 (0.220 to 0.270) | 0.739 (0.644 to 0.828) |
| skytnt_medium | 0.757 (0.735 to 0.777) | 0.278 (0.258 to 0.297) | 0.842 (0.764 to 0.909) |
| real_shuffled | 0.479 (0.452 to 0.506) | -0.021 (-0.044 to 0.003) | 0.039 (0.026 to 0.054) |

cont96 no_drums, first segment, own cosine: share of draws where the row beats the column

| | amt | f1 | long_450m | prompt | skytnt_large | skytnt_medium | real_shuffled |
|---|---|---|---|---|---|---|---|
| amt |  | 0.878 | 0.000 | 0.011 | 0.998 | 0.556 | 1.000 |
| f1 | 0.122 |  | 0.000 | 0.001 | 0.955 | 0.117 | 1.000 |
| long_450m | 1.000 | 1.000 |  | 0.973 | 1.000 | 1.000 | 1.000 |
| prompt | 0.989 | 0.999 | 0.027 |  | 1.000 | 0.996 | 1.000 |
| skytnt_large | 0.002 | 0.045 | 0.000 | 0.000 |  | 0.000 | 1.000 |
| skytnt_medium | 0.444 | 0.883 | 0.000 | 0.004 | 1.000 |  | 1.000 |
| real_shuffled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |

cont96 no_drums, first segment, contrast: share of draws where the row beats the column

| | amt | f1 | long_450m | prompt | skytnt_large | skytnt_medium | real_shuffled |
|---|---|---|---|---|---|---|---|
| amt |  | 0.829 | 0.000 | 0.004 | 0.980 | 0.319 | 1.000 |
| f1 | 0.171 |  | 0.000 | 0.000 | 0.853 | 0.066 | 1.000 |
| long_450m | 1.000 | 1.000 |  | 0.944 | 1.000 | 1.000 | 1.000 |
| prompt | 0.996 | 1.000 | 0.056 |  | 1.000 | 0.990 | 1.000 |
| skytnt_large | 0.020 | 0.147 | 0.000 | 0.000 |  | 0.001 | 1.000 |
| skytnt_medium | 0.681 | 0.934 | 0.000 | 0.010 | 0.999 |  | 1.000 |
| real_shuffled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |

## cont96 with_drums, first segment

| system | own cosine | contrast | MRR |
|---|---|---|---|
| amt | 0.736 (0.696 to 0.774) | 0.236 (0.200 to 0.271) | 0.684 (0.536 to 0.822) |
| f1 | 0.761 (0.727 to 0.794) | 0.243 (0.211 to 0.275) | 0.752 (0.623 to 0.870) |
| long_450m | 0.793 (0.763 to 0.823) | 0.279 (0.249 to 0.306) | 0.909 (0.810 to 0.984) |
| prompt | 0.780 (0.752 to 0.806) | 0.275 (0.244 to 0.305) | 0.879 (0.769 to 0.971) |
| skytnt_large | 0.735 (0.706 to 0.766) | 0.213 (0.180 to 0.246) | 0.609 (0.458 to 0.752) |
| skytnt_medium | 0.778 (0.745 to 0.809) | 0.268 (0.237 to 0.299) | 0.807 (0.682 to 0.919) |
| real_shuffled | 0.509 (0.471 to 0.546) | 0.013 (-0.018 to 0.043) | 0.050 (0.031 to 0.076) |

cont96 with_drums, first segment, own cosine: share of draws where the row beats the column

| | amt | f1 | long_450m | prompt | skytnt_large | skytnt_medium | real_shuffled |
|---|---|---|---|---|---|---|---|
| amt |  | 0.104 | 0.000 | 0.011 | 0.505 | 0.035 | 1.000 |
| f1 | 0.896 |  | 0.009 | 0.127 | 0.918 | 0.150 | 1.000 |
| long_450m | 1.000 | 0.991 |  | 0.823 | 1.000 | 0.865 | 1.000 |
| prompt | 0.990 | 0.874 | 0.177 |  | 0.995 | 0.569 | 1.000 |
| skytnt_large | 0.495 | 0.082 | 0.000 | 0.005 |  | 0.000 | 1.000 |
| skytnt_medium | 0.965 | 0.850 | 0.135 | 0.431 | 1.000 |  | 1.000 |
| real_shuffled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |

cont96 with_drums, first segment, contrast: share of draws where the row beats the column

| | amt | f1 | long_450m | prompt | skytnt_large | skytnt_medium | real_shuffled |
|---|---|---|---|---|---|---|---|
| amt |  | 0.357 | 0.006 | 0.025 | 0.848 | 0.074 | 1.000 |
| f1 | 0.643 |  | 0.008 | 0.025 | 0.960 | 0.048 | 1.000 |
| long_450m | 0.994 | 0.992 |  | 0.597 | 1.000 | 0.780 | 1.000 |
| prompt | 0.975 | 0.975 | 0.403 |  | 1.000 | 0.684 | 1.000 |
| skytnt_large | 0.152 | 0.040 | 0.000 | 0.000 |  | 0.000 | 1.000 |
| skytnt_medium | 0.926 | 0.952 | 0.220 | 0.316 | 1.000 |  | 1.000 |
| real_shuffled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |

## cont96 published, first segment

| system | own cosine | contrast | MRR |
|---|---|---|---|
| amt | 0.746 (0.720 to 0.774) | 0.270 (0.241 to 0.299) | 0.830 (0.712 to 0.935) |
| f1 | 0.726 (0.689 to 0.762) | 0.252 (0.220 to 0.287) | 0.648 (0.514 to 0.783) |
| long_450m | 0.787 (0.757 to 0.816) | 0.309 (0.275 to 0.341) | 0.852 (0.746 to 0.943) |
| prompt | 0.786 (0.760 to 0.811) | 0.311 (0.283 to 0.339) | 0.927 (0.854 to 0.984) |
| skytnt_large | 0.712 (0.671 to 0.751) | 0.243 (0.208 to 0.278) | 0.703 (0.563 to 0.835) |
| skytnt_medium | 0.740 (0.705 to 0.773) | 0.274 (0.246 to 0.301) | 0.834 (0.727 to 0.930) |
| real_shuffled | 0.480 (0.439 to 0.521) | -0.022 (-0.059 to 0.016) | 0.039 (0.023 to 0.062) |

cont96 published, first segment, own cosine: share of draws where the row beats the column

| | amt | f1 | long_450m | prompt | skytnt_large | skytnt_medium | real_shuffled |
|---|---|---|---|---|---|---|---|
| amt |  | 0.854 | 0.002 | 0.001 | 0.976 | 0.678 | 1.000 |
| f1 | 0.146 |  | 0.000 | 0.000 | 0.795 | 0.229 | 1.000 |
| long_450m | 0.998 | 1.000 |  | 0.522 | 1.000 | 1.000 | 1.000 |
| prompt | 0.999 | 1.000 | 0.478 |  | 1.000 | 1.000 | 1.000 |
| skytnt_large | 0.025 | 0.205 | 0.000 | 0.000 |  | 0.012 | 1.000 |
| skytnt_medium | 0.322 | 0.771 | 0.000 | 0.000 | 0.988 |  | 1.000 |
| real_shuffled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |

cont96 published, first segment, contrast: share of draws where the row beats the column

| | amt | f1 | long_450m | prompt | skytnt_large | skytnt_medium | real_shuffled |
|---|---|---|---|---|---|---|---|
| amt |  | 0.814 | 0.004 | 0.001 | 0.956 | 0.396 | 1.000 |
| f1 | 0.186 |  | 0.000 | 0.000 | 0.711 | 0.131 | 1.000 |
| long_450m | 0.996 | 1.000 |  | 0.450 | 1.000 | 0.993 | 1.000 |
| prompt | 0.999 | 1.000 | 0.550 |  | 1.000 | 0.995 | 1.000 |
| skytnt_large | 0.044 | 0.289 | 0.000 | 0.000 |  | 0.009 | 1.000 |
| skytnt_medium | 0.604 | 0.869 | 0.007 | 0.005 | 0.991 |  | 1.000 |
| real_shuffled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |

## cont96 new, first segment

| system | own cosine | contrast | MRR |
|---|---|---|---|
| amt | 0.753 (0.728 to 0.777) | 0.256 (0.234 to 0.278) | 0.784 (0.693 to 0.871) |
| f1 | 0.759 (0.735 to 0.783) | 0.255 (0.233 to 0.278) | 0.796 (0.710 to 0.875) |
| long_450m | 0.805 (0.785 to 0.824) | 0.302 (0.282 to 0.323) | 0.922 (0.866 to 0.970) |
| prompt | 0.780 (0.761 to 0.798) | 0.283 (0.262 to 0.303) | 0.873 (0.804 to 0.935) |
| skytnt_large | 0.731 (0.709 to 0.753) | 0.230 (0.205 to 0.256) | 0.694 (0.595 to 0.790) |
| skytnt_medium | 0.775 (0.754 to 0.795) | 0.275 (0.254 to 0.296) | 0.829 (0.747 to 0.904) |
| real_shuffled | 0.494 (0.467 to 0.520) | -0.004 (-0.026 to 0.018) | 0.044 (0.031 to 0.060) |

cont96 new, first segment, own cosine: share of draws where the row beats the column

| | amt | f1 | long_450m | prompt | skytnt_large | skytnt_medium | real_shuffled |
|---|---|---|---|---|---|---|---|
| amt |  | 0.323 | 0.000 | 0.020 | 0.929 | 0.051 | 1.000 |
| f1 | 0.677 |  | 0.000 | 0.055 | 0.983 | 0.083 | 1.000 |
| long_450m | 1.000 | 1.000 |  | 0.991 | 1.000 | 0.998 | 1.000 |
| prompt | 0.980 | 0.945 | 0.009 |  | 1.000 | 0.687 | 1.000 |
| skytnt_large | 0.071 | 0.017 | 0.000 | 0.000 |  | 0.000 | 1.000 |
| skytnt_medium | 0.949 | 0.917 | 0.002 | 0.313 | 1.000 |  | 1.000 |
| real_shuffled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |

cont96 new, first segment, contrast: share of draws where the row beats the column

| | amt | f1 | long_450m | prompt | skytnt_large | skytnt_medium | real_shuffled |
|---|---|---|---|---|---|---|---|
| amt |  | 0.529 | 0.000 | 0.017 | 0.952 | 0.081 | 1.000 |
| f1 | 0.471 |  | 0.000 | 0.013 | 0.967 | 0.036 | 1.000 |
| long_450m | 1.000 | 1.000 |  | 0.956 | 1.000 | 0.996 | 1.000 |
| prompt | 0.983 | 0.987 | 0.044 |  | 1.000 | 0.798 | 1.000 |
| skytnt_large | 0.048 | 0.033 | 0.000 | 0.000 |  | 0.000 | 1.000 |
| skytnt_medium | 0.919 | 0.964 | 0.004 | 0.202 | 1.000 |  | 1.000 |
| real_shuffled | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |  |

## cont96, whole file (the authors' feature)

| system | own cosine | contrast | MRR |
|---|---|---|---|
| amt | 0.751 (0.732 to 0.769) | 0.261 (0.243 to 0.278) | 0.799 (0.728 to 0.866) |
| f1 | 0.748 (0.728 to 0.768) | 0.254 (0.236 to 0.273) | 0.747 (0.672 to 0.817) |
| long_450m | 0.799 (0.782 to 0.815) | 0.305 (0.287 to 0.322) | 0.899 (0.847 to 0.944) |
| prompt | 0.782 (0.767 to 0.796) | 0.292 (0.276 to 0.309) | 0.891 (0.839 to 0.938) |
| skytnt_large | 0.725 (0.705 to 0.744) | 0.235 (0.215 to 0.255) | 0.697 (0.619 to 0.773) |
| skytnt_medium | 0.763 (0.745 to 0.781) | 0.275 (0.258 to 0.291) | 0.831 (0.766 to 0.891) |
| real_shuffled | 0.489 (0.466 to 0.511) | -0.010 (-0.029 to 0.009) | 0.042 (0.031 to 0.056) |

## cont96 no_drums, whole file (the authors' feature)

| system | own cosine | contrast | MRR |
|---|---|---|---|
| amt | 0.758 (0.738 to 0.778) | 0.273 (0.254 to 0.291) | 0.854 (0.776 to 0.922) |
| f1 | 0.742 (0.717 to 0.767) | 0.260 (0.237 to 0.282) | 0.744 (0.653 to 0.831) |
| long_450m | 0.801 (0.781 to 0.820) | 0.317 (0.295 to 0.338) | 0.894 (0.832 to 0.949) |
| prompt | 0.783 (0.764 to 0.800) | 0.301 (0.281 to 0.320) | 0.897 (0.836 to 0.949) |
| skytnt_large | 0.719 (0.693 to 0.745) | 0.245 (0.220 to 0.270) | 0.739 (0.644 to 0.828) |
| skytnt_medium | 0.757 (0.735 to 0.777) | 0.278 (0.258 to 0.297) | 0.842 (0.764 to 0.909) |
| real_shuffled | 0.479 (0.452 to 0.506) | -0.021 (-0.044 to 0.003) | 0.039 (0.026 to 0.054) |

## cont96 with_drums, whole file (the authors' feature)

| system | own cosine | contrast | MRR |
|---|---|---|---|
| amt | 0.736 (0.696 to 0.774) | 0.236 (0.200 to 0.271) | 0.684 (0.536 to 0.822) |
| f1 | 0.761 (0.727 to 0.794) | 0.243 (0.211 to 0.275) | 0.752 (0.623 to 0.870) |
| long_450m | 0.793 (0.763 to 0.823) | 0.279 (0.249 to 0.306) | 0.909 (0.810 to 0.984) |
| prompt | 0.780 (0.752 to 0.806) | 0.275 (0.244 to 0.305) | 0.879 (0.769 to 0.971) |
| skytnt_large | 0.735 (0.706 to 0.766) | 0.213 (0.180 to 0.246) | 0.609 (0.458 to 0.752) |
| skytnt_medium | 0.778 (0.745 to 0.809) | 0.268 (0.237 to 0.299) | 0.807 (0.682 to 0.919) |
| real_shuffled | 0.509 (0.471 to 0.546) | 0.013 (-0.018 to 0.043) | 0.050 (0.031 to 0.076) |

## cont96 published, whole file (the authors' feature)

| system | own cosine | contrast | MRR |
|---|---|---|---|
| amt | 0.746 (0.720 to 0.774) | 0.270 (0.241 to 0.299) | 0.830 (0.712 to 0.935) |
| f1 | 0.726 (0.689 to 0.762) | 0.252 (0.220 to 0.287) | 0.648 (0.514 to 0.783) |
| long_450m | 0.787 (0.757 to 0.816) | 0.309 (0.275 to 0.341) | 0.852 (0.746 to 0.943) |
| prompt | 0.786 (0.760 to 0.811) | 0.311 (0.283 to 0.339) | 0.927 (0.854 to 0.984) |
| skytnt_large | 0.712 (0.671 to 0.751) | 0.243 (0.208 to 0.278) | 0.703 (0.563 to 0.835) |
| skytnt_medium | 0.740 (0.705 to 0.773) | 0.274 (0.246 to 0.301) | 0.834 (0.727 to 0.930) |
| real_shuffled | 0.480 (0.439 to 0.521) | -0.022 (-0.059 to 0.016) | 0.039 (0.023 to 0.062) |

## cont96 new, whole file (the authors' feature)

| system | own cosine | contrast | MRR |
|---|---|---|---|
| amt | 0.753 (0.728 to 0.777) | 0.256 (0.234 to 0.278) | 0.784 (0.693 to 0.871) |
| f1 | 0.759 (0.735 to 0.783) | 0.255 (0.233 to 0.278) | 0.796 (0.710 to 0.875) |
| long_450m | 0.805 (0.785 to 0.824) | 0.302 (0.282 to 0.323) | 0.922 (0.866 to 0.970) |
| prompt | 0.780 (0.761 to 0.798) | 0.283 (0.262 to 0.303) | 0.873 (0.804 to 0.935) |
| skytnt_large | 0.731 (0.709 to 0.753) | 0.230 (0.205 to 0.256) | 0.694 (0.595 to 0.790) |
| skytnt_medium | 0.775 (0.754 to 0.795) | 0.275 (0.254 to 0.296) | 0.829 (0.747 to 0.904) |
| real_shuffled | 0.494 (0.467 to 0.520) | -0.004 (-0.026 to 0.018) | 0.044 (0.031 to 0.060) |

## Frechet to the real continuations, first segment (lower is better; side number)

N = 96 files per row against the 96 real continuations, raw features: at this N the covariances are rank N - 1 of 768. 300 draws over files, each row's draws centred on its value.

| system | Frechet |
|---|---|
| amt | 96.856 (88.593 to 106.221) |
| f1 | 98.875 (89.450 to 109.195) |
| long_450m | 80.928 (71.327 to 90.599) |
| prompt | 93.791 (85.101 to 105.107) |
| skytnt_large | 110.835 (102.473 to 121.801) |
| skytnt_medium | 93.989 (85.367 to 104.147) |

Frechet, first segment: share of draws where the row beats the column

| | amt | f1 | long_450m | prompt | skytnt_large | skytnt_medium |
|---|---|---|---|---|---|---|
| amt |  | 0.693 | 0.000 | 0.173 | 1.000 | 0.213 |
| f1 | 0.307 |  | 0.000 | 0.143 | 0.997 | 0.117 |
| long_450m | 1.000 | 1.000 |  | 1.000 | 1.000 | 1.000 |
| prompt | 0.827 | 0.857 | 0.000 |  | 1.000 | 0.497 |
| skytnt_large | 0.000 | 0.003 | 0.000 | 0.000 |  | 0.000 |
| skytnt_medium | 0.787 | 0.883 | 0.000 | 0.503 | 1.000 |  |

## Frechet to the real continuations, whole file (the authors' feature) (lower is better; side number)

N = 96 files per row against the 96 real continuations, raw features: at this N the covariances are rank N - 1 of 768. 300 draws over files, each row's draws centred on its value.

| system | Frechet |
|---|---|
| amt | 96.856 (88.593 to 106.221) |
| f1 | 98.875 (89.450 to 109.195) |
| long_450m | 80.928 (71.327 to 90.599) |
| prompt | 93.791 (85.101 to 105.107) |
| skytnt_large | 110.835 (102.473 to 121.801) |
| skytnt_medium | 93.989 (85.367 to 104.147) |

Frechet, whole file (the authors' feature): share of draws where the row beats the column

| | amt | f1 | long_450m | prompt | skytnt_large | skytnt_medium |
|---|---|---|---|---|---|---|
| amt |  | 0.693 | 0.000 | 0.173 | 1.000 | 0.213 |
| f1 | 0.307 |  | 0.000 | 0.143 | 0.997 | 0.117 |
| long_450m | 1.000 | 1.000 |  | 1.000 | 1.000 | 1.000 |
| prompt | 0.827 | 0.857 | 0.000 |  | 1.000 | 0.497 |
| skytnt_large | 0.000 | 0.003 | 0.000 | 0.000 |  | 0.000 |
| skytnt_medium | 0.787 | 0.883 | 0.000 | 0.503 | 1.000 |  |
