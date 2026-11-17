# Scaling ladder fit

12 arms (GigaMIDI replay, under one pass each), held-out GigaMIDI CE in nats per note (Δt + pitch + duration + velocity).

| Parameters | 40M notes | 80M notes | 160M notes |
|---|---|---|---|
| 6.4M | 4.109 | 3.270 | 2.866 |
| 20.5M | 3.947 | 2.935 | 2.542 |
| 43.1M | 4.088 | 2.842 | 2.426 |
| 108.0M | 4.563 | 2.821 | 2.353 |

Fit: L = 2.232 + 4.05·(N/1M)^−1.500 + 471.96·(D/1M)^−1.500, RMSE 0.177 on the arms.

Check 1, the largest budget predicted from the smaller ones: error -0.486, -0.278, -0.178, -0.111 (predicted − measured, by rising size).

| Point | Predicted | Leave-one-out range | Measured |
|---|---|---|---|
| musicar-L (464M), D = all 8.3B notes | 2.233 | 2.074–2.269 | 1.641 |
| musicar-L, D = its GigaMIDI share 3.3B | 2.235 | 2.078–2.271 | 1.641 |
| XL: 1B, 20B notes | 2.233 | 2.062–2.268 |  |
| XL: 1B, 20B notes, GigaMIDI share | 2.233 | 2.064–2.269 |  |
| XXL: 1.8B, 35B notes | 2.232 | 2.057–2.268 |  |
| XXL: 1.8B, 35B notes, GigaMIDI share | 2.233 | 2.058–2.268 |  |
