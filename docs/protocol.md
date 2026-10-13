# Experimental protocol

Protocol identifier: `DTD-CODE2024-GAPSAFE-V1`. Units are nanoseconds; epoch labels use the product's GPS time system. This release preserves the first paper's historical 28-satellite population, including C41, and seeds 2024–2028.

| Item | 12 h | 24 h |
|---|---|---|
| Input differences | 5,760 (48 h) | 5,760 (48 h) |
| Forecast differences | 1,440 | 2,880 |
| Sampling interval | 30 s | 30 s |
| Window stride | 240 samples (2 h) | 240 samples (2 h) |
| Training target intervals | Jan 3–11, Jan 14–15, Jan 18–24, 2024 | Jan 3–11, Jan 18–24, 2024 |
| Evaluation target interval | Jan 27–31, 2024 | Jan 27–31, 2024 |
| Training windows per satellite | 162 | 144 |
| Evaluation windows per satellite | 42 | 36 |
| Basis initialization | Training-target PCA | Random |

Intervals are half-open: `[start, end)`. The Jan 27–31 block is used both for checkpoint selection and for the reported development comparisons. The release reproduces that protocol and labels the corresponding outputs as development evaluation. These records do not represent an independently held-out test. Evaluation windows overlap, and satellite-level paired comparisons use seed-averaged satellite values rather than treating windows as independent observations.

## Window alignment

Each allowed target interval is built separately. Its raw range starts exactly `ceil(input_length / 2880)` days before the target start and ends on the day before the exclusive target end. No difference is formed across a source-day gap. With 48 h input and this minimal raw range, the first eligible target begins at **00:00:30**, and stride counting starts at that first eligible window. Loading an entire month and starting at 00:00:00 would change the archived sample alignment.

For raw clock biases `b`, the input comprises differences `d[i:i+L]`, the target differences are `d[i+L:i+L+H]`, the reconstruction anchor is `b[i+L]`, and the clock target is `b[i+L+1:i+L+H+1]`. Differencing and anchors use float64; the network input and difference targets use float32. Inference uses one full-window batch per satellite to match the archived evaluator. Predicted differences are converted to float64 before cumulative reconstruction.

## Model and objective

Normalization uses the last 24 h input mean and population standard deviation, with the latter floored at `1e-5`; the divisor is this floored scale. Slow inputs are average-pooled by 24 and decomposed using centered moving averages. Every slow convolution branch consumes all three bands. Periodic features, gated slow/fast cross-attention summaries, and historical statistics are fused into a context vector.

The decoder predicts a mean, scale ratio, persistence gate and residual-basis coefficients. Both persistence and residual shapes are centered, combined, centered again, rescaled, and added to the predicted difference mean. Forecast clock biases are obtained by cumulative summation from the final observed clock bias. The residual spectrum concerns the learned centered residual mapping; it does not replace the full decoder, including persistence and scalar pathways.

Training loss is `mean(e²) + mean(cumsum(e)²) + 0.1 * mean(sum(e)²)`, where `e` is the difference prediction error. PaLR-DF uses Adam, learning rate 0.001, batch size 8, gradient norm clipping at 1, at most 100 epochs and patience 15. The initial state is eligible for selection (epoch 0). An improvement requires at least `1e-8` ns in mean per-window reconstructed RMSE. Refer to each config for the comparison adapters' distinct learning rates and epoch budgets.

The compact parameter count is `28,504 + (H + 65) r`, including the archived scalar `trend_logit` parameter retained for checkpoint compatibility; trend anchoring is disabled in the paper configs.

## Metrics and spectra

Compute each window's RMSE, then average windows for each satellite and seed. Average the 28 satellites to obtain one macro score per seed. Report the mean and **sample SD (`ddof=1`) of the five macro scores**. RE is the range of the clock error within a window; EAE is the absolute error at its last epoch. Do not pool all epochs before RMSE or substitute the SD across satellites.

Spectral context covariance is centered with divisor `N` and uses training windows only. For centered basis transpose `L` and coefficient matrix `W`, ordinary singular values describe `L W`; context weighting adds the square root of the context covariance. Task weighting applies the difference, integration and endpoint loss metric on the left. `spectra.py` uses QR factors and a small basis Gram matrix to avoid allocating an `H × H` metric. The 95%-energy rank uses squared singular values. The released table also includes other energy thresholds and loss geometries.

The full sweep uses ranks 2, 3, 4, 6, 8, 12 and seeds 2024–2026. Five-seed configurations are ranks 2, 3, 6, 12 at 12 h and ranks 2, 3, 12 at 24 h. They are aggregated separately; missing extra seeds for other ranks are never filled or implied.

## Comparison implementations

`baselines.py` contains the local direct-forecasting adapters actually used for these results. CNN-D, LSTM-D and LSTM-ATT-D operate on the last 240 input samples. Informer-D uses convolutional tokenization and distilled Transformer encoding. SSA-TCN-D uses centered smoothing bands with three TCN branches. MGAMNet follows a multigranularity patch/mixing/attention hierarchy. The last three are **local adaptations**, and their result labels should retain that qualification when comparisons are reused.

INT-Best denotes the frozen satellite-specific reference-selection policy from the manuscript. Its complete comparison records are included; the focused model zoo contains PaLR-DF and V2. The release does not claim to include every reference-policy branch or pretrained baseline checkpoint.
