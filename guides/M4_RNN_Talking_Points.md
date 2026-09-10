# M4 Talking Points — Recurrent Networks for Numeric Sequences

**OPIM 5509 - Introduction to Deep Learning · Dr. Dave Wanik · University of Connecticut**
*Fall 2026 · recording notes — read before you hit record*

Fifteen videos across nine notebooks (🔴 markers placed after the Keras-3 execution audit; full talking points live inside each marker's HTML comment — double-click the red dot while recording). This file is the running order and the per-video one-liner.

**Target: ≤8 minutes per video.** The 2022 M4 videos ran 4:30–10:06; the two long ones (LSTM-by-hand 10:06, many-to-many 8:05) are split below. Same pattern as M2/M3: motivate with the example, then work the concrete case, then the number.

**Proposed split (confirm against the HuskyCT M4 folders):** M4.1 = theory + by hand + univariate (videos 1–7); M4.2 = multivariate + stock + advanced (8–15). If HuskyCT has an M4.3 "advanced topics" folder, videos 12–15 slide into it.

## Running order

| # | Video | Notebook | Old source (2022) |
| :-- | :-- | :-- | :-- |
| **M4.1 — Theory, by hand, univariate** | | | |
| 1 | Sequences are different + the window method (lags) | `Univariate_Temperature_Lags` | `1_7xr2r7ll` 7:52 |
| 2 | The vanilla RNN, one time step at a time | `RNNs_By_Hand_basic` (SimpleRNN basic) | `1_ntq9yduo` 8:00 |
| 3 | Trainable parameters and output shape of a SimpleRNN | `RNNs_By_Hand_basic` (SimpleRNN advanced) | `1_b8zl03ki` 5:13 |
| 4 | LSTM by hand: four networks and a cell state | `RNNs_By_Hand_basic` (LSTM basic/advanced) | `1_wb9uz377` 10:06 → first half |
| 5 | GRU by hand, and stacking/mixing cells | `RNNs_By_Hand_basic` (GRU + Advanced) | `1_wb9uz377` second half |
| 6 | Univariate RNN Pt 1: the 3-D tensor and `split_sequence` | `Univariate_Temperature_RNN` | `1_4awr1wvf` 7:51 |
| 7 | Univariate RNN Pt 2: fit SimpleRNN, then LSTM, then beat the baselines | `Univariate_Temperature_RNN` | `1_qjmddsos` 7:00 + `1_f4bzdd6i` 5:22 (trim to one) |
| **M4.2 — Multivariate, stock, advanced** | | | |
| 8 | Multivariate window method: room occupancy | `Multivariate_Occupancy_Lags` | `1_0q71m2q6` 7:36 |
| 9 | Multivariate RNN Pt 1: `split_sequences`, column order, the classification head | `Multivariate_Occupancy_RNN` | `1_k20syja5` 7:00 |
| 10 | Multivariate RNN Pt 2: LSTM swap, stacking, persistence baseline | `Multivariate_Occupancy_RNN` | `1_jbfi30un` 5:12 |
| 11 | Predict the stock market (an honest result) | `Simple_Predict_The_Stock_Market_DL` | `1_r89vibek` 7:50 |
| 12 | Conv1D + MaxPooling1D on a sequence, by hand | `Advanced_RNN_Theory` | `1_w7pmbgt1` 5:24 |
| 13 | Recurrent dropout, stacking, bidirectional (the monsters) | `Advanced_RNN_Theory` | 2022 stacking/bidirectional videos |
| 14 | ConvLSTM on the temperature and occupancy series | `Univariate_Temperature_RNN_Advanced` + `Multivariate_Occupancy_RNN_AdvancedTopics` | `1_3bfa459i` 4:30 |
| 15 | Many-to-many: two targets at once, then multi-step ahead | `a_Many_To_Many_BDL_tmpf_and_vsby` + `b_..._tmpfPlus1` | `1_hru76zg7` 8:05 |

## Per-video one-liners (with the anchor numbers from the 2022 runs — re-verify on the Keras 3 run before you say them)

1. **Window method.** Shuffle Boston/California and nothing changes; shuffle a temperature series and you've destroyed it. The window method *deliberately* destroys the order — past 10 days become 10 columns — so any model works. 90/10 **chronological** split (no shuffle, or you leak the future). Dense net, ~361 params, MAE ≈ 1.86. Close with the trap: a 45° scatter *plus* a time-series plot, because a model can look great by just repeating yesterday.
2. **Vanilla RNN.** Three features meet two hidden units (the red dots) — a 5-input dense net with tanh — and the trick is the **handoff**: each time step's hidden state feeds the next, so the final state has seen the whole window. `SimpleRNN(units)` sets the red-dot count, and the **output shape is always `units`**. Data prep is the whole game: `(samples, look-back, features)`.
3. **SimpleRNN parameters.** Cell = (features + units) × units + bias. H=2, I=3 → 12, plus a 1-unit dense head (3) = **15**. Bigger: 30 features, 25 units → (30+25)×25+25 = 1,400, +26 = **1,426**. The general formula is **G · [H(H+I) + H]** and G is the number of little networks in the cell — SimpleRNN G=1.
4. **LSTM by hand.** G=4: four networks, plus a **cell state** (long memory) beside the hidden state (recent memory) — that's what fixes the vanishing gradient. 5 inputs × 2 units + 2 bias = 12 per network, ×4 = **48**; a bigger one gives 160. Same formula, G=4. Say why it's slow: every step spins all four networks.
5. **GRU + mixing.** G=3 (reset/update gates), counted the same way ×3 (`reset_after=False` to match the hand math). Then the by-hand appendix: stack cells, mix SimpleRNN → LSTM → GRU — one-word swaps in Keras, everything else identical.
6. **Univariate RNN Pt 1.** Drop the window feature engineering; hand the RNN the raw sequence as a **3-D tensor**. 3,650 daily temperatures, look-back 10 → 3,640 samples of 10×1. `split_sequence`: [1,2,3]→4, [2,3,4]→5. The **reshape** to add the trailing 1 is where 90% of RNN errors live. 90/10 chronological: 3,276 train / 364 test.
7. **Univariate RNN Pt 2.** Confirm `(3640, 10, 1)`; inherit `n_steps`/`n_features` from the shape; 30 red dots → 31×30+30 params, spinning 10 times. Linear output, MSE, MAE tracked, early stopping on val_loss. SimpleRNN MAE ≈ 1.77 vs window method 1.76 — *same*. LSTM ≈ 1.74. Then the sermon: beat **mean-only** and **persistence** (shift-1 ≈ 2.02) or you've learned nothing — and always metrics + scatter + time-series plot, so a shifted copy can't fool you.
8. **Multivariate window.** Still the window method, now with covariates: HVAC sensors (temp, humidity, light, CO₂) → occupied? Mostly unoccupied (imbalance); plot target with CO₂ and the story tells itself. Lag features from the past few minutes, **drop lagged occupancy to avoid leakage** (20 columns), 50/50 unshuffled, sigmoid → ~97% and a mostly-diagonal confusion matrix.
9. **Multivariate RNN Pt 1.** Multivariate differs only in prep: all X on the left, **target as the last column** (drop the date), look-back is the hyperparameter. `split_sequences` with look-back 10 → 2,655 samples of 10×features. SimpleRNN with a sigmoid head, `n_steps`/`n_features` inherited from the shape; it may learn in one epoch — add dropout. Show the time-series plot even for classification: it misses the quick in/out transitions.
10. **Multivariate RNN Pt 2.** One-word LSTM swap: (features+units)×units+bias, ×4 = **4,320**. It predicts the zeros a bit better; weighted F1 comparable. Stacked SimpleRNN with `return_sequences=True` keeps the 10×30 sequence into a second RNN — and does *worse* here (too complex for an easy problem). Persistence is brutal to beat; show value over the dummy. The `-1` predicts the next step — change it for multi-step.
11. **Stock market.** Several tickers 2017–2020, percent change, label Walmart's next-day up/down (shift by 1 — no same-day cheating). StandardScaler, 50/50, look-back 5, 11 features. Stacked LSTMs + dropout, patience 20. Result: barely beats 50%, weighted F1 < 0.5, lots of false positives — **"don't trade on it."** That's the honest lesson, and it's the best video in the module.
12. **Conv1D by hand.** A 21×9 sample, kernel 2, one filter → 20×1 (rows = 21−2+1, columns = filters). Params: a 1×2 kernel per column (9) + 1 bias = **19**. `MaxPooling1D(2)` is the domino: 20×1 → 10×1. Into a SimpleRNN(30): 10 spins → 1×30, 960 params, **1,010** total. More filters → richer sequences (20×3 → 10×3). It's `Conv1D`, not `Conv2D`, and pull `input_shape` from `X_train`.
13. **The monsters.** Recurrent dropout regularizes the recurrent connection, not just the inputs. Stack with `return_sequences=True` (more layers ≠ better — it's a hyperparameter). `Bidirectional(LSTM(3))` reads forward and backward with two independent cells and **concatenates** → width 6; patterns hard to see forward sometimes pop out backward. Build Monster #1 and #2 and read the parameter counts off `summary()`.
14. **ConvLSTM.** Data prep unchanged — add convolution + pooling in front of the LSTM. Univariate: look-back 30, kernel 3, 32 filters → 30×1 becomes 28×32. Multivariate: look-back 50, 5 features → 48×32, pooled to 24×32, then a SimpleRNN. The secret sauce is just shapes: mind `input_shape`, `return_sequences` when stacking, and know every output shape and param count.
15. **Many-to-many.** Two flavors. (a) Several targets at once — dew point and pressure from the other weather variables, both Y columns placed **last** (`split_sequences` chops from the right), `Dense(2, linear)` so one LSTM's weights are shaped by both targets ("borrow strength"). (b) Multi-step — forecast the next 3 hours (3 outputs); quality degrades further out. One model doing a whole forecasting job.

## The through-lines to keep hitting

- **It's all data prep.** ConvNets: get the folders right. RNNs: get the tensor right — `(samples, look-back, features)`. Say the deck-of-cards picture every time.
- **One formula for every cell.** G·[H(H+I)+H], G = 1/3/4. Videos 3, 4, 5, 10, 12 all count parameters — same muscle, and Assignment 5 (RNN Math) grades it.
- **Beat the dumb models.** Mean-only, persistence, linear on the lags. Persistence is the villain of the module; the stock video is the humility check.
- **Bridge back.** Chapter 2's dense head is still the head; Chapter 3's convolution shows up again as `Conv1D`; early stopping, curves, confusion matrix — unchanged methodology.

## Guardrails (slips to not make twice)

- The **`-1` in `split_sequences` predicts the next step**; multi-step needs the horizon changed (video 15).
- **`Conv1D`, never `Conv2D`** on sequences; derive `input_shape` from the data.
- **Chronological splits, no shuffle** — say why (future leakage) in videos 1, 6, 8.
- The LSTM hand math assumes `reset_after=False` for GRU parity; Keras' GRU default differs — say it, don't let the summary contradict you.
- Persistence looks great on a plot — always pair the plot with the metric (video 7).
- Numbers above are the **2022 runs**; unseeded Keras will drift a little — quote the Keras-3 rerun numbers in the markers.

## Recording checklist

- [ ] Restart-and-run-all each notebook in Colab before recording (nbclient-verified locally on TF 2.21 / Keras 3 — see the audit notes in `Module4/README.md`)
- [ ] 🔴 cells must render as a bare red dot — anything visible means the comment leaked
- [ ] ≤8 min; split rather than rush
- [ ] Tag uploads `(IDL)` — captions auto-generate, no ordering needed
- [ ] **Record Kaltura Capture at 1280×720** — kills the "CPU exceeded 99%" toast at the source (three M3.3 videos needed patching)
- [ ] Close the other Colab tabs — a background tab's "99%" progress lit up on camera in M3.3
