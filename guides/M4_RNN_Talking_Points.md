# M4 Talking Points — Recurrent Networks for Numeric Sequences

**OPIM 5509 - Introduction to Deep Learning · Dr. Dave Wanik · University of Connecticut**
*Fall 2026 · recording notes — read before you hit record*

Twenty-two videos across fifteen notebooks (🔴 markers placed after the Keras-3 execution audit; full talking points live inside each marker's HTML comment — double-click the red dot while recording). This file is the running order and the per-video one-liner.

**Target: ≤8 minutes per video.** The 2022 M4 videos ran 4:30–10:06; the two long ones (LSTM-by-hand 10:06, many-to-many 8:05) are split below. Same pattern as M2/M3: motivate with the example, then work the concrete case, then the number.

**Split (mirrors the 2022 module):** M4.1 = theory + by hand + univariate (videos 1–7); M4.2 = multivariate + stock (8–11); M4.3 = advanced + the demand capstone (12–19).

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
| 7b | Univariate RNN Pt 2b: reload the saved LSTM, explain it, roll it forward | `Univariate_Temperature_RNN_pt2` | NEW (Fall 2026) |
| **M4.2 — Multivariate, stock** | | | |
| 8 | Multivariate window method: room occupancy | `Multivariate_Occupancy_Lags` | `1_0q71m2q6` 7:36 |
| 9 | Multivariate RNN Pt 1: `split_sequences`, column order, the classification head | `Multivariate_Occupancy_RNN` | `1_k20syja5` 7:00 |
| 10 | Multivariate RNN Pt 2: LSTM swap, stacking, persistence baseline | `Multivariate_Occupancy_RNN` | `1_jbfi30un` 5:12 |
| 10b | Multivariate RNN Pt 2b: reload the classifier, score two unseen days, explain it | `Multivariate_Occupancy_RNN_pt2` | NEW (Fall 2026) |
| 11 | Predict the stock market (an honest result) | `Simple_Predict_The_Stock_Market_DL` | `1_r89vibek` 7:50 |
| **M4.3 — Advanced + the capstone** | | | |
| 12 | Conv1D + MaxPooling1D on a sequence, by hand | `Advanced_RNN_Theory` | `1_w7pmbgt1` 5:24 |
| 13 | Recurrent dropout, stacking, bidirectional (the monsters) | `Advanced_RNN_Theory` | 2022 stacking/bidirectional videos |
| 14 | ConvLSTM on the temperature and occupancy series | `Univariate_Temperature_RNN_Advanced` + `Multivariate_Occupancy_RNN_AdvancedTopics` | `1_3bfa459i` 4:30 |
| 15 | Many-to-many: two targets at once, then multi-step ahead | `a_Many_To_Many_BDL_tmpf_and_vsby` + `b_..._tmpfPlus1` | `1_hru76zg7` 8:05 |
| 16 | Forecasting electricity demand, Pt 1: data, baselines, a univariate LSTM | `Forecasting_Electricity_Demand_RNN` | NEW (Fall 2026) |
| 17 | Forecasting electricity demand, Pt 2: add the weather, 24 hours ahead, call the peak | `Forecasting_Electricity_Demand_RNN` | NEW (Fall 2026) |
| 17b | Demand Pt 2b: reload, forecast 2020, watch the distribution shift | `Forecasting_Electricity_Demand_RNN_pt2` | NEW (Fall 2026) |
| 18 | Tomorrow, three ways: recursive vs direct vs multi-output, and where future covariates come from | `Multi_Step_Forecasting_Strategies` | NEW (Fall 2026) |
| 19 | What is the LSTM looking at? xAI for sequences | `Explaining_an_LSTM` | NEW (Fall 2026) |

## Per-video one-liners (with the anchor numbers from the 2022 runs — re-verify on the Keras 3 run before you say them)

1. **Window method.** Shuffle Boston/California and nothing changes; shuffle a temperature series and you've destroyed it. The window method *deliberately* destroys the order — past 10 days become 10 columns — so any model works. 90/10 **chronological** split (no shuffle, or you leak the future). Dense net, ~361 params, MAE ≈ 1.86. Close with the trap: a 45° scatter *plus* a time-series plot, because a model can look great by just repeating yesterday.
2. **Vanilla RNN.** Three features meet two hidden units (the red dots) — a 5-input dense net with tanh — and the trick is the **handoff**: each time step's hidden state feeds the next, so the final state has seen the whole window. `SimpleRNN(units)` sets the red-dot count, and the **output shape is always `units`**. Data prep is the whole game: `(samples, look-back, features)`.
3. **SimpleRNN parameters.** Cell = (features + units) × units + bias. H=2, I=3 → 12, plus a 1-unit dense head (3) = **15**. Bigger: 30 features, 25 units → (30+25)×25+25 = 1,400, +26 = **1,426**. The general formula is **G · [H(H+I) + H]** and G is the number of little networks in the cell — SimpleRNN G=1.
4. **LSTM by hand.** G=4: four networks, plus a **cell state** (long memory) beside the hidden state (recent memory) — that's what fixes the vanishing gradient. 5 inputs × 2 units + 2 bias = 12 per network, ×4 = **48**; a bigger one gives 160. Same formula, G=4. Say why it's slow: every step spins all four networks.
5. **GRU + mixing.** G=3 (reset/update gates), counted the same way ×3 (`reset_after=False` to match the hand math). Then the by-hand appendix: stack cells, mix SimpleRNN → LSTM → GRU — one-word swaps in Keras, everything else identical.
6. **Univariate RNN Pt 1.** Drop the window feature engineering; hand the RNN the raw sequence as a **3-D tensor**. 3,650 daily temperatures, look-back 10 → 3,640 samples of 10×1. `split_sequence`: [1,2,3]→4, [2,3,4]→5. The **reshape** to add the trailing 1 is where 90% of RNN errors live. 90/10 chronological: 3,276 train / 364 test.
7. **Univariate RNN Pt 2.** Confirm `(3640, 10, 1)`; inherit `n_steps`/`n_features` from the shape; 30 red dots → 31×30+30 params, spinning 10 times. Linear output, MSE, MAE tracked, early stopping on val_loss. SimpleRNN MAE ≈ 1.77 vs window method 1.76 — *same*. LSTM ≈ 1.74. Then the sermon: beat **mean-only** and **persistence** (shift-1 ≈ 2.02) or you've learned nothing — and always metrics + scatter + time-series plot, so a shifted copy can't fool you.
8. **Multivariate window.** Still the window method, now with covariates: HVAC sensors (temp, humidity, light, CO₂) → occupied? Mostly unoccupied (imbalance); plot target with CO₂ and the story tells itself. Two weeks of 1-minute sensor data (17,895 rows — the real UCI training + test files, new for 2026). Lag features from the past few minutes, **drop lagged occupancy to avoid leakage** (20 columns), 50/50 unshuffled, sigmoid → ~94% and a mostly-diagonal confusion matrix.
9. **Multivariate RNN Pt 1.** Multivariate differs only in prep: all X on the left, **target as the last column** (drop the date), look-back is the hyperparameter. `split_sequences` with look-back 10 → 8,943 test samples of 10×features (two weeks of data now, so `batch_size=64`). SimpleRNN with a sigmoid head, `n_steps`/`n_features` inherited from the shape; it may learn in one epoch — add dropout. Show the time-series plot even for classification: it misses the quick in/out transitions.
10. **Multivariate RNN Pt 2.** One-word LSTM swap: (features+units)×units+bias, ×4 = **4,320**. SimpleRNN ~0.93 → LSTM ~0.95 weighted F1 — a little better. Stacked SimpleRNN with `return_sequences=True` keeps the 10×30 sequence into a second RNN — and lands at ~0.94 — no better (more capacity isn't automatically better on an easy problem). Persistence scores ~0.99 — brutal to beat at one minute ahead; show value over the dummy honestly. The `-1` predicts the next step — change it for multi-step.
11. **Stock market.** Several tickers 2017–2020, percent change, label Walmart's next-day up/down (shift by 1 — no same-day cheating). StandardScaler, 50/50, look-back 5, 11 features. Stacked LSTMs + dropout, patience 20. Result: barely beats 50%, weighted F1 < 0.5, lots of false positives — **"don't trade on it."** That's the honest lesson, and it's the best video in the module.
12. **Conv1D by hand.** A 21×9 sample, kernel 2, one filter → 20×1 (rows = 21−2+1, columns = filters). Params: a 1×2 kernel per column (9) + 1 bias = **19**. `MaxPooling1D(2)` is the domino: 20×1 → 10×1. Into a SimpleRNN(30): 10 spins → 1×30, 960 params, **1,010** total. More filters → richer sequences (20×3 → 10×3). It's `Conv1D`, not `Conv2D`, and pull `input_shape` from `X_train`.
13. **The monsters.** Recurrent dropout regularizes the recurrent connection, not just the inputs. Stack with `return_sequences=True` (more layers ≠ better — it's a hyperparameter). `Bidirectional(LSTM(3))` reads forward and backward with two independent cells and **concatenates** → width 6; patterns hard to see forward sometimes pop out backward. Build Monster #1 and #2 and read the parameter counts off `summary()`.
14. **ConvLSTM.** Data prep unchanged — add convolution + pooling in front of the LSTM. Univariate: look-back 30, kernel 3, 32 filters → 30×1 becomes 28×32. Multivariate: look-back 50, 5 features → 48×32, pooled to 24×32, then a SimpleRNN. The secret sauce is just shapes: mind `input_shape`, `return_sequences` when stacking, and know every output shape and param count.
15. **Many-to-many.** Two flavors. (a) Several targets at once — dew point and pressure from the other weather variables, both Y columns placed **last** (`split_sequences` chops from the right), `Dense(2, linear)` so one LSTM's weights are shaped by both targets ("borrow strength"). (b) Multi-step — forecast the next 3 hours (3 outputs); quality degrades further out. One model doing a whole forecasting job.

16. **Demand Pt 1.** You are the utility; forecast tomorrow's hourly load - the Assignment 2 data as a sequence. The file is **unsorted** (show it). EDA beats: a July week, hour-of-day and month profiles, the demand-vs-temperature U, the 2020 COVID dip. Train 2017-18, test 2019, no shuffle. **Baselines first:** mean-only 554 MW, same-hour-yesterday 227, last-hour persistence 129 - read them off the table on screen. Univariate LSTM, look-back 24, scaled on train only: **~44 MW**, three times better than persistence, because two years of history taught it the daily shape.
17. **Demand Pt 2.** Add temperature, dew point, humidity and the **clock as sin/cos**; Demand last. Demand's own past stays in the inputs (that's the fix students always miss). Weather + clock: ~44 -> **~34 MW** at one hour ahead - helps, but the last hour already carries most of it. Then the real job: past week -> next 24 hours, `Dense(24)`, and the **MAE-by-horizon** plot against seasonal-naive (~167 vs 227 averaged over the day). Peak-hour classifier: top-10% hours, majority baseline 91.5%, so read precision/recall (~0.91 / ~0.88). Close on the seed + save/reload.

18. **Three ways to tomorrow.** Start from the one-step LSTM. **Recursive:** predict hour 1, write it into the demand column, slide, predict hour 2 - draw the window sliding. The trap is in the *columns*: clock is known (free), weather is NOT (a forecast - we cheat with actuals and label it perfect foresight, then freeze the weather for the honest curve), and past demand becomes your own guesses - **exposure bias**, errors compound. **Direct:** one model per horizon, nothing fed back (we train 1/6/12/24). **Multi-output:** `Dense(24)`, the capstone. Numbers to say (MAE at h=1/6/12/24): recursive-frozen 33/168/265/**414**, recursive-perfect-weather 33/159/236/331, direct 42/142/178/207, multi-output 87/139/171/206, seasonal naive ~227 flat. Recursive wins hour 1 and loses to *naive* from hour 8; direct and multi-output tie by hour 24. Say what you'd ship (direct or multi-output with forecast weather). Mention seq2seq + teacher forcing as the Module 5 bridge.
19. **Explaining the LSTM.** Against 5512's SHAP/LIME bars: the input is 24 x 8, so the explanation is a **heatmap**. Permutation by feature (demand's own past ~690 MW >> hour sin/cos ~80 >> weather ~10 - exactly what persistence told you). Occlusion by hour (hour -1 costs ~800 MW when blanked, the effect is gone by hour -9, and hour -24 does *nothing* - the model reads the daily rhythm off the clock features, not the window). Gradient saliency via `tf.GradientTape` - the training derivative pointed at the inputs. Integrated gradients: attributions that **add up** (completeness check ~0). What-if +10 F by month: the U, recovered from the model. SHAP: `GradientExplainer` works on the LSTM and agrees with integrated gradients - show the two heatmaps side by side; `DeepExplainer` doesn't (Keras 3), and SHAP wants NumPy < 2.3 - one sentence, move on. Attention = free explanations, Module 5.


**Part 2b videos (reload → unseen data → explain → roll forward).** Same recipe each time, and say so - it's the professional loop. **7b:** download the .keras, rebuild the windows, MAE matches Part 1; occlusion and saliency by lag (yesterday dominates, fades over a week); roll forward 14 days: 1.78 C on day 1, settles at ~2.25 and stays there, still beating persistence (2.75) and climatology (3.2) - recursive behaves on a mean-reverting series with no covariates; the energy notebooks show where it doesn't. **10b:** the third UCI file is two days the model never saw; confusion matrix vs persistence; 97.4% on the unseen days, persistence still 0.99; permutation by sensor - it's a *light detector*, ask whether that's a feature or a shortcut; +200 lux moves P(occupied) by 0.15, +300 ppm CO₂ by 0.01. **17b:** two artifacts (day-ahead + one-hour) and the *scaler*; 2019 reproduces (~167 MW); 2020 month by month (year MAE 167 -> 181; April-May worse where 2019 got better, August 288 vs 214) - distribution shift you can point at; recursive vs multi-output on the unseen year, same shape as 2019; permutation says demand's past > clock > weather. Close each one with: interpretability is not a separate topic, it's the last cell of every model.

## The through-lines to keep hitting

- **It's all data prep.** ConvNets: get the folders right. RNNs: get the tensor right — `(samples, look-back, features)`. Say the deck-of-cards picture every time.
- **One formula for every cell.** G·[H(H+I)+H], G = 1/3/4. Videos 3, 4, 5, 10, 12 all count parameters — same muscle, and Assignment 5 (RNN Math) grades it.
- **Beat the dumb models.** Mean-only, persistence, seasonal-naive, linear on the lags. Persistence is the villain of the module at one step ahead; seasonal-naive is the fair villain a day ahead; the stock video is the humility check.
- **Bridge back.** Chapter 2's dense head is still the head; Chapter 3's convolution shows up again as `Conv1D`; early stopping, curves, confusion matrix — unchanged methodology.

## Guardrails (slips to not make twice)

- The **`-1` in `split_sequences` predicts the next step**; multi-step needs the horizon changed (video 15).
- **`Conv1D`, never `Conv2D`** on sequences; derive `input_shape` from the data.
- **Chronological splits, no shuffle** — say why (future leakage) in videos 1, 6, 8.
- The LSTM hand math assumes `reset_after=False` for GRU parity; Keras' GRU default differs — say it, don't let the summary contradict you.
- Persistence looks great on a plot — always pair the plot with the metric (video 7).
- Numbers above are the **2022 runs**. The notebooks are now **seeded (5509)** and ship with the stored Keras-3 outputs — **quote the numbers on screen**, not these. On a T4 a recurrent kernel can drift a hair from the stored CPU numbers; say "within a few hundredths" rather than a false-precision match.
- **Say the reproducibility beat out loud once per notebook:** seed at the top, and the closing "Save the model and use it again" cell — save to `.keras`, `load_model`, identical predictions. Students will reuse saved models in the final project; this is where they learn it.

## Recording checklist

- [ ] Restart-and-run-all each notebook in Colab before recording (nbclient-verified locally on TF 2.21 / Keras 3 — see the audit notes in `Module4/README.md`)
- [ ] 🔴 cells must render as a bare red dot — anything visible means the comment leaked
- [ ] ≤8 min; split rather than rush
- [ ] Tag uploads `(IDL)` — captions auto-generate, no ordering needed
- [ ] **Record Kaltura Capture at 1280×720** — kills the "CPU exceeded 99%" toast at the source (three M3.3 videos needed patching)
- [ ] Close the other Colab tabs — a background tab's "99%" progress lit up on camera in M3.3
