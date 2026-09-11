<img src="https://raw.githubusercontent.com/drdave-teaching/OPIM5509-notebooks/main/_banners/opim5509_banner.svg" width="100%" alt="OPIM 5509 banner"/>

# Module 4 — energy track (Option B, side-by-side proposal)

**Status: proposal, not wired into the module.** This folder re-anchors the M4.2/M4.3 arc on one dataset — the hourly Connecticut demand + Bradley weather series students already met in Assignment 2 — instead of hopping across room occupancy, `cleanBDL` dew point/pressure and the temperature series. Same teaching beats, same cell order, a regression target you can read in megawatts. The stock notebook (the "don't trade on it" lesson) stays as is either way.

| Track video | Replaces | Notebook | The beats |
| :-- | :-- | :-- | :-- |
| 8B | 8 · occupancy window method | `E1_Multivariate_Demand_Lags` | lag table for target + covariates, two dense nets (demand lags vs + weather), the two honest plots |
| 9B–10B | 9–10 · occupancy RNN Pt 1/2 | `E2_Multivariate_Demand_RNN` | `split_sequences`, target last, SimpleRNN → LSTM swap → stacked, bake-off vs baselines, peak-hour sigmoid head |
| 14B | 14 · ConvLSTM on temperature/occupancy | `E3_Advanced_Demand_RNN` | Conv1D + MaxPooling1D → LSTM, recurrent dropout, stacking, Bidirectional, bake-off |
| 15B | 15 · many-to-many (a) + (b) | `E4_Many_To_Many_Demand` | two targets (demand + temperature), then past week → next 24 hours with MAE-by-horizon |

**If Option B is adopted:** the capstone `Forecasting_Electricity_Demand_RNN` (videos 16–17) becomes the M4.3 *wrap-up* rather than a new dataset — its baselines/EDA half can shrink because students have seen the data since video 8. **If Option A stays:** delete this folder; nothing else references it.

All four notebooks: seeded (`keras.utils.set_random_seed(5509)`), train 2018 / test 2019 chronological, scale on train only, end with save → `load_model` → identical-predictions check. Executed top-to-bottom on TF 2.21 / Keras 3 (stored outputs).
