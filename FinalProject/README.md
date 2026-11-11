# Final Project: A Live Deep Learning Energy Forecast

An LSTM that forecasts Connecticut's electricity load for the next 24 hours, every hour, live on Google Cloud, and is
scored against the real load by the class leaderboard.

| Notebook | What you build | Ships with `range_method` |
| :-- | :-- | :-- |
| [Energy_Project_Setup_5509](https://github.com/drdave-teaching/OPIM5509-notebooks/blob/main/FinalProject/Energy_Project_Setup_5509.ipynb) | **Start here, once (about 45 minutes):** your Google Cloud project, your fork of energy-forecasting, your ISO Express account, and the hourly functions your LSTM runs in | |
| [FP1_Basic_LSTM_Forecast](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/FinalProject/FP1_Basic_LSTM_Forecast.ipynb) | A basic LSTM in the Module 4 style: one table per forecast (the past week of load, then the next 24 hours of weather and calendar with the load hidden), `Sequential` LSTM, `Dense(24)` for the whole next day. One forecast per hour. | `none` |
| [FP1b_Tune_Your_LSTM](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/FinalProject/FP1b_Tune_Your_LSTM.ipynb) | Tune the Part 1 network in rounds (Conv1D + pooling, stacked layers, units, GRU or LSTM, dropout) on your own tuning leaderboard, picked on 2025; actual-vs-predicted and holdout plots; ship the winner | `none` |
| [FP2_Probabilistic_LSTM](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/FinalProject/FP2_Probabilistic_LSTM.ipynb) | The same LSTM predicting 20 percentiles per hour (pinball loss), scored with CRPS, then three ways to build the 80% range on real HRRR weather forecasts | `lstm_hrrr_runs` (A), `lstm_quantiles` (B) or `lstm_quantiles_pooled` (C) |
| [FP2b_Tune_Your_Probabilistic_LSTM](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/FinalProject/FP2b_Tune_Your_Probabilistic_LSTM.ipynb) | Tune the percentile network in rounds and the range width (`range_widen`) on 2025, run the winner through the real 2026 test with HRRR forecasts, ship it if it beats Part 2 | `lstm_quantiles` (B) or `lstm_quantiles_pooled` (C), with `range_widen` |

**Deep learning only.** The forecast that goes live must come from a Keras network you trained (LSTM, GRU, Conv1D,
Dense, or a stack or ensemble of them inside one `model.keras`), run by `forecast-24h-dl`. Keras models only: no
scikit-learn, gradient boosting, random or quantile forests, XGBoost or hand-written rules in production (fine as a
baseline in your report). The [OPIM 5509 leaderboard](https://storage.googleapis.com/opim5509-leaderboard-f26/index.html)
only scores forecasts made by `forecast-24h-dl` with a Keras network (Part 9 of FP1 joins it, once).

Each notebook's input function (`make_window` in FP1, `make_inputs` in FP1b, FP2 and FP2b) is copied word for word into the cloud function `forecast-24h-dl` in
[drdave-teaching/energy-forecasting](https://github.com/drdave-teaching/energy-forecasting): if you change it in a
notebook, change it there too. Train and save your model **in Colab** (a model saved by a newer Keras on a laptop may
not load elsewhere).

What the notebooks scored on 2026 (CRPS skill against "same hour yesterday", higher is better):

| | Weather | CRPS skill | Hours inside the 80% range |
| :-- | :-- | --: | --: |
| FP1, one forecast per hour | what actually happened | 0.52 | (no range) |
| FP1b champion (GRU, Conv1D in front, picked on 2025) | what actually happened | 0.59 | (no range) |
| FP2, 20 percentiles | what actually happened | 0.70 | 52% |
| FP2 option A, 3 weather runs | HRRR as issued | 0.55 | 15% |
| FP2 option B, model percentiles | HRRR as issued | 0.64 | 47% |
| FP2 option C, both pooled | HRRR as issued | 0.65 | 51% |
| FP2b champion (GRU, Conv1D in front) + pooled + `range_widen` 1.2, picked on 2025 | HRRR as issued | 0.68 | 64% |

Every notebook trains on 2023-2024, validates (early stopping, tuning) on 2025 and tests once on 2026.

Every range is still too narrow (80% is the goal). Fixing that is a good place to start your improvement.
