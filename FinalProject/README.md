# Final Project: A Live Deep Learning Energy Forecast

An LSTM that forecasts Connecticut's electricity load for the next 24 hours, every hour, live on Google Cloud, and is
scored against the real load by the class leaderboard.

| Notebook | What you build | Ships with `range_method` |
| :-- | :-- | :-- |
| [FP1_Basic_LSTM_Forecast](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/FinalProject/FP1_Basic_LSTM_Forecast.ipynb) | A two-input LSTM: the past week of load, then the next 24 hours of weather and calendar. One forecast per hour. | `none` |
| [FP2_Probabilistic_LSTM](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/FinalProject/FP2_Probabilistic_LSTM.ipynb) | The same LSTM predicting 20 percentiles per hour (pinball loss), scored with CRPS, then three ways to build the 80% range on real HRRR weather forecasts | `lstm_hrrr_runs` (A), `lstm_quantiles` (B) or `lstm_quantiles_pooled` (C) |

Both notebooks use the same `make_inputs` function as the cloud function `forecast-24h-dl` in
[drdave-teaching/energy-forecasting](https://github.com/drdave-teaching/energy-forecasting): if you change it in a
notebook, change it there too. Train and save your model **in Colab** (a model saved by a newer Keras on a laptop may
not load elsewhere).

What the notebooks scored on 2026 (CRPS skill against "same hour yesterday", higher is better):

| | Weather | CRPS skill | Hours inside the 80% range |
| :-- | :-- | --: | --: |
| FP1, one forecast per hour | what actually happened | 0.58 | (no range) |
| FP2, 20 percentiles | what actually happened | 0.74 | 59% |
| FP2 option A, 3 weather runs | HRRR as issued | 0.58 | 15% |
| FP2 option B, model percentiles | HRRR as issued | 0.66 | 48% |
| FP2 option C, both pooled | HRRR as issued | 0.67 | 52% |

Every range is still too narrow (80% is the goal). Fixing that is a good place to start your improvement.
