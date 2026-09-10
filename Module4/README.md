<img src="https://raw.githubusercontent.com/drdave-teaching/OPIM5509-notebooks/main/_banners/opim5509_banner.svg" width="100%" alt="OPIM 5509 banner"/>

# Module 4 — Recurrent Networks for Numeric Sequences

**Fall 2026 · Dr. Dave Wanik · University of Connecticut**

Everything so far had no order — shuffle the rows and nothing changes. A time series is different: yesterday matters for today. Module 4 is about models that respect that. We start with **numbers, not text** on purpose (text would force embeddings *and* sequences at once); Module 5 generalizes to words.

```
  M4.1  Theory, by hand, univariate   window method → SimpleRNN → params (G·[H(H+I)+H]) → LSTM/GRU → temperature series
  M4.2  Multivariate, stock, advanced  occupancy → stock returns (honest) → Conv1D → dropout/stacking/bidirectional → ConvLSTM → many-to-many
```

## M4.1 — Theory, by hand, univariate

| # | Notebook | What you'll do | Open |
| :-- | :-- | :-- | :-- |
| 1 | **Univariate Temperature — Lags** | The window method: past `n_steps` days become columns, chronological split, a dense net baseline (and the persistence trap) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module4/Univariate_Temperature_Lags.ipynb) |
| 2 | **RNNs By Hand** | SimpleRNN → LSTM → GRU with a pencil: the hidden-state handoff, `output = units`, and one parameter formula for every cell | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module4/RNNs_By_Hand_basic.ipynb) |
| 3 | **Univariate Temperature — RNN** | `split_sequence` → the 3-D tensor `(samples, look-back, 1)` → SimpleRNN, then LSTM, then beat mean-only and persistence | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module4/Univariate_Temperature_RNN.ipynb) |

## M4.2 — Multivariate, stock, advanced

| # | Notebook | What you'll do | Open |
| :-- | :-- | :-- | :-- |
| 4 | **Multivariate Occupancy — Lags** | Window method with covariates (HVAC sensors → occupied?), lagged features without leaking the target | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module4/Multivariate_Occupancy_Lags.ipynb) |
| 5 | **Multivariate Occupancy — RNN** | `split_sequences` with the target last, a sigmoid head, the LSTM swap, stacking with `return_sequences`, the persistence baseline | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module4/Multivariate_Occupancy_RNN.ipynb) |
| 6 | **Predict the Stock Market (simple)** | Returns of 11 tickers → will Walmart rise tomorrow? Stacked LSTMs, and an honest near-50% answer: *don't trade on it* | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module4/Simple_Predict_The_Stock_Market_DL.ipynb) |
| 7 | **Advanced RNN Theory** | `Conv1D` + `MaxPooling1D` by hand, recurrent dropout, stacking, `Bidirectional` — and two "monster" architectures to read off `summary()` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module4/Advanced_RNN_Theory.ipynb) |
| 8 | **Univariate Temperature — Advanced** | ConvLSTM on the temperature series: conv + pooling in front of the recurrent layer | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module4/Univariate_Temperature_RNN_Advanced.ipynb) |
| — | *Multivariate Occupancy — Advanced Topics* | The same upgrades on the occupancy series — reference, no separate video | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module4/Multivariate_Occupancy_RNN_AdvancedTopics.ipynb) |
| 9 | **Many-to-Many (a): two targets** | Predict dew point *and* pressure at once — targets last, `Dense(2, linear)`, one model borrowing strength across both | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module4/a_Many_To_Many_BDL_tmpf_and_vsby.ipynb) |
| — | *Many-to-Many (b): multi-step* | Forecast the next 3 hours — and watch quality fade with the horizon | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module4/b_Many_To_Many_BDL_tmpf_and_tmpfPlus1.ipynb) |
| ✍ | **Assignment 5 (RNN Math)** | Parameter counts and output shapes for recurrent, Conv1D and bidirectional layers, by hand | *(HuskyCT)* |

**Data** loads from stable links in [`OPIM5509Files/OPIM5509_Module4_Files/data`](https://github.com/drdave-teaching/OPIM5509Files/tree/main/OPIM5509_Module4_Files/data): `daily-min-temperatures.csv` (Melbourne/Sydney daily minimums, 10 years), `datatest.txt` (room-occupancy sensors), `cleanBDL.csv` (Bradley airport weather), and `stock_adjclose_2017_2020.csv` — a snapshot of 11 tickers' adjusted closes replacing the live scrape the old notebook used.

## Guides

| Guide | Use it for |
| :-- | :-- |
| [🎙 Talking Points](../guides/M4_RNN_Talking_Points.md) | Instructor — the 15-video recording plan with running order and anchor numbers |
| [✅ Skills Sheet](../guides/M4_RNN_Skills.md) | The checklist of what you should own before Module 5 |

## Keras 3 / pandas 3 audit (Fall 2026)

All eleven notebooks were executed top-to-bottom on TensorFlow 2.21 / Keras 3 before this module was recorded. What changed:

- **Stock notebook rebuilt around a stable CSV.** The original pulled prices live through the `yahoo_fin` scraper, which no longer returns data (Yahoo changed). Prices now load from `stock_adjclose_2017_2020.csv` in the course repo — same 11 tickers and dates — with a commented `yfinance` cell for anyone who wants fresh data. Its import block also dropped the dead Keras-2 paths (`keras.preprocessing.text`, `keras.utils.np_utils`, `keras.layers.convolutional`) and the unused text-model imports.
- **`fillna(method='ffill')` → `.ffill()`** in the multi-step many-to-many notebook (pandas 3 removed the `method=` argument), and **`df.drop(axis=1, columns=[...])` → `df.drop(columns=[...])`** in both many-to-many notebooks (pandas 3 rejects passing `axis` together with `columns`).
- **Two content fixes:** the multivariate RNN notebook titled itself "Ozone" (it's the occupancy data), and the advanced-topics notebook said "conv2d" where the layer is — and must be — `Conv1D`.
- **Colab badges** re-pointed from the old OPIM5509Files path to this repo, and **15 🔴 recording markers** added (a bare red dot; talking points live in the cell's HTML comment).
- The red `input_shape` UserWarning Keras 3 prints on the first model cell is harmless — same as Modules 2–3.
