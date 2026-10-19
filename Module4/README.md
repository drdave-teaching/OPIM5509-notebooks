<img src="https://raw.githubusercontent.com/drdave-teaching/OPIM5509-notebooks/main/_banners/opim5509_banner.svg" width="100%" alt="OPIM 5509 banner"/>

# Module 4 — Recurrent Networks for Numeric Sequences

**Fall 2026 · Dr. Dave Wanik · University of Connecticut**

Everything so far had no order — shuffle the rows and nothing changes. A time series is different: yesterday matters for today. Module 4 is about models that respect that. We start with **numbers, not text** on purpose (text would force embeddings *and* sequences at once); Module 5 generalizes to words.

```
  M4.1  Theory (videos 1-9)            window method → samples and the hidden state → trainable parameters by hand (SimpleRNN, LSTM, GRU)
  M4.2  Implementation (videos 10-18)  univariate temperature RNN → reload + explain → multivariate occupancy → stacking → xAI
  M4.3  Advanced topics (videos 19-26) primer (conv/pool/dropout/bidirectional) → univariate → multivariate energy → many-to-many → multi-step
```

The sections match HuskyCT: each has its own *Videos* folder and *Notebooks and Data* page.

## M4.1 — Theory

| Videos | Notebook | What you'll do | Open |
| :-- | :-- | :-- | :-- |
| 1-3 | **Univariate Temperature — Lags** | The window method: past `n_steps` days become columns, chronological split, a dense net baseline (and the persistence trap) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module4/Univariate_Temperature_Lags.ipynb) |
| 4-5 | **RNN Samples and the Hidden State** | How `split_sequence` / `split_sequences` build the 3-D tensor (look-back 5), then what a SimpleRNN does with one sample — red dots updated step by step, checked by hand against Keras — and what `return_sequences=True` hands to a stacked layer. Three animated GIFs | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module4/RNN_Samples_and_the_Hidden_State.ipynb) |
| 6-9 | **RNNs By Hand** | SimpleRNN → LSTM → GRU with a pencil: one parameter formula for every cell, `output = units`, and `check()` asserting your arithmetic against Keras layer by layer, single and stacked | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module4/RNNs_By_Hand_basic.ipynb) |

## M4.2 — Implementation

| Videos | Notebook | What you'll do | Open |
| :-- | :-- | :-- | :-- |
| 10-12 | **Univariate Temperature — RNN** | `split_sequence` → the 3-D tensor `(samples, look-back, 1)` → SimpleRNN, then LSTM, then beat mean-only and persistence | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module4/Univariate_Temperature_RNN.ipynb) |
| 13 | **Univariate Temperature — RNN, part 2** | Reload the saved model (no retraining), occlusion, and the autoregressive roll-forward | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module4/Univariate_Temperature_RNN_pt2.ipynb) |
| 14-15 | **Multivariate Occupancy — Lags** | Window method with covariates (HVAC sensors → occupied?), lagged features without leaking the target | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module4/Multivariate_Occupancy_Lags.ipynb) |
| 16-17 | **Multivariate Occupancy — RNN** | `split_sequences` with the target last, a sigmoid head, the LSTM swap, stacking with `return_sequences`, the persistence baseline | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module4/Multivariate_Occupancy_RNN.ipynb) |
| 18 | **Multivariate Occupancy — RNN, part 2** | xAI for RNNs: permutation importance, occlusion and what-if on the reloaded model | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module4/Multivariate_Occupancy_RNN_pt2.ipynb) |

## M4.3 — Advanced topics

Five notebooks, in running order. The primer introduces every new layer; the four after it put those layers to
work, moving from one series to many, then from one answer to a whole day of answers. *In the videos these
notebooks are still called A1-A5 (renamed M4_3a-M4_3e on 10/19/2026 so they don't read like Assignments 5 and 6).*

| Videos | Notebook | What you'll do | Open |
| :-- | :-- | :-- | :-- |
| 19-20 | **M4.3a · Primer for Advanced Topics** | `Conv1D`, `MaxPooling1D`, the three dropouts (with a picture of what `recurrent_dropout` actually cuts), the ConvLSTM chain, `Bidirectional` — and why reading a window backwards is *not* seeing the future | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module4/M4_3a_Primer_for_Advanced_Topics.ipynb) |
| 21-22 | **M4.3b · Univariate Temperature — Advanced** | The same daily-minimum series as M4.1, now with convolution and pooling in front of the recurrent layer: four architectures, each with its shapes and parameter count worked out | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module4/M4_3b_Univariate_Temperature_Advanced.ipynb) |
| 23-24 | **M4.3c · Multivariate Energy — RNN** | Hourly Connecticut demand with weather and the clock: ConvLSTM and bidirectional against the naive baselines, scored in megawatts | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module4/M4_3c_Multivariate_Energy_RNN.ipynb) |
| 25 | **M4.3d · Many-to-Many — Energy** | One model, a whole day of answers: a 24-hour-ahead forecast in a single shot, and what happens to accuracy as the horizon stretches | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module4/M4_3d_Many_to_Many_Energy.ipynb) |
| 26 | **M4.3e · Multi-Step Forecasting** | Recursive vs direct vs multi-output — error by horizon for each, and where the *future* covariates come from (calendar: known; weather: a forecast; past demand: your own guesses) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module4/M4_3e_Multi_Step_Forecasting.ipynb) |

## Assignments (in HuskyCT)

| Assignment | Where | Due | What |
| :-- | :-- | :-- | :-- |
| ✍ **Assignment 5 (RNN Math)** | M4.1 | Fri Nov 6 | Parameter counts and output shapes for recurrent layers, single and stacked, by hand: a PDF of the handwritten math plus the notebook with your bullets |
| 💻 **Assignment 6 (RNN Implementation)** | M4.3 | Fri Nov 13 | Hourly energy demand from weather: at least two recurrent layers, a convolution and dropout, walk-forward validation (50k and 80k samples) against a persistence baseline |

**Archived.** Nine notebooks the Fall 2026 videos don't use live in [`_archive/`](_archive/) and still run:
`Advanced_RNN_Theory`, `Explaining_an_LSTM`, `Forecasting_Electricity_Demand_RNN` (+ `_pt2`),
`Gradient_Saliency_by_Hand`, `Multivariate_Occupancy_RNN_AdvancedTopics`, `Simple_Predict_The_Stock_Market_DL`, and the
two `Many_To_Many_BDL` notebooks. The `_energy_track/` drafts are an alternative M4.2 that was not recorded.

**Part 2 notebooks** never train: they download the `.keras` file the Part 1 notebook saved (published in [`OPIM5509Files/OPIM5509_Module4_Files/models`](https://github.com/drdave-teaching/OPIM5509Files/tree/main/OPIM5509_Module4_Files/models)), rebuild the prep, prove the reload, then test on **unseen data** and explain the model — interpretability for the simple models, not just the capstone.

**Data** loads from stable links in [`OPIM5509Files/OPIM5509_Module4_Files/data`](https://github.com/drdave-teaching/OPIM5509Files/tree/main/OPIM5509_Module4_Files/data): `daily-min-temperatures.csv` (Melbourne/Sydney daily minimums, 10 years), `datatraining.txt` + `datatest2.txt` (UCI room-occupancy sensors, two weeks at 1-minute cadence), `cleanBDL.csv` (Bradley airport weather), `BDL_cleanweather_energy.csv` (hourly Connecticut demand + Bradley weather, 2011–2021 — also the Assignment 2 and Assignment 6 data), and `stock_adjclose_2017_2020.csv` — a snapshot of 11 tickers' adjusted closes replacing the live scrape the old notebook used.

## Guides

| Guide | Use it for |
| :-- | :-- |
| [🎙 Talking Points](../guides/M4_RNN_Talking_Points.md) | Instructor — the 25-video recording plan with running order and anchor numbers |
| [✅ Skills Sheet](../guides/M4_RNN_Skills.md) | The checklist of what you should own before Module 5 |

## Keras 3 / pandas 3 audit (Fall 2026)

All seventeen notebooks were executed top-to-bottom on TensorFlow 2.21 / Keras 3 before this module was recorded. What changed:

- **Stock notebook rebuilt around a stable CSV.** The original pulled prices live through the `yahoo_fin` scraper, which no longer returns data (Yahoo changed). Prices now load from `stock_adjclose_2017_2020.csv` in the course repo — same 11 tickers and dates — with a commented `yfinance` cell for anyone who wants fresh data. Its import block also dropped the dead Keras-2 paths (`keras.preprocessing.text`, `keras.utils.np_utils`, `keras.layers.convolutional`) and the unused text-model imports.
- **`fillna(method='ffill')` → `.ffill()`** in the multi-step many-to-many notebook (pandas 3 removed the `method=` argument), and **`df.drop(axis=1, columns=[...])` → `df.drop(columns=[...])`** in both many-to-many notebooks (pandas 3 rejects passing `axis` together with `columns`).
- **Two content fixes:** the multivariate RNN notebook titled itself "Ozone" (it's the occupancy data), and the advanced-topics notebook said "conv2d" where the layer is — and must be — `Conv1D`.
- **Colab badges** re-pointed from the old OPIM5509Files path to this repo, and **15 🔴 recording markers** added (a bare red dot; talking points live in the cell's HTML comment).
- **Every notebook is now seeded** (`keras.utils.set_random_seed(5509)` right after the imports — none of the original eleven had a seed before), so your run reproduces the stored outputs exactly on CPU; a Colab GPU can drift by a little because some cuDNN recurrent kernels are non-deterministic, and that's fine.
- **Every fitting notebook ends with "Save the model and use it again":** the final model is saved to a single `.keras` file, reloaded with `load_model`, and checked to give identical predictions — reproducibility is a seed *and* a saved artifact.
- **Occupancy data upgraded.** The three occupancy notebooks used the 2-day `datatest.txt` slice (2,665 rows — the 50/50 split trained on about a day). They now stack the real UCI training and test files into one two-week series (17,895 rows) and keep the chronological split. With 7× the rows, the fits use `batch_size=64` instead of 5–10 so each epoch stays seconds long on CPU/Colab (early stopping is unchanged).
- **New capstone notebook:** `Forecasting_Electricity_Demand_RNN.ipynb` — the Assignment 2 demand data treated as a sequence, with a business framing (you're the utility) and the baselines that make an RNN forecast honest.
- **Two more notebooks (planned as videos 18–19):** `M4_3e_Multi_Step_Forecasting` (recorded as video 26) (recursive vs direct vs multi-output, exposure bias, future covariates) and `Explaining_an_LSTM` (now in `_archive/`; permutation, occlusion, saliency, integrated gradients, what-if, and SHAP's `GradientExplainer` compared against integrated gradients). SHAP notes: `DeepExplainer` does not work on Keras 3, and SHAP needs NumPy < 2.3 (Colab is fine; a fresh local TensorFlow install may need `pip install "numpy<2.3" shap`).
- The red `input_shape` UserWarning Keras 3 prints on the first model cell is harmless — same as Modules 2–3.
