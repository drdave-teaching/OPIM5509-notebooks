# M4 Skills — Recurrent Networks for Numeric Sequences

**OPIM 5509 - Introduction to Deep Learning · Dr. Dave Wanik · University of Connecticut**
*Fall 2026 · what you can do after Module 4 · mapped to the videos and notebooks*

## M4.1 — Theory, by hand, univariate

- [ ] Explain why **sequence data is different** — order carries information — and why shuffling a time series destroys it
- [ ] Apply the **window method** (lags): turn a series into a supervised table with `n_steps` past values as features, and know that any model can then fit it
- [ ] Split a time series **chronologically** (no shuffle) and explain the leakage that shuffling would cause
- [ ] Describe a **SimpleRNN** one time step at a time: the hidden-state handoff, tanh, and why the final hidden state summarizes the whole window
- [ ] State that a recurrent layer's **output shape equals its number of units**, independent of look-back
- [ ] Count recurrent-cell parameters with **G · [H(H+I) + H]** and know G = 1 (SimpleRNN), 3 (GRU), 4 (LSTM); reproduce 15, 1,426, 48, 4,320 by hand
- [ ] Explain the **LSTM's four networks and cell state** (long memory) vs. hidden state (recent memory), and why gates fix the vanishing gradient
- [ ] Swap `SimpleRNN` → `LSTM` → `GRU` in Keras with everything else unchanged
- [ ] Shape a univariate series into the **3-D tensor `(samples, look-back, features)`** with `split_sequence` + `reshape`, and read the shape back (e.g., `(3640, 10, 1)`)
- [ ] Fit a SimpleRNN / LSTM regressor with early stopping and compare MAE against the window-method dense net
- [ ] Build and **beat the baselines** — mean-only, **persistence** (shift-1), linear on lags — and report metric + scatter + time-series plot together

## M4.2 — Multivariate, stock

- [ ] Build **lag features from multiple covariates** and drop the lagged target to avoid leakage (occupancy example)
- [ ] Use `split_sequences` for multivariate data: X columns left, **target last**, look-back as the hyperparameter; inherit `n_steps`/`n_features` from the tensor shape
- [ ] Put a **classification head** (sigmoid, binary crossentropy) on a recurrent model and read a confusion matrix *and* a time-series plot of predictions
- [ ] Stack recurrent layers with **`return_sequences=True`** and explain why deeper isn't automatically better
- [ ] Run an honest end-to-end sequence classifier on **stock returns** (percent change, next-day shift, scaling, look-back 5) and interpret a near-50% result correctly

## M4.3 — Advanced + the capstone

- [ ] Work a **`Conv1D` + `MaxPooling1D`** layer by hand on a sequence — output length, filter count, parameter count — and say why it's `Conv1D`, not `Conv2D`
- [ ] Add **recurrent dropout**, stacking, and **`Bidirectional`** wrappers, and read the concatenated output width off `model.summary()`
- [ ] Assemble a **ConvLSTM** (conv + pooling → recurrent) for univariate and multivariate series with correct `input_shape`
- [ ] Build a **many-to-many** model: several targets at once (`Dense(k, linear)`, target columns last) and **multi-step-ahead** forecasts, and explain how quality degrades with horizon
- [ ] Take a raw hourly series (electricity demand) from file to forecast: **check it is sorted**, split by year, and score **mean-only / seasonal-naive / persistence** before any network
- [ ] Encode the clock as **sine/cosine** features and put the target **last** for `split_sequences`; explain why weather barely helps at one hour ahead
- [ ] Build a **24-hour-ahead** model (`Dense(24)`) and read an **MAE-by-horizon** plot against the seasonal-naive baseline
- [ ] Frame a **peak-hour** flag as imbalanced classification and judge it by precision/recall, not accuracy
- [ ] Save a trained model to a `.keras` file, reload it, and prove the predictions are identical (seed + saved artifact = reproducibility)
- [ ] Name the three multi-step strategies — **recursive** (autoregressive), **direct**, **multi-output** — and explain exposure bias: a model trained on true history but run on its own predictions
- [ ] Classify a covariate at a future hour as *known* (calendar), *forecastable* (weather → use the forecast, inherit its error) or *the target itself* (only real history, or your own predictions) and choose the strategy accordingly
- [ ] Explain a sequence model with **permutation importance**, **occlusion by time step**, **gradient saliency** (`tf.GradientTape`) and **integrated gradients**, read the hours × features heatmap, and verify the completeness check
- [ ] Run a **what-if** (counterfactual) through a trained LSTM and interpret the response — and know which SHAP explainer works on a Keras 3 RNN (`GradientExplainer`, not `DeepExplainer`)

## Assessed by

**Assignment 5 (RNN Math)** — parameter counts and output shapes for recurrent, Conv1D, and bidirectional layers, by hand.
