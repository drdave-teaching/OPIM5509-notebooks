# M2 Skills Sheet — Dense Neural Networks

**OPIM 5509 - Introduction to Deep Learning · Dr. Dave Wanik · University of Connecticut**
*Fall 2026 · Weeks 3–5*

After Module 2, these are the skills you own. Check yourself off — the notebook and video to revisit are named for each block. Everything from Module 1 is assumed.

*(Refreshed from the Fall 2026 recordings — 21 videos, 2:11:31.)*

---

## ✋ Forward propagation, by hand
*Notebook: `Module2/1_ForwardPropagation.ipynb` · Videos 1–3*

- [ ] Name the three parts of a neural network operation: the forward pass, the error signal, the backpropagation of that signal
- [ ] Compute a prediction from one input and one weight (85 × 0.9 = 76.5) — and say why the weights start random
- [ ] Compute a dot product by hand: element-wise multiply, **then sum** — and say why you sum (you're predicting one thing, not three)
- [ ] Explain what a dot product *means*: it measures similarity — vectors pointing the same way maximize it, perpendicular ones give zero
- [ ] Apply the **golden rule of dot products**: columns of A must equal rows of B; the output is (rows of A) × (columns of B), and the inner dimensions cancel
- [ ] Chain dot products through a hidden layer (1×3 · 3×5 = 1×5, then 1×5 · 5×1 = 1×1)
- [ ] Describe a hidden layer as a *reprojection* of the input into a new dimensional space — "a distillation enhancement"
- [ ] **Count trainable parameters** for any dense stack: one weight per arrow, **plus one bias per output node** (3→5 = 15 + 5 = 20; then 5→1 = 5 + 1 = 6)
- [ ] Say why everything up to this point is still linear — and could collapse into a single calculation

## 🌡️ How networks learn
*Notebook: `Module2/2_HotAndCold.ipynb` · Videos 4–6*

- [ ] Walk the loop: predict → compare (error = predicted − actual) → update
- [ ] State the core rule: **overestimated → shrink the weight; underestimated → grow it**
- [ ] Explain hot-and-cold ("Goldilocks") learning — try a bigger weight, try a smaller one, keep the winner — and why it's silly at scale (~450 tries)
- [ ] Say why we square the error (extremes count more, and it makes a parabola whose slope gives direction *and* amount)
- [ ] Write the real update: **direction and amount = error × input**, scaled by a learning rate — big miss, big step
- [ ] Recite **P-E-D-W**: Prediction, Error, Direction-and-amount, Weight update
- [ ] Explain the **negative sign reversal** — subtracting a negative direction raises the weight
- [ ] Compare the two approaches honestly: ~450 epochs brute force vs. ~30 with derivatives
- [ ] Demonstrate **divergence** (prediction −3,070 → billions → e³⁴) and name the two defenses: **scale your data, use a small learning rate**

## 🔁 Backpropagation and ReLU
*Notebook: `Module2/3_BackProp_and_ReLU.ipynb` · Videos 7–10*

- [ ] One-hot encode words into numbers, and say why: "you can't do math on words like hot and cold"
- [ ] Explain **error attribution** — only weights that carried information into the prediction get updated; an input of 0 sends nothing, so its weight never moves
- [ ] Distinguish stochastic / full / **batch** gradient descent — and connect batch to Keras's `batch_size` (count by eight: 8, 16, 32, 256)
- [ ] Explain why stacking linear layers is pointless, and what **ReLU** adds: indirect correlations, plus different regions of the network firing at different times
- [ ] Define `relu(x)` — negatives become 0, otherwise return x — **and distinguish it from `relu2deriv(x)`**, which returns **1 where x is positive and 0 otherwise**. One transforms the data on the way *forward*; the other is the **gate on the way back**, deciding which weights are allowed to update. Same test (`x > 0`), completely different jobs — this is the single most confusing pair in the module
- [ ] Trace one full iteration on paper: layer 0 → `weights_0_1` → ReLU → layer 1 → `weights_1_2` → prediction 0.33 vs. a target of 1
- [ ] Read `layer_2_delta` (−0.667) and say which weights grow, which shrink, and **which stay exactly the same, and why**
- [ ] Point at the one weight that changed (0.44 → 0.49) and justify it from the input row `[1, 0, 0]`
- [ ] Compare the linear plateau (~7.5) with the ReLU network's error (~5.5) and say what bought the difference
- [ ] Read a weight-trajectory plot: some weights evolve, some never contribute — a first look at **dead neurons**
- [ ] Summarize a neural network in one sentence: **a nonlinear weighted sum**

## 🏗️ Keras for regression
*Notebooks: `Module2/CA_Housing_Regression.ipynb` · `Module2/CheatSheet_BuildingFFNNs.ipynb` · Videos 11–15*

- [ ] One-hot encode a categorical column with `get_dummies` — and say why a dot product needs numbers
- [ ] **Fit the scaler on `X_train` only**, then apply it to test — and explain why scaling before splitting is data leakage
- [ ] Choose the Sequential API over the functional one (and know that this class requires it)
- [ ] Build a `Sequential` stack of `Dense` layers: ReLU hidden, **linear output** for regression
- [ ] Explain `input_shape=(features,)` — features only, because rows are handled by `batch_size`
- [ ] Avoid hard-coding: pull the input width from `X_train.shape[1]`
- [ ] Verify `model.summary()` by hand (13×50 + 50 = 700; 50×25 + 25 = 1,275; total 2,246)
- [ ] Compile with an optimizer, a loss, and a metric — and say what each is for; default to **Adam** for its adaptive momentum
- [ ] Use **early stopping** with `restore_best_weights=True` instead of guessing an epoch count — and say what you get without that argument
- [ ] Assign the fit to `history`, plot the learning curve, and point at where early stopping fired
- [ ] Read train-vs-validation loss for overfitting — and know that train will *always* drift down, so judge on validation
- [ ] Report the MAE in dollars (~$45k), and spot the M1 censoring ceiling in the prediction scatter against the 45° line
- [ ] Explain **dropout** with the bank-teller analogy, place it after a Dense layer, and pick a rate (start ~0.1, stay under 0.5)
- [ ] Say why **dropout adds zero trainable parameters**
- [ ] Predict what more dropout does to a learning curve (**gentler, slower**) — and note the final metrics may land in the same neighborhood
- [ ] Choose a starting architecture from the four strategies — 1× the input width, 2× the input width, stacked same-width layers, or an information funnel — and **start simple**

## 🎯 Keras for classification
*Notebooks: `Module2/0_...Titanic...` · `0b_...TrainValTest` · `0c_...KFold` · `1_...Iris...` · Videos 16–21*

- [ ] State the majority-class **baseline** before fitting anything (Titanic: ~62% — always predict "did not survive") and say why a model that can't beat it has lost to no model at all
- [ ] Name the two — and only two — changes from regression to binary classification: **sigmoid output**, **`binary_crossentropy`**
- [ ] Say what goes wrong without sigmoid: a linear output happily predicts −0.3 or 2.7
- [ ] Motivate cross-entropy by confidence: 0.03 and 0.992 beats 0.48 and 0.52
- [ ] **Clear the session AND redefine the architecture** before refitting — clearing alone is not enough, or the new model inherits the old weights and starts at the old accuracy
- [ ] Split before you judge: fit on `X_train`, evaluate the confusion matrix on `X_test`
- [ ] Produce a **confusion matrix and classification report**, and read them in domain terms — high recall on non-survivors, poor recall on survivors, and what those false negatives mean
- [ ] Distinguish **validation** (steers early stopping, tunes the model) from **test** (never, ever seen by the model)
- [ ] Build a 60/20/20 train/validation/test split with `stratify`, and use `validation_data=` instead of `validation_split=`
- [ ] Keep test looking like the real world — any oversampling or SMOTE happens on **train only**
- [ ] Run **k-fold cross-validation**, clearing and redefining the model inside every fold, and say what leakage you'd cause otherwise
- [ ] Report k-fold results as **mean ± standard deviation**, and explain why one split is a lottery ticket
- [ ] Go multiclass: **one output node per class**, **softmax**, **`categorical_crossentropy`**, dummy-encoded y
- [ ] Explain softmax as one unit of probability mass split across classes (70% / 20% / 10% = 100%)
- [ ] Use **`argmax`** to convert three-column predictions back to a single column for the confusion matrix
- [ ] Explain why sorted data (Iris!) must be shuffled before a validation split
- [ ] Flatten a 28×28 image into 784 features and scale by 255
- [ ] Say why a dense net fails on off-center digits — and what Module 3 does about it
- [ ] Encode text as one-hot word-presence, watch the model overfit, and stop it
- [ ] Say what information one-hot text encoding throws away — and which modules get it back

## 🌉 Carrying it forward

- [ ] Given any Keras `model.summary()`, verify the parameter counts by hand
- [ ] Point at each line of a Keras script and name the by-hand concept it automates
- [ ] Say which of this module's ideas Conv2D changes in Module 3 — and which it doesn't touch
