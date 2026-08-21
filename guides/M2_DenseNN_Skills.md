# M2 Skills Sheet — Dense Neural Networks

**OPIM 5509 - Introduction to Deep Learning · Dr. Dave Wanik · University of Connecticut**
*Fall 2026 · Weeks 3–5*

After Module 2, these are the skills you own. Check yourself off — the notebook to revisit is named for each block. Everything from Module 1 is assumed.

*(Drafted from the recording plan; refreshed from transcripts once the new videos are up.)*

---

## ✋ Forward propagation, by hand
*Notebook: `Module2/1_ForwardPropagation.ipynb`*

- [ ] Compute a prediction from one input and one weight — and say why the weights start random
- [ ] Compute a dot product by hand: element-wise multiply, then sum
- [ ] Apply the **golden rule of shapes**: inner dimensions must match, and they cancel (1×10 · 10×7 = 1×7)
- [ ] Chain dot products through a hidden layer (1×3 · 3×5 = 1×5, then onward to the output)
- [ ] **Count trainable parameters** for any dense stack: weights *plus* one bias per node (3→5→1 = 20 + 6 = 26)
- [ ] Say what "deep" literally means (more stacked layers — nothing else)

## 🌡️ How networks learn
*Notebooks: `Module2/2_HotAndCold.ipynb` · `Module2/3_BackProp_and_ReLU.ipynb`*

- [ ] Walk the loop: predict → compare (error = predicted − actual) → update
- [ ] Explain hot-and-cold learning and why a constant step is slow (~400 epochs) and can overshoot
- [ ] Write the real update: `weight -= (error × input) × learning_rate` — big miss big step, small miss small step
- [ ] Say why we square the error (a parabola whose tangent slope gives direction *and* amount — a derivative)
- [ ] Demonstrate **divergence** and name the two defenses: scale your inputs, use a learning rate
- [ ] Distinguish stochastic / full / **batch** gradient descent — and connect batch to Keras's `batch_size`
- [ ] Explain why stacking linear layers is pointless (2×2×2 is just ×8) and what **ReLU** adds
- [ ] Trace one backward pass: overestimate → shrink only the weights whose inputs carried information

## 🏗️ Keras for regression
*Notebooks: `Module2/CA_Housing_Regression.ipynb` · `Module2/CheatSheet_BuildingFFNNs.ipynb`*

- [ ] One-hot encode a categorical column with `get_dummies` — and say why a dot product needs numbers
- [ ] Build a `Sequential` stack of `Dense` layers: ReLU hidden, **linear output** for regression
- [ ] Explain `input_shape=(features,)` and what the `None` dimension is
- [ ] Compile with an optimizer, a loss, and a metric — and say what each one is for
- [ ] Use **early stopping** with `restore_best_weights` instead of guessing an epoch count
- [ ] Plot the learning curve from the fit history and point at where early stopping fired
- [ ] Read train-vs-validation loss for overfitting (it's the M1 train/test gap, live)
- [ ] Report the MAE in dollars, and spot the M1 censoring ceiling in the prediction scatter
- [ ] Explain dropout (rotate the bank tellers), place it after a Dense layer, pick 0.1–0.5
- [ ] Choose a starting architecture from the four cheat-sheet strategies — and start simple

## 🎯 Keras for classification
*Notebooks: `Module2/0_...Titanic...` · `1_...Iris...` · `2_...MNIST...` · `3_Fashion_MNIST...` · `4_IMDB...`*

- [ ] State the majority-class **baseline** before fitting anything (Titanic: 61.7%)
- [ ] Name the two — and only two — changes from regression to binary classification: sigmoid output, `binary_crossentropy`
- [ ] Read a sigmoid output as a probability and connect it to `predict_proba` from Module 1
- [ ] Go multiclass: one output node per class, **softmax**, `categorical_crossentropy`, dummy-encoded y, `argmax` to predict
- [ ] Explain why sorted data (Iris!) must be shuffled before a validation split
- [ ] Flatten a 28×28 image into 784 features and scale by 255
- [ ] Say why a dense net fails on off-center digits — and what Module 3 does about it
- [ ] Encode text as one-hot word-presence, watch the model overfit, and stop it
- [ ] Say what information one-hot text encoding throws away — and which modules get it back

## 🌉 Carrying it forward

- [ ] Given any Keras `model.summary()`, verify the parameter counts by hand
- [ ] Point at each line of a Keras script and name the by-hand concept it automates
- [ ] Say which of this module's ideas Conv2D changes in Module 3 — and which it doesn't touch
