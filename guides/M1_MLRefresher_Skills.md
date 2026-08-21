# M1 Skills Sheet — Machine Learning Refresher

**OPIM 5509 - Introduction to Deep Learning · Dr. Dave Wanik · University of Connecticut**
*Fall 2026 · Week 1*

After the nine Module 1 videos (51:33), these are the skills you own. Check yourself off — if one feels shaky, the video and notebook to revisit are named right there.

Nothing on this list is new material. All of it is assumed from here forward.

---

## 🧰 Environment and workflow
*Notebook: `Module1/0_Welcome_and_Setup.ipynb`*

- [ ] Open any course notebook in Colab from its badge and run it top to bottom (**Runtime → Run all**)
- [ ] Save your own copy to Drive **before** you start typing, and explain why that matters
- [ ] Diagnose a `NameError` caused by running cells out of order, and fix it with restart-and-run-all
- [ ] Print the versions of numpy, pandas, matplotlib, seaborn, and scikit-learn — and know why you'd want to
- [ ] Find `Runtime → Change runtime type` and say when in this course you will need a GPU
- [ ] Read a `.shape` out loud and say what each number means
- [ ] Explain why selecting a GPU in Colab is **not** enough — your code has to be written to use it
- [ ] Find the Chollet textbook free through library.uconn.edu with your NetID

## 🔍 Exploratory data analysis
*Notebook: `Module1/1_CaliforniaHousing_EDA.ipynb`*

- [ ] State in one plain sentence **what a single row represents** before touching a model
- [ ] Use `.info()` as a reflex and get shape, dtypes, and missingness from one call
- [ ] Count missing values per column and across the whole table
- [ ] Read `.describe()` critically — hunt the `min` and `max` rows for values that cannot be true
- [ ] Spot a **censored (top-coded) target** from a histogram and explain what it costs your model
- [ ] Spot **impossible averages**, explain the small-denominator mechanism behind them, and make a defensible keep-or-drop call
- [ ] Build a flag/dummy/indicator variable with `np.where()`
- [ ] Compare two populations with a `groupby()` pivot table and say what the difference means
- [ ] Read a correlation heatmap, and name **multicollinearity** when two features carry the same information
- [ ] Build histograms, boxplots, and scatterplots — every one of them titled and axis-labeled
- [ ] Treat latitude and longitude as ordinary features and produce a map from a scatterplot
- [ ] Say why mean and standard deviation are most meaningful for **normal** data, and reach for percentiles when the sd exceeds the mean
- [ ] Apply the **reasonable person test** — does the story the data tells match what you'd expect of the business? If it's backwards, raise it
- [ ] Argue *against* deleting outliers: they are often the most interesting thing in the business
- [ ] Order bubbles largest-underneath so small ones aren't hidden

## 🏗️ The modeling skeleton
*Notebook: `Module1/2_AllTheModels_Regression.ipynb`*

- [ ] Recite the skeleton without looking: **read → split → scale on train → fit → evaluate**
- [ ] Separate `X` from `y` and explain why `X` has one fewer column
- [ ] Split 80/20 with `shuffle=True` and a fixed `random_state`, and explain what `random_state` buys you
- [ ] Apply `fit_transform()` to train and `transform()` to test — and explain **data leakage** in one sentence
- [ ] Explain why test values landing outside 0 to 1 after scaling is correct, not a bug
- [ ] Fit `LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`, `GradientBoostingRegressor`, and `KNeighborsRegressor` using identical surrounding code
- [ ] Write a reusable evaluation function instead of copy-pasting the metric block five times
- [ ] **Save your feature names before scaling** — the array survives, the names don't, and you'll need them for permutation importance
- [ ] Explain why `df` is never mutated (no `inplace=True`) and why that matters when you re-run cells

## 📏 Evaluating regression
*Notebook: `Module1/2_AllTheModels_Regression.ipynb`*

- [ ] Choose between **MAE**, **RMSE**, and **R²** and defend the choice
- [ ] Translate an MAE into the units a stakeholder actually cares about (dollars, not decimals)
- [ ] Diagnose **overfitting** from a train/test gap without being told it is there
- [ ] Explain why reporting training R² as your result is meaningless
- [ ] Build a predicted-vs-actual plot with a 45° reference line and read its patterns
- [ ] Connect a pattern in the residuals back to something you found during EDA
- [ ] Rank features with built-in tree importance **and** `permutation_importance`, and say why the second is more trustworthy
- [ ] Explain **why random forests do so well on tabular data** — many small trees on random subsets of rows and columns, averaged into a stable estimate
- [ ] Say where a neural network wins instead (unstructured data — it does the feature engineering for you)
- [ ] Name the dial that stops a decision tree overfitting (`min_samples_leaf` / `min_samples_split`)

## 🎯 Evaluating classification
*Notebook: `Module1/3_AllTheModels_Classification.ipynb`*

- [ ] Turn a regression problem into a classification problem with one `np.where()`
- [ ] Explain **target leakage** and spot a feature that secretly contains the answer
- [ ] Use `stratify=y` in a split and say what it protects you from
- [ ] Read all four quadrants of a confusion matrix **in the language of the business problem**
- [ ] Define **precision** and **recall** without hedging, and say which one your problem cares about more
- [ ] Explain why accuracy is a dangerous metric on imbalanced classes
- [ ] Get raw probabilities from `predict_proba()` and show that `predict()` is just a threshold at 0.5
- [ ] Move the threshold and describe how precision and recall trade against each other
- [ ] Interpret an **AUC** and explain why it is threshold-free
- [ ] Read a `predict_proba` score as model **confidence** — 0.99 is confident, 0.53 is not
- [ ] Say why the raw house value had to be dropped from `X`, not just the flag
- [ ] Give a real situation where a "chatty" model with false alarms is the *right* call

## 📈 Optional — ROC, AUC, and thresholds
*Notebook: `Module1/5_Appendix_ROC_AUC_and_Thresholds.ipynb` (no video)*

- [ ] Build an ROC curve from a confusion-matrix table **by hand**, without calling a library
- [ ] Explain why each step on the curve goes **up** for a true positive and **right** for a false positive
- [ ] Say what lowering the threshold does to TPR *and* FPR at the same time — and why you cannot improve one for free
- [ ] Define AUC two ways: area under the curve, **and** the probability a random positive outranks a random negative
- [ ] Explain why AUC measures **ranking**, not calibration, and does not choose your threshold
- [ ] Pick a threshold from the **cost** of each mistake instead of defaulting to 0.5
- [ ] Say why ROC flatters a model when positives are rare, and what to report instead

## 🌉 Carrying it into Module 2

- [ ] Say what changes when a random forest becomes a neural network — and what does not
- [ ] Explain why a probability from `predict_proba()` and a probability from a **sigmoid** get handled identically
- [ ] Point at any step of the skeleton and name the deep learning equivalent

---

**If more than a few boxes are unchecked, come to office hours in week one.** Module 2 assumes all of this and does not slow down for it.
