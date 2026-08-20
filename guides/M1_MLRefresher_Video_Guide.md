# M1 Video Guide — Machine Learning Refresher

**OPIM 5509 - Introduction to Deep Learning · Dr. Dave Wanik · University of Connecticut**
*Fall 2026 · Week 1*

Eight videos, roughly 55 minutes. You get your environment working, meet a real dataset, find two genuine problems in it, and rebuild the machine learning workflow that every neural network in this course will sit on top of.

The 🔴 markers inside the notebooks show exactly where each video starts.

**New for Fall 2026:** Module 1 has moved off Boston Housing. scikit-learn removed that dataset in version 1.2 because one of its columns was an explicit racial proxy, so modern installs cannot even fetch it. We now use **California Housing**, which loads in one line from scikit-learn and carries two teaching landmines of its own.

---

## Notebook 1 · `Module1/0_Welcome_and_Setup.ipynb`

**The big ideas:** Colab as the course computer · save-a-copy before you type · reading library versions · where notebooks, data, the ebook, and grades each live · `.shape` as the language of deep learning.

**Video 1 — Welcome to OPIM 5509**
What the semester looks like and what Module 1 is for. Module 1 is a *refresh*, not new material — if it feels easy that is the design working, and if it feels hard that is an early warning worth acting on in week one rather than week nine. Covers the prerequisites honestly (5604 required, 5512 strongly suggested), the two textbooks, and the async rhythm.

**Video 2 — Setting up Colab and running a course notebook**
Done live on screen: open the repo, click the Colab badge, `File > Save a copy in Drive`, `Runtime > Run all`. Includes the deliberate demo of the bug every student hits eventually — running cells out of order, getting a `NameError`, and fixing it with restart-and-run-all. Ends by pointing at the GPU menu without switching to it, so the path is familiar when Module 3 needs it.

---

## Notebook 2 · `Module1/1_CaliforniaHousing_EDA.ipynb`

**The big ideas:** unit of analysis · `.info()` as a reflex · censored targets · impossible averages and small-denominator artifacts · multicollinearity · geography as ordinary numeric features.

**Video 3 — EDA Part 1: meeting the data and reading its shape**
Loading in one line, then the question that comes before every other question: **what is a row?** Here a row is a census block group of roughly 600 to 3,000 people, not a house — so every "Ave" column is an average *within* a block group. Walks `.shape`, `.dtypes`, `.info()`, and `.isnull().sum()`, then lands on `.describe()` and asks the students to find the weird numbers themselves before video 4 reveals them.

**Video 4 — EDA Part 2: finding the landmines, then visualizing**
The payoff. **Landmine one:** `MedHouseVal` is capped at exactly 5.00001 — the 1990 census top-coded home values at \$500,001, so about 965 block groups sit on a ceiling and no model can ever predict past it. **Landmine two:** `AveRooms` reaches 141 and `AveOccup` reaches 1,243, because those block groups have almost no households and the average is a small number over a smaller one. Then the visual build-up: histograms, boxplots, a flag variable, a group-by, the correlation heatmap (`MedInc` at 0.69; `AveRooms`/`AveBedrms` at 0.85 and the multicollinearity conversation), and finally the latitude/longitude scatter where California draws itself out of two ordinary numeric columns.

---

## Notebook 3 · `Module1/2_AllTheModels_Regression.ipynb`

**The big ideas:** the five-step skeleton · `random_state` and reproducibility · data leakage · MAE vs RMSE vs R² · the train/test gap as visible overfitting · residuals that remember your EDA.

**Video 5 — Regression Part 1: the skeleton, split, scale, and leakage**
Setting up X and y, the 80/20 split, and a real conversation about why `random_state` matters — without it you cannot tell a genuine improvement from a luckier shuffle. Then the centerpiece: `fit_transform` on train, `transform` on test, and the sentence worth repeating out loud — *scale before you split and the test set's maximum leaks into your scaler, so your test score becomes a lie.* Includes the check students always ask about: yes, the test set can land slightly outside 0 to 1, and that is correct.

**Video 6 — Regression Part 2: five models and reading the results**
Instantiate, fit, predict — five times, first by hand and then as a loop. Metrics with their jobs spelled out, and the MAE converted into real money so it means something to a stakeholder. The decision tree scores near-perfect on train and mediocre on test; the random forest closes that gap by averaging a hundred trees, and overfitting stops being a definition and becomes a bar chart. The predicted-vs-actual plot brings back the censoring from video 4 as a visible stripe of residuals along the top. Closes with the bridge: next module that one model line becomes `Sequential([Dense(...)])` and nothing else changes.

---

## Notebook 4 · `Module1/3_AllTheModels_Classification.ipynb`

**The big ideas:** recoding a target with one `np.where` · target leakage · the confusion matrix in business language · precision vs recall · **`predict_proba` as the bridge to the sigmoid** · AUC as threshold-free comparison.

**Video 7 — Classification: a different question, the same skeleton**
The same data asked a yes/no question. Recoding at the median gives a balanced 50/50 problem by construction — which sets up the warning that real problems are never balanced and that accuracy is useless when they are not. Split and scale get thirty seconds because they are identical to last time. Then the confusion matrix, walked slowly in the language of the actual problem: a false positive is a wasted marketing dollar, a false negative is a missed opportunity, and which one hurts more is a business question, not a technical one. The most important minute of Module 1 is here — `predict()` is just `predict_proba()` compared against 0.5, and in two weeks that probability comes out of a sigmoid at the end of a neural network while everything you do with it stays the same.

---

## Notebook 5 · `Module1/Assignment1_OPIM5509.ipynb`

**Video 8 — Assignment 1 walkthrough and where Module 2 picks up**
Reading the questions without solving them, and setting expectations for what a good answer looks like. Points at the traps without giving them away: missing values in every column, a column whose missing values are hiding as the *string* `"N/A "` so `.isnull()` reports zero, a few duplicated rows, and a target that is only weakly predictable by design. Sets the expectation out loud that **about 0.5 test R² is the right answer here** — a fat cloud along the 45-degree line, not a tight diagonal — so nobody panics. Closes Module 1 with the skeleton one last time and the hook for Module 2: forward propagation, by hand, on paper.

---

## The through-line

Everything in Module 1 exists to make one picture obvious:

```
  read data     ->  split  ->  scale on train  ->  [ MODEL ]  ->  fit / predict  ->  evaluate
```

Only the bracketed step changes for the rest of the semester. Dense networks, convolutional networks, recurrent networks — they all drop into the same slot.

---

*Companions: [Skills](M1_MLRefresher_Skills.md) · [Talking Points](M1_MLRefresher_Talking_Points.md). Next up: Module 2 — Dense Neural Networks.*
