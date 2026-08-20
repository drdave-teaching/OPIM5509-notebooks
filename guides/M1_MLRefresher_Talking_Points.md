# M1 Talking Points — Machine Learning Refresher

**OPIM 5509 - Introduction to Deep Learning · Dr. Dave Wanik · University of Connecticut**
*Fall 2026 · Week 1 · recording notes*

Everything below also lives inside the notebooks as hidden comments under each 🔴 marker. This file is the version to read *before* you hit record.

**Marker convention:** a 🔴 cell renders as a bare red dot and nothing else. The video number, title, and all of these notes sit inside an HTML comment so students never see them. Double-click the cell to read them while recording.

**Target runtime:** 8 videos, roughly 55 minutes total. Videos 4 and 6 are the long ones.

---

## Video 1 — Welcome to OPIM 5509 *(~5 min)*
`0_Welcome_and_Setup.ipynb`

- OPEN with the promise: by December you will build models that read text, watch images, and forecast time series. Today we make sure your laptop can run them.
- Set expectations about Module 1: this is a **refresh**. If it feels easy, good — that is the point. If it feels hard, that is an early warning signal and office hours start Tuesday.
- Name the prereq honestly: 5604 required, 5512 strongly suggested. Students who skipped 5512 are not doomed, but Module 1 is where they find out how much catching up they have to do.
- The two textbooks: Chollet (buy this one — the Keras author wrote it) and Trask (*Grokking* — the by-hand intuition). Both reachable through the UConn library; show the library link path on screen.
- The course rhythm: async videos + notebooks, one assignment per module, group project at the end.
- **Do not mention the red dot markers.** They are for you, not for them.

## Video 2 — Setting up Colab and running a course notebook *(~7 min)*
`0_Welcome_and_Setup.ipynb`

- **Show, do not tell.** Screen share: go to the GitHub repo, open a notebook, click the Colab badge, watch it load.
- Then immediately `File > Save a copy in Drive`. Show the "Copy of ..." title change. Say out loud: *"this is now MY notebook."*
- **"Where did it go?!"** Ask it out loud right after the save, then answer it: `My Drive / Colab Notebooks / Copy of 0_Welcome_and_Setup.ipynb`. **Always** that folder — Colab creates it on the first save and never asks again. Opening from GitHub is irrelevant; the copy does not land near GitHub.
- Demo `File > Locate in Drive` live. Fastest answer to the question you would otherwise field by email.
- Tell them to **rename on save** and to make a `Drive / OPIM 5509 /` folder now. Six files called `Copy of ...` in one flat folder is unusable by week four.
- Note that `File > Save a copy in GitHub` commits to a repo **they** own — it cannot write to yours, so nobody can break the class by clicking it. Drive is the simpler habit here.
- Show `Runtime > Run all`. Let it churn. Point out the cell numbering `[1] [2] [3]` as it goes — that ordering **is** the mental model.
- Demo the classic student bug live: run a cell out of order, get a `NameError`, then show restart-and-run-all as the fix. They will hit this in week 3 at 11pm.
- Run the version cell. Tell them: if a version mismatch ever bites you, screenshot **this** cell into your office hours question.
- `Runtime > Change runtime type > GPU`. Show the menu, do not switch yet. Plant the flag for Module 3.
- Mention the 12-hour Colab timeout and that long training runs need checkpoints — a preview of Module 2.

---

## Video 3 — EDA Part 1: meeting the data and reading its shape *(~7 min)*
`1_CaliforniaHousing_EDA.ipynb`

- OPEN with the framing: *"we are refreshing, but I am going to teach EDA the way I want it done in your final project."*
- Say why Boston Housing is gone: scikit-learn **removed** it in 1.2 because one of its columns was an explicit racial proxy, and modern sklearn will not even fetch it. Be direct and brief — one minute, do not turn it into a lecture. Then move on.
- Load in one line. Emphasize: no Drive, no CSV, no upload. This is the standard for the whole course now.
- **Unit of analysis is the big idea of this segment.** Each row is a census *block group* — roughly 600 to 3,000 people — not a house. So `AveRooms` is average rooms per household *in that block group*. Students who forget this write nonsense in their write-ups.
- Walk `.shape`, `.columns`, `.dtypes`, `.info()`, `.isnull().sum()`. Say the sentence: *"info() gives me shape, dtypes, and missingness in one call."*
- Point out this data has **no** missing values and say why that is unrealistic — real data is messy, and Assignment 1 will be.
- Then `.describe()` and **slow down**. Both landmines are visible here. Do not reveal them — ask the students to stare at the table and find the weird numbers. Tease into video 4.

## Video 4 — EDA Part 2: finding the landmines, then visualizing *(~11 min)*
`1_CaliforniaHousing_EDA.ipynb`

Pay off the cliffhanger. Two landmines:

1. **`MedHouseVal` is capped at 5.00001** (\$500,001). About 965 block groups sit exactly on that ceiling. Show the `value_counts()` cell, then the histogram spike on the right edge. This is **censored data**. Connect it back to Boston Housing where `medv` topped out at exactly 50 — same phenomenon, and students who took 5512 with you will recognize it. Consequence: no model can ever predict above 5, and every expensive neighborhood is squashed together.
2. **`AveRooms` max ≈ 141.9 and `AveOccup` max ≈ 1,243.** A household does not have 141 rooms or 1,243 people. These are block groups with very few households — a divide-by-almost-nothing artifact. Show the sorted tail, then show it is only a handful of rows.

- The judgment call: do we delete them? Say honestly *"it depends"* and give the rule — **understand the mechanism first, then decide.** Do not let them think deleting outliers is automatic.
- Then plots, in order: univariate (histograms, boxplots) → bivariate (scatter, correlation) → geographic. Build up.
- Correlation heatmap: `MedInc` is strongest at ~0.69. Say it: *"income predicts home value — that is not surprising, and a model that did NOT find this would be broken."*
- `AveRooms` and `AveBedrms` are ~0.85 correlated. Name it: **multicollinearity.** Trees do not care much; linear regression does. This pays off in notebook 2.
- The lat/lon scatter is the money shot. Coast is expensive, Central Valley is cheap, and San Francisco and Los Angeles appear out of nothing but two numeric columns. **Let it land.**
- CLOSE by tying to deep learning: everything we just did by eye, a network does by weight — but the network cannot tell you the target was censored. Only you can.

---

## Video 5 — Regression Part 1: the skeleton, split, scale, and leakage *(~8 min)*
`2_AllTheModels_Regression.ipynb`

- OPEN with the promise: *"by the end you will have fit five models, and the code for the fifth will look identical to the code for the first."* That predictability **is** the skill.
- Rebuild the data quickly — do not re-teach EDA. Just load and apply the cleaning already justified in notebook 1.
- `X` and `y`. Say it plainly: `y` is what we predict, `X` is everything else. Show the shapes and make them say out loud why `X` lost a column.
- `train_test_split`: 80/20, `shuffle=True`, `random_state`. **Spend time on `random_state`.** Without it your numbers change every run and you cannot tell a real improvement from noise. It is about reproducibility, not about "the right answer."
- **Data leakage is the centerpiece of this video.** Draw it out:
  - `fit_transform` on **train** — the scaler learns min and max from training data only
  - `transform` on **test** — test data gets the training scaler applied to it
- Then say the wrong version out loud: *"if I scale before splitting, the test set's maximum leaks into the training scaler, and my test score is now a lie."* This is the most common mistake in student final projects. Say that sentence.
- Show the check — all train mins are 0, all maxes are 1. Then point out that **test** can land slightly outside 0 to 1. That is correct and expected, not a bug. Students always ask.
- Note we do **not** scale `y` for regression. Mention that in Module 2 we sometimes will, and why.

## Video 6 — Regression Part 2: five models and reading the results *(~11 min)*
`2_AllTheModels_Regression.ipynb`

- OPEN by pointing at the code: instantiate, fit, predict. Three lines. Now do it five times.
- Fit them one at a time on screen so they see the pattern, **then** show the loop that does all five. The loop is the payoff — *"this is what it looks like when you stop copy-pasting."*
- Metrics and what each is **for**:
  - **MAE** — average miss, in the target's own units. Most interpretable. Lead with it.
  - **RMSE** — same units, punishes big misses harder. Use when large errors are expensive.
  - **R²** — proportion of variance explained. Unitless, so it travels between problems.
- Say the MAE out loud in dollars: *"0.31 means we are off by about thirty-one thousand dollars on a typical block group."* Making the number **concrete** is the whole job.
- **Train vs test is the second big idea.** Show the decision tree: near-perfect on train, mediocre on test. That gap **is** overfitting, made visible. Then random forest — same family, much smaller gap. Averaging many trees is what closes it.
- Warn about reading train R². Students report it in projects every semester. Do not.
- The predicted-vs-actual plot: a good model hugs the 45° line. Point at the horizontal stripe of points at 5.0 — **that is the censoring from notebook 1**, showing up in the residuals. Beautiful payoff. No model can beat it, and now they can see why.
- Feature importance: `MedInc` dominates, then the geography columns. Ties straight back to the EDA map.
- **Close with the bridge to Module 2:** *"next module the model object becomes `Sequential([Dense(...)])`. Everything above and below that line stays exactly the same."* Show them the skeleton one more time.

---

## Video 7 — Classification: a different question, the same skeleton *(~11 min)*
`3_AllTheModels_Classification.ipynb`

- OPEN by putting notebook 2 side by side with this one on screen. Say: *"watch how little changes."* The whole pedagogical point is the **sameness**.
- The recode: `np.where(y > median, 1, 0)`. One line. Show `value_counts()` and point out we get a balanced 50/50 split **by construction** because we split at the median. Then immediately say real problems are almost never balanced — fraud is 0.1%, churn is 5% — and warn that accuracy is a terrible metric when classes are imbalanced. Set up the accuracy paradox now, pay it off at the confusion matrix.
- Speed through split + scale. They just saw it. Say *"identical to last notebook"* and move.
- Fit the same five model families, now the `Classifier` versions. Point at the import line — literally `Regressor` → `Classifier`.
- **The confusion matrix is the heart of the video.** Go slow. Walk all four quadrants out loud in the language of *this* problem:
  - true positive = expensive block group, we said expensive
  - false positive = cheap block group, we said expensive → **a wasted marketing dollar**
  - false negative = expensive block group, we said cheap → **a missed opportunity**
- Give the medical-test analogy too — it always lands.
- **Precision vs recall:** precision = *"when I said yes, how often was I right."* Recall = *"of all the real yeses, how many did I catch."* Then the punchline: you can always trade one for the other by moving the threshold, so **ask which error costs more** before you optimize anything.
- **`predict_proba` is the bridge to Module 2 — say this explicitly and slowly.** Show the raw probabilities. Show that `predict()` is just `predict_proba() > 0.5`. Then: *"in two weeks that probability comes out of a **sigmoid** at the end of a neural network, and the decision you make with it is exactly the same."* **This is the single most important sentence in Module 1.**
- Show the threshold sweep. Watch precision and recall trade off as the threshold moves. Land the point that 0.5 is a **default**, not a law.
- ROC/AUC briefly: AUC summarizes performance across **all** thresholds, so it is threshold-free. 0.5 is coin-flipping, 1.0 is perfect.
- **Close Module 1:** *"you now own the skeleton for regression and classification. Module 2 swaps in a neural network and changes nothing else. Go do Assignment 1."*

---

## Video 8 — Assignment 1 walkthrough *(~6 min)*
`Assignment1_OPIM5509.ipynb`

- Read the questions on screen, **do not solve them**. Set expectations about what a good answer looks like.
- Show the data loading cell running so nobody emails about `gdown` — it is two raw GitHub URLs now, no Drive, no downloads.
- Point out the traps **without** solving them, because these are exactly where the emails come from:
  1. Missing values in **every** numeric column, at rates from 2% to 11%.
  2. `Company Size (normalized)` arrives as an **object** dtype — it is text. The sentinel is the string `"N/A "` **with a trailing space**, so `pd.read_csv` does not flag it as missing and `.isnull()` reports zero. They have to look at `.dtypes` and actually investigate. This is the best teaching moment in the assignment — a column that *looks* complete and is not.
  3. There are a few duplicated rows.
  4. The target is only **weakly predictable on purpose** — about 0.50 test R². Their actual-vs-predicted plot will be a fat cloud along the 45° line, not a tight diagonal. Say this out loud before they panic and email: *"if your test R² is around 0.5, you did it right."* A student reporting 0.99 has leaked something.
  5. Only 6 of the 10 columns carry any signal. Education, Age, Work Experience, and Company Size are pure noise. Q9 is where they should discover that — a 5-feature model should hold up almost as well as a 9-feature one.
  6. Random forest will roughly **tie or slightly lose** to plain linear regression here (RF train R² ≈ 0.93, test ≈ 0.47; LR test ≈ 0.52). Not a bug — the best lesson in the assignment. The fancier model is not automatically better, and that train/test gap is textbook overfitting. Let them find it; Q9 is the place to discuss it.
  7. Q5 says drop `State Score`. It genuinely drives the target, so dropping it costs a little accuracy. That is the point — sometimes you are told to drop a feature for reasons outside the model.
- Remind them: `random_state` = **their student ID**, so no two students have the same split. Say it twice.
- Emphasize Q10 — the five bullets. Quantitative, tells a story. Most points are lost here every semester.
- **Close Module 1.** Recap the skeleton one final time: read, split, scale on train, fit, evaluate. Then the hook for Module 2: *"next module we replace the model line with a dense neural network, we compute forward propagation by hand on paper, and you will finally see what `.fit()` has been doing."*

---

## Recording checklist

- [ ] Restart-and-run-all each notebook before recording so outputs are fresh and cell numbers are clean
- [ ] Zoom the browser to ~125% — the plots and `describe()` tables need to be readable on a phone
- [ ] The 🔴 marker cells must render as a **bare red dot**. If a video number or any prose is visible, the talking points leaked outside the HTML comment — fix before recording
- [ ] Videos 4 and 6 run long. If either passes 13 minutes, split it rather than rushing the landmines
- [ ] Order captions in Kaltura after upload
