# M1 Video Guide — Machine Learning Refresher

**OPIM 5509 - Introduction to Deep Learning · Dr. Dave Wanik · University of Connecticut**
*Fall 2026 · Week 1 · nine videos, 51:33*

Written from the actual recordings, not from a plan. Transcripts and polished scripts live in [`opim5509-transcripts/fall2026_idl/module1`](https://github.com/drdave-teaching/opim5509-transcripts/tree/main/fall2026_idl/module1).

**New this semester:** Module 1 moved off Boston Housing. scikit-learn removed it in version 1.2 because one of its columns was an explicit racial proxy, so a current install cannot fetch it. We use **California Housing** — 20,640 rows, mostly clean, a few good gotchas, and spatial, so you can plot it.

---

## Getting set up

### 1 · Welcome to IDL from Dr. Dave! *(2:55)*
Who Dave is and the answer to the question you're probably holding: **what is deep learning and why do I need it?** Traditional methods like random forests need structured tabular data. Neural networks handle structured *and* unstructured — text, time series, images, video, audio. They look like magic, but at a high level they are "just a nonlinear weighted sum of information that gets transformed and creates an output."

Then the map of the semester: ML refresher → neural network bootcamp → convolutional networks (theory, then classification, then fine-tuning pre-trained weights) → recurrent networks, **time series first** because going to text first would mean learning embeddings and sequences of embeddings at the same time → text sequences last.

### 2 · Google Colaboratory and the UConn Library *(2:42)*
Attaching Colab to your Google Drive — New → More → Connect more apps → search Colab → Install. Dave's suggestion: make a dedicated Gmail for class if you want the clean 20 GB. Then finding the **Chollet** textbook free through library.uconn.edu with a NetID login — full PDF, all the readings.

### 3 · Welcome, Colab and GitHub *(7:03)* · `0_Welcome_and_Setup.ipynb`
The full environment walkthrough. Materials live on GitHub and HuskyCT. **`File → Save a Copy in Drive`** puts your copy in the *Colab Notebooks* folder, where you can move and organize it.

Colab is your computer — Python 3.12, package manager behind the scenes, libraries already at stable versions. GPU is under Runtime → Change runtime type, with the caveat worth writing down: **selecting a GPU is not enough, your code has to actually use it.** That's extra code, covered later.

Also introduces the experimental **ebook** built from Dave's own lectures and transcripts, and closes with a Python sanity check: list indexing from zero (`nums[0:3]` stops before 3), dictionaries as a clean way to hold a network config, and writing a function once instead of pasting the same code ten times.

---

## Exploratory data analysis · `1_CaliforniaHousing_EDA.ipynb`

### 4 · Introduction to EDA on CA Housing *(5:11)*
Why this dataset, then the single most important framing in the module: **a row is a census block, not a house.** Every "average" column is a summary statistic over a geographic unit.

Commit the shape to memory — 20,640 rows, 9 columns — so that when a later join explodes or drops rows, you notice. Then `df.info()` for dtypes and missingness in one call, `.isnull().sum()`, and `.describe()`.

The statistical point worth the whole video: **mean and standard deviation are most meaningful for normally distributed data.** When the standard deviation is larger than the mean you probably have skew, and you should be reading percentiles instead. Dave adds the 1st and 99th because he comes from an engineering background and cares about extremes.

### 5 · Outliers, censored data, univariate and bivariate plots *(7:00)*
The two gotchas, both found **by plotting**.

**The target is right-censored.** `value_counts()` shows the most common value is 5 — that's \$500,000 *or more* — with 965 rows piled on the cap. Anything more expensive got smushed into that bin, so your model will struggle there.

**Average rooms hits 141 and average occupancy hits 1,200.** No household has 141 rooms. It happens because the population in those blocks is tiny and the ratio blows up.

Then histogram matrix, boxplots, and a warning students should tattoo somewhere: **outliers tend to be the most interesting thing in the business.** People delete them to make a model fit better and throw away the signal. Flag variables with `np.where`, group-by to compare expensive vs. cheap blocks, and the **"reasonable person test"** — does the story the data tells agree with what you'd expect? If something is backwards, that's what you raise in the meeting. Closes on the correlation matrix: income is the strongest driver of house value, and average rooms/bedrooms are multicollinear.

### 6 · Geographic EDA and final data cleaning *(2:33)*
Latitude and longitude are spatial coordinates, so **the data assembles itself into the shape of California** with no base map at all. Color by median house value and the coast lights up expensive, inland goes cheap, with something interesting near Lake Tahoe.

Bonus exercise: size the bubbles by population — and **sort so the biggest bubbles draw underneath**, or the small ones vanish. Then the cleaning decision: drop blocks with occupancy > 10 or rooms > 20.

---

## Modeling · `2_AllTheModels_Regression.ipynb`

### 7 · Intro to end-to-end ML for regression *(5:50)*
The imports Dave reaches for every time, then the workflow. Split X from Y and notice **`df` is never mutated** — no `inplace=True` anywhere. `train_test_split` with shuffle and a random seed, because if your rows have structure (newest to oldest, say) chopping off the last 20% gives you a biased sample.

Two things students miss. **Keep your feature names before you scale** — scaling returns a NumPy array and the names are gone, but the column positions survive, so save them and you can label permutation importance later. Otherwise it's X1, X2, X3 "and that's hard to explain in a meeting."

And **scaling and leakage**: fit the scaler on train, transform test. Not `fit_transform` on both, and definitely not `fit_transform` on the whole X before splitting — that leaks the distribution, including outliers, into the model. Also: the test partition won't span the full 0–1 range, and that is fine.

### 8 · Fitting and evaluating ML models for regression *(9:40)*
One model by hand first — LinearRegression, MAE 0.5, R² 0.63, "a moderate fit" — then an `evaluate` function that loops five models into a summary table.

The decision tree overfits "like crazy" because scikit-learn will happily put a single observation in a leaf; `min_samples_leaf` is the dial. **Random forest wins out of the gate**, and Dave explains *why* it wins on tabular data: many small trees, each on a random subset of rows and columns, averaged into a stable estimate. Where neural networks win instead is unstructured data, because they do the feature engineering for you.

Predicted-vs-actual shows the forest hugging the 45° line while linear regression is a cloud — and **both fail on the censored data**, which is the EDA finding reappearing in the residuals.

Closes with the strongest argument in Module 1: **permutation importance over Gini importance.** Built-in tree importance is biased toward high-cardinality features, because a column with many unique values simply offers more places to split. Permutation importance shuffles one column on a copy of the test set and measures how far the score falls — model-agnostic, so you can compare a tree to a neural network fairly. *"No longer are you allowed to say, I fit a model, I have no idea how it fit."*

---

## Classification · `3_AllTheModels_Classification.ipynb`

### 9 · End-to-end ML for classification *(8:39)*
The same dataset recycled with a twist, deliberately, so the technique stands out instead of the data. Recode to a flag at the median, and **drop both the flag and the raw house value from X** — leaving the raw value in hands the model the answer. Classes come out about 50/50.

Same skeleton, different algorithms — logistic regression instead of linear, classifier versions of the trees, KNN — and different metrics, because mean absolute error means nothing here.

The key idea: **`predict_proba` gives you the raw score.** A model predicting 0.99 is confident; one predicting 0.53 is not, and those scores can be manipulated for a better fit. **Accuracy only applies when your classes are balanced** — precision and recall are the better tools. Confusion matrix in seaborn, the recall-starts-with-R mnemonic (recall = TP / (TP + FN), precision is the column), and a real example from Dave's IoT analytics work where a "chatty" model with false alarms was the right call because catching every true event mattered more.

Ends by pointing you at threshold manipulation on your own — which is exactly what the optional appendix picks up.

---

## Optional appendix — no video

**`5_Appendix_ROC_AUC_and_Thresholds.ipynb`** builds an ROC curve **by hand** from twelve loan applicants you can check with a pencil, watches the false positive rate move as the threshold slides, computes AUC two ways, and shows how the cost of each mistake picks the threshold for you. About twenty minutes, and it makes video 9's closing suggestion concrete.

---

## The through-line

```
  read data  ->  split  ->  scale on train  ->  [ MODEL ]  ->  fit / predict  ->  evaluate
```

Only the bracketed step changes for the rest of the semester.

---

*Companion: [Skills sheet](M1_MLRefresher_Skills.md). Next up: Module 2 — Dense Neural Networks.*
