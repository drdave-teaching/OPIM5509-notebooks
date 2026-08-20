# Module 1 — Machine Learning Refresher

**OPIM 5509: Introduction to Deep Learning · Dr. Dave Wanik · University of Connecticut**
*Fall 2026 · Week 1*

Before we build a single neural network, we rebuild the machine learning workflow every deep learning model in this course sits on top of:

```
  read data  ->  split  ->  scale on train  ->  [ MODEL ]  ->  fit / predict  ->  evaluate
```

Only the bracketed step changes for the rest of the semester.

---

## Work through them in this order

| # | Notebook | What it covers |
| --- | --- | --- |
| 1 | [`0_Welcome_and_Setup.ipynb`](0_Welcome_and_Setup.ipynb) | Colab, GitHub, library versions, where everything lives |
| 2 | [`1_CaliforniaHousing_EDA.ipynb`](1_CaliforniaHousing_EDA.ipynb) | EDA, censored targets, impossible averages, multicollinearity, maps |
| 3 | [`2_AllTheModels_Regression.ipynb`](2_AllTheModels_Regression.ipynb) | The skeleton, data leakage, five models, MAE/RMSE/R², overfitting |
| 4 | [`3_AllTheModels_Classification.ipynb`](3_AllTheModels_Classification.ipynb) | Confusion matrix, precision/recall, `predict_proba`, ROC/AUC |
| 5 | [`4_General_EDA_Template.ipynb`](4_General_EDA_Template.ipynb) | A blank worksheet to fill in from memory |
| 6 | [`Assignment1_OPIM5509.ipynb`](Assignment1_OPIM5509.ipynb) | Your turn, on messy data you have not seen |

Companion guides live in [`../guides/`](../guides/): the [video guide](../guides/M1_MLRefresher_Video_Guide.md) and the [skills sheet](../guides/M1_MLRefresher_Skills.md).

## Data

| File | Used by |
| --- | --- |
| California Housing | Loaded in one line from `sklearn.datasets.fetch_california_housing` — no download |
| [`data/X_F26.csv`](data/X_F26.csv) · [`data/y_F26.csv`](data/y_F26.csv) | Assignment 1 |

## A note on Boston Housing

Module 1 used to run on the Boston Housing dataset. scikit-learn **removed** it in version 1.2 because one of its columns was an explicit racial proxy, so a current install cannot fetch it at all. We now use California Housing, which loads just as easily and carries two useful teaching problems of its own — a top-coded target and a set of impossible averages.

The retired notebooks are preserved in [`_archive_boston/`](_archive_boston/) for reference. They are not part of the course.
