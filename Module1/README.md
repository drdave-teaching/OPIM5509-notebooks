<img src="https://raw.githubusercontent.com/drdave-teaching/OPIM5509-notebooks/main/_banners/opim5509_banner.svg" width="100%" alt="OPIM 5509 banner"/>

# Module 1 — Machine Learning Refresher

**Week 1 · Fall 2026 · Dr. Dave Wanik · University of Connecticut**

Before we build a single neural network, we rebuild the machine learning workflow that every deep learning model in this course sits on top of:

```
  read data  ->  split  ->  scale on train  ->  [ MODEL ]  ->  fit / predict  ->  evaluate
```

Only the bracketed step ever changes. In Module 2 it becomes a dense neural network and nothing around it moves.

---

## Start here — click a badge, then Runtime → Run all

| # | Notebook | What it covers | Open |
| :-- | :-- | :-- | :-- |
| **1** | **Welcome & Setup** | Colab, GitHub, library versions, reading a `.shape` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module1/0_Welcome_and_Setup.ipynb) |
| **2** | **California Housing EDA** | Unit of analysis, censored targets, impossible averages, multicollinearity, maps | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module1/1_CaliforniaHousing_EDA.ipynb) |
| **3** | **All the Models — Regression** | The skeleton, data leakage, five models, MAE/RMSE/R², overfitting | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module1/2_AllTheModels_Regression.ipynb) |
| **4** | **All the Models — Classification** | Confusion matrix, precision vs recall, `predict_proba`, ROC/AUC | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module1/3_AllTheModels_Classification.ipynb) |
| **5** | **General EDA Template** | A blank worksheet to fill in from memory | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module1/4_General_EDA_Template.ipynb) |
| **6** | **Assignment 1** | Your turn, on messy data you have not seen | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module1/Assignment1_OPIM5509.ipynb) |

> **Save a copy before you type.** `File → Save a copy in Drive`. The GitHub version is read-only for you, and unsaved edits vanish when the tab closes.

---

## Guides

| Guide | Use it for |
| :-- | :-- |
| [📺 Video Guide](../guides/M1_MLRefresher_Video_Guide.md) | What each of the 8 videos covers, and which notebook it runs on |
| [✅ Skills Sheet](../guides/M1_MLRefresher_Skills.md) | The checklist of what you should own before Module 2 |

## Data

| Source | Used by |
| :-- | :-- |
| California Housing | Loaded in one line via `sklearn.datasets.fetch_california_housing` — no download, no Drive |
| [`data/X_F26.csv`](data/X_F26.csv) · [`data/y_F26.csv`](data/y_F26.csv) | Assignment 1 |

---

## Why not Boston Housing?

Module 1 used to run on the Boston Housing dataset. scikit-learn **removed it in version 1.2** because one of its columns was an explicit racial proxy, so a current install cannot fetch it at all.

California Housing loads just as easily and brings two teaching problems of its own — a target top-coded at \$500,001, and a set of per-household averages that cannot possibly be true. Both get found during EDA and both come back to haunt the residuals in notebook 3, which is exactly the lesson.

The retired notebooks are preserved in [`_archive_boston/`](_archive_boston/) for reference. They are **not** part of the course.
