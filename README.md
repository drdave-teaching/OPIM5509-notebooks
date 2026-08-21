<img src="https://raw.githubusercontent.com/drdave-teaching/OPIM5509-notebooks/main/_banners/opim5509_banner.svg" width="100%" alt="OPIM 5509 banner"/>

# OPIM 5509 — Introduction to Deep Learning

**Course notebooks · Dr. Dave Wanik · University of Connecticut**

Every notebook here is standardized and ready to run: a banner, a title, an **Open in Colab** badge. Click the badge, then **Runtime → Run all**. Data loads from stable public sources — no Drive mounting, no uploading CSVs by hand.

---

## 👉 Start here — Week 1

### **[Module 1 — Machine Learning Refresher](Module1/)**

Six notebooks that rebuild the ML workflow every neural network in this course sits on top of. **[Open the Module 1 page →](Module1/)**

| Notebook | Open |
| :-- | :-- |
| Welcome & Setup | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module1/0_Welcome_and_Setup.ipynb) |
| California Housing EDA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module1/1_CaliforniaHousing_EDA.ipynb) |
| All the Models — Regression | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module1/2_AllTheModels_Regression.ipynb) |
| All the Models — Classification | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module1/3_AllTheModels_Classification.ipynb) |
| General EDA Template | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module1/4_General_EDA_Template.ipynb) |
| Assignment 1 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module1/Assignment1_OPIM5509.ipynb) |
| Appendix — ROC, AUC & Thresholds *(optional)* | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module1/5_Appendix_ROC_AUC_and_Thresholds.ipynb) |

---

## All modules

| Module | Topic | Notebooks |
| :-- | :-- | :-- |
| **[1](Module1/)** | **Machine Learning Refresher** — EDA, the modeling skeleton, regression & classification | 6 |
| **[2](Module2/)** | **Dense Neural Networks** — by-hand theory, Keras regression & classification | 14 |
| [3](Module3/) | Convolutional Neural Networks — convnets, transfer learning, autoencoders | 7 |
| [4](Module4/) | Recurrent Neural Networks (numeric sequences) — RNN/LSTM/GRU, univariate & multivariate | 11 |
| [5](Module5/) | Recurrent Neural Networks (text) — tokenizers, embeddings, GloVe | 6 |
| [6](Module6/) | Special Topics — image segmentation, deep recommenders | 2 |

## Guides

Recording guides, skills checklists, and video breakdowns live in **[`guides/`](guides/)**.

## The through-line

```
  read data  ->  split  ->  scale on train  ->  [ MODEL ]  ->  fit / predict  ->  evaluate
```

Only the bracketed step changes across the whole semester. Dense networks, convolutional networks, recurrent networks — they all drop into the same slot.

## Related repositories

| Repo | What's in it |
| :-- | :-- |
| [OPIM5509Files](https://github.com/drdave-teaching/OPIM5509Files) | Larger data assets (GloVe, aclImdb, image sets) via Git LFS |
| [opim5509-textbook](https://drdave-teaching.github.io/opim5509-textbook/) | The course ebook — transcripts, chapter prose, lecture key points |

## Setup

```bash
pip install -r requirements.txt
```

Not needed on Colab — everything is preinstalled there.
