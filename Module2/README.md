<img src="https://raw.githubusercontent.com/drdave-teaching/OPIM5509-notebooks/main/_banners/opim5509_banner.svg" width="100%" alt="OPIM 5509 banner"/>

# Module 2 — Dense Neural Networks

**Weeks 3–5 · Fall 2026 · Dr. Dave Wanik · University of Connecticut**

Module 1 ended with a promise: only the `[ MODEL ]` line ever changes. Module 2 cashes it in — first **by hand on paper**, so `.fit()` stops being magic, then in Keras for regression and classification.

```
  M2.1  Theory by hand   forward prop → hot & cold → gradient descent → backprop + ReLU
  M2.2  Regression       California housing in Keras, early stopping, dropout, architectures
  M2.3  Classification   sigmoid (Titanic) → softmax (Iris) → images (MNIST, Fashion) → text (IMDB)
```

## M2.1 — Theory, by hand

| # | Notebook | What you'll do | Open |
| :-- | :-- | :-- | :-- |
| 1 | **Forward Propagation** | One weight → dot products → hidden layers; the golden rule of shapes; count trainable parameters | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module2/1_ForwardPropagation.ipynb) |
| 2 | **Hot and Cold Learning** | Nudge-and-check learning, then real gradient descent — and a live divergence demo | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module2/2_HotAndCold.ipynb) |
| 3 | **Backprop and ReLU** | Learn a whole dataset row by row; why linearity stacks to nothing; ReLU and the full backward pass | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module2/3_BackProp_and_ReLU.ipynb) |
| — | *Backprop practice (new data)* | The same machinery on a fresh dataset — practice, no video | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module2/3_BackProp_and_ReLU_newData.ipynb) |

## M2.2 — Regression in Keras

| # | Notebook | What you'll do | Open |
| :-- | :-- | :-- | :-- |
| 4 | **NN Regression for CA Housing** | The raw CSV this time — `get_dummies`, build/compile/fit, early stopping, learning curves, dropout | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module2/CA_Housing_Regression.ipynb) |
| 5 | **Cheat Sheet: Building FFNNs** | Four architecture strategies + the bells and whistles; Optuna tuning as optional enrichment | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module2/CheatSheet_BuildingFFNNs.ipynb) |
| ✍ | **Assignment 2** | Trainable params + NN regression | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module2/Assignment2.ipynb) |

## M2.3 — Classification in Keras

| # | Notebook | What you'll do | Open |
| :-- | :-- | :-- | :-- |
| 6 | **Titanic (binary)** | The 61.7% baseline, sigmoid + `binary_crossentropy`, confusion matrix | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module2/0_BinaryClassification_Titanic_structured_data.ipynb) |
| 7 | **Iris (multiclass)** | Softmax, `categorical_crossentropy`, the shuffle trap, argmax | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module2/1_MulticlassClassification_Iris_structured_data.ipynb) |
| 8 | **MNIST (images)** | Flatten 28×28 → 784, /255, softmax → ~98% — and why off-center digits break it | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module2/2_A_FirstLook_atNN_with_MNIST_images.ipynb) |
| 9 | **Fashion MNIST** | Same recipe, harder problem — ~88% and the reason for the drop | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module2/3_Fashion_MNIST_with_fashion_images.ipynb) |
| 10 | **IMDB reviews (text)** | One-hot word presence, live overfitting, and what order-blindness costs | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module2/4_IMDB_Movie_Reviews_text_data.ipynb) |
| ✍ | **Assignment 3** | Individual NN classification | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drdave-teaching/OPIM5509-notebooks/blob/main/Module2/Assignment3.ipynb) |

**Optional deep dive:** [BatchNorm From Scratch](BatchNorm_FromScratch.ipynb) — what it normalizes and where its parameters live.

## Guides

| Guide | Use it for |
| :-- | :-- |
| [🎙 Talking Points](../guides/M2_DenseNN_Talking_Points.md) | Instructor — the 18-video recording plan with running order |
| [✅ Skills Sheet](../guides/M2_DenseNN_Skills.md) | The checklist of what you should own before Module 3 |

## A note on Boston Housing

The original "first NN regression" notebook ran on Boston Housing. Same story as Module 1: retired, preserved in [`_archive_boston/`](_archive_boston/), not part of the course. The mainline is California — which is also what the lectures actually use.
