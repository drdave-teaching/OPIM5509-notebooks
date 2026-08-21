# M2 Talking Points — Dense Neural Networks

**OPIM 5509 - Introduction to Deep Learning · Dr. Dave Wanik · University of Connecticut**
*Fall 2026 · recording notes — read before you hit record*

Eighteen 🔴 markers across ten notebooks. Full talking points sit inside each marker's HTML comment (double-click the red dot while recording). This file is the running order and the per-video one-liner.

**Target: ≤8 minutes per video.** Your old M2.1 videos ran 9–13 minutes; the splits below cut them at the natural seams. Lead with the motivating example, then work the concrete case — same pattern as the 5641 recordings.

## Running order

| # | Video | Notebook | Old source |
| :-- | :-- | :-- | :-- |
| **M2.1 — Theory, by hand** | | | |
| 1 | Forward prop: one weight to dot products | `1_ForwardPropagation` | ForwardProp Pt 1 (12:11 — trim) |
| 2 | Hidden layers, the shape rule, trainable params | `1_ForwardPropagation` (cell 65) | ForwardProp Pt 2 |
| 3 | Hot and cold learning | `2_HotAndCold` | HotCold Pt 1 |
| 4 | Direction and amount + divergence | `2_HotAndCold` (cell 29) | HotCold Pt 2 |
| 5 | Learning a whole dataset (SGD) | `3_BackProp_and_ReLU` | BackProp Pt 1 |
| 6 | ReLU and backprop end-to-end | `3_BackProp_and_ReLU` (cell 44) | BackProp Pt 2 (13:37 — trim) |
| **M2.2 — Regression** | | | |
| 7 | Prep CA housing (get_dummies!) | `CA_Housing_Regression` | First NN Pt 1 |
| 8 | Build, compile, fit, early stopping | `CA_Housing_Regression` (cell 27) | First NN Pt 2–3 |
| 9 | Evaluate: curves, scatter, dollars | `CA_Housing_Regression` (cell 45) | First NN Pt 4 |
| 10 | Dropout (bank tellers) | `CA_Housing_Regression` (cell 54) | First NN Pt 5 |
| 11 | Architecture cheat sheet | `CheatSheet_BuildingFFNNs` | Best practices (6:36) |
| **M2.3 — Classification** | | | |
| 12 | Titanic prep + the 61.7% baseline | `0_BinaryClassification_Titanic...` | Titanic Pt 1 |
| 13 | Sigmoid, binary_crossentropy, confusion matrix | same (cell 22) | Titanic Pt 2 (13:15 — trim) |
| 14 | Iris: three classes, softmax | `1_MulticlassClassification_Iris...` | Iris Pt 1 |
| 15 | Iris: fit + honest evaluation | same (cell 19) | Iris Pt 2 |
| 16 | MNIST with a dense net | `2_A_FirstLook_atNN_with_MNIST...` | DNNs for MNIST (11:00 — split at Evaluate if long) |
| 17 | Fashion MNIST: same recipe, harder | `3_Fashion_MNIST...` | Fashion half of the old combo |
| 18 | IMDB reviews + the Module 2 close | `4_IMDB_Movie_Reviews...` | IMDB half of the old combo |

Optional whiteboard capstone after video 6: draw a 3→4→1 net, count 21 params, trace one forward+backward pass (the old *PuttingItAllTogether*, 10:44). No notebook — record it only if the energy is there.

## The through-lines to keep hitting

- **Bridge back to Module 1 constantly.** The skeleton is unchanged; only the model line grew. The sigmoid probability IS `predict_proba`. The $500k censoring reappears in video 9's scatter. Scaling habits pay off in video 4's divergence demo.
- **Shape is the language.** Videos 2, 5, 16, 17 all count parameters — it's the same muscle every time, and Assignment 2 grades it.
- **Deflate the magic.** "A network is a nonlinear weighted sum" — you said it in the M1 welcome; M2.1 is where you prove it with a pencil.

## Guardrails (slips to not make twice)

- **AUC measures class separation.** M1 video 9 said high AUC "means your false positives are outrageous" — inverted. If AUC comes up in video 13, it's *good separation*, full stop.
- Boxplot whiskers are **1.5×IQR**, not "a standard deviation or two."
- Iris is **categorical_crossentropy**, Titanic is **binary_crossentropy** — say the difference out loud in both videos.

## Recording checklist

- [ ] Restart-and-run-all each notebook in Colab before recording (needs TF — can't be verified locally)
- [ ] 🔴 cells must render as a bare red dot — anything visible means the comment leaked
- [ ] ≤8 min; split rather than rush
- [ ] Tag uploads `(IDL)` — captions auto-generate, no ordering needed
- [ ] Kaltura Capture at 1280×720 kills the CPU toast at the source
