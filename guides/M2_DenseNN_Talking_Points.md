# M2 Talking Points — Dense Neural Networks

**OPIM 5509 - Introduction to Deep Learning · Dr. Dave Wanik · University of Connecticut**
*Fall 2026 · recording notes — read before you hit record*

Eighteen 🔴 markers across ten notebooks. Full talking points sit inside each marker's HTML comment (double-click the red dot while recording). This file is the running order and the per-video one-liner.

**Target: ≤8 minutes per video.** Your old M2.1 videos ran 9–13 minutes; the splits below cut them at the natural seams. Lead with the motivating example, then work the concrete case — same pattern as the 5641 recordings.

## Running order

| # | Video | Notebook | Old source |
| :-- | :-- | :-- | :-- |
| **M2.1 — Theory, by hand** | | | |
| 1 | Introduction to forward propagation | `1_ForwardPropagation` | ✅ `1_dxxk9va7` 6:51 |
| 2 | Golden rule of dot products | `1_ForwardPropagation` | ✅ `1_4elqzd8d` 7:25 |
| 3 | Trainable parameters in a dense NN | `1_ForwardPropagation` | ✅ `1_s79qdqs7` 7:51 |
| 4 | Hot and cold learning | `2_HotAndCold` | ✅ `1_hhfkfh4x` 5:35 |
| 5 | Direction and amount (derivatives) | `2_HotAndCold` | ✅ `1_6cxt8yaf` 6:45 |
| 6 | Divergence: scale or use a learning rate | `2_HotAndCold` | ✅ `1_glh15yiu` 2:10 |
| 7 | Learning a full dataset with SGD | `3_BackProp_and_ReLU` | ✅ `1_jkk3r4t7` 7:47 |
| 8 | Gradient descent + intro to ReLU | `3_BackProp_and_ReLU` | ✅ `1_wr6nd6l4` 6:18 |
| 9 | One full iteration for a single row | `3_BackProp_and_ReLU` | ✅ `1_kqlg1wdg` 9:04 ⚠️ title typo "(ILD)" |
| 10 | Learn the entire dataset! | `3_BackProp_and_ReLU` | ✅ `1_h1s4lyp1` 3:47 |
| **M2.2 — Regression** | | | |
| 11 | Regression EDA (get_dummies, scale after split) | `CA_Housing_Regression` | ✅ `1_9vlk1176` 5:36 |
| 12 | Sequential API, compile, early stopping | `CA_Housing_Regression` | ✅ `1_39ni0sxe` 8:36 |
| 13 | Fitting and evaluating | `CA_Housing_Regression` | ✅ `1_dng927yr` 6:48 |
| 14 | Dropout (bank tellers) | `CA_Housing_Regression` | ✅ `1_8pwsuvwt` 5:22 |
| 15 | Dropout bake-off + architecture strategies | `CA_Housing_Regression` | ✅ `1_nx5trycu` 5:34 |
| **M2.3 — Classification** | | | |
| 16 | Titanic prep + the ~62% baseline | `0_BinaryClassification_Titanic...` | ✅ `1_mgw1gcs3` 6:33 |
| 17 | Sigmoid, binary_crossentropy, clear-and-redefine | same | ✅ `1_vayyxlwv` 7:52 |
| 18 | Classification metrics and evaluation | same | ✅ `1_ro87cml7` 4:00 |
| 19 | Cross-validation: train/val/test **and** k-fold | `0b_...TrainValTest` + `0c_...KFold` | ✅ `1_35travsg` 7:42 (both notebooks, one video) |
| 20 | Iris: three classes, softmax | `1_MulticlassClassification_Iris...` | ✅ `1_e5xctxtj` 5:18 |
| 21 | Iris: fit + honest evaluation | same | ✅ `1_rqca98zz` 4:37 |
| 22 | MNIST with a dense net | `2_A_FirstLook_atNN_with_MNIST...` | ⬜ to record (pre-flighted, runs clean) |
| 23 | Fashion MNIST: same recipe, harder | `3_Fashion_MNIST...` | ⬜ to record (pre-flighted, runs clean) |
| 24 | IMDB reviews + the Module 2 close | `4_IMDB_Movie_Reviews...` | ⬜ to record (pre-flighted, runs clean) |

**Recorded 2026-08-24/25** — 21 of 24 done, 2:11:31. Entry ids above are live on Kaltura; all 21 are embedded in HuskyCT under *Videos for M2.1/M2.2/M2.3 (new)*. Transcripts and per-video summaries: `opim5509-transcripts/fall2026_idl/module2/`.

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
