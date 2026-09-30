# M5 Talking Points — Deep Learning for Text

**OPIM 5509 - Introduction to Deep Learning · Dr. Dave Wanik · University of Connecticut**
*Fall 2026 · recording notes — read before you hit record (recording Thursday Oct 1, 2026)*

Twelve videos across six notebooks, plus an optional SHAP review notebook (6b, not recorded). The 🔴 markers sit in the notebooks where each video starts; the full talking points live inside each marker's HTML comment (double-click the red dot while recording). This file is the running order, the per-video one-liner, and what changed since 2022.

**Target: ≤8 minutes per video.** The 2022 M5 ran 11 videos, 5:36–17:59. The 14-minute embeddings video and the 18-minute Twitter capstone are split below.

**The line between M5.1 and M5.2 — BAG OF WORDS vs SEQUENCE.** M5.1 throws word order away: count words, weight them with TF-IDF, hand them to models students already know. M5.2 keeps the order, so everything from M4 carries straight over — embeddings turn a sentence into a time series and the recurrent layers, Conv1D and bidirectional come back unchanged. Same jump M4 made from the window method to RNNs; say it in one sentence at the top of video 7.

## Running order

| # | Video | Notebook | Old source (2022) |
| :-- | :-- | :-- | :-- |
| **M5.1 — Text as a bag of words** | | | |
| 1 | Text is data: storm narratives, hail vs flash flood | `EDA_ML_Storms` | `1_ut3hc73l` 8:33 |
| 2 | Cleaning text: lowercase, strip, stop words, tokenize, stem | `EDA_ML_Storms` | `1_e7nbb1em` 10:13 |
| 3 | CountVectorizer and bag of words: unigrams → n-grams | `EDA_ML_Storms` | `1_hvrr4cd5` 9:57 |
| 4 | Bag of words vs TF-IDF with classic ML, and what the forest is looking at | `EDA_ML_Storms` | `1_n02ycznd` 5:36 |
| 5 | The Keras Tokenizer: TF-IDF of the top 10,000 words, 15 storm types | `Tokenizer_FFNN_Storms` | `1_t74xs5wm` 7:33 |
| 6 | A dense network on TF-IDF: baseline, early stopping, macro F1, class weights | `Tokenizer_FFNN_Storms` | `1_ktrofqie` 8:39 |
| 6b | *Optional, not recorded:* **What is the model looking at? SHAP for text** (the forest from video 4 and the network from video 6) — a review notebook for students | `SHAP_for_Text_Storms` | NEW (Fall 2026) |
| **M5.2 — Text as a sequence** | | | |
| 7 | Tokenize, pad, and the Embedding layer (count its parameters) | `6_1` (to the "pre-trained" heading) | `1_r6lvkx2f` 11:16 + first half of `1_8hnc5erp` |
| 8 | Learned vs pretrained: GloVe, frozen | `Dave_s_GLoVE_Example` → `6_1` second half | second half of `1_8hnc5erp` 14:09 |
| 9 | Embeddings into recurrent layers: SimpleRNN → LSTM + GRU on IMDB | `6_2` | `1_jx5s1qn6` 10:14 |
| 10 | The monster: embedding + Conv1D + pooling + bidirectional + stacked | `6_2` | `1_st1lihyd` 9:48 |
| 11 | **Capstone Pt 1:** pull posts from Bluesky, one folder per account (NYT vs ESPN) | `Who_Posted_It_Bluesky` | replaces `1_xjq8byz8` 17:59 (Twitter) |
| 12 | **Capstone Pt 2:** 100 posts vs all of them, macro F1, "what is it really looking at?" | `Who_Posted_It_Bluesky` | NEW (Fall 2026) |
| — | *Exercise: swap in @npr.org* (end of the capstone notebook, five questions) | `Who_Posted_It_Bluesky` | NEW |

## The scoreboard (say it more than once)

Every model in M5 is compared on **test macro F1** — every class counts the same, big or small.

| where | model | test macro F1 |
| :-- | :-- | --: |
| video 4 | bag-of-words tree / forest (hail vs flood, honest split) · TF-IDF tree / forest | 0.993 / 0.995 · 0.988 / 0.997 |
| video 7–8 | *(validation accuracy, no test score)* first-20-words embedding (5,000 val reviews) · GloVe on 150 · learned on 150 (51 val reviews each) | *0.76 · ≤0.53 · ≤0.59* |
| video 6 | always guess Thunderstorm Wind (baseline) | 0.042 |
| video 6 | dense net, 15 storm types (accuracy 0.93!) · with class weights | **0.67** · 0.69 |
| video 9 | SimpleRNN(64), IMDB | 0.789 |
| video 9 | LSTM → GRU, IMDB | 0.812 |
| video 10 | the monster, IMDB | **0.842** |
| video 11 | always guess the Times (baseline) | 0.350 |
| video 12 | monster, 100 posts per account | 0.854 |
| video 12 | monster, every post | **0.938** |

## Per-video one-liners (numbers verified against the stored outputs, 2026-09-28, seeded 5509)

1. **Text is data.** NOAA Storm Events 2019: 61,324 events × 28 columns; EVENT_NARRATIVE (this storm) not EPISODE_NARRATIVE (the whole outbreak). Drop the 13,000 with no narrative → 48,324. Keep types with ≥300 events, pick Hail (3,783) vs Flash Flood (3,851). Hail EDA: 1 direct injury, 0 deaths. DAMAGE_CROPS is *text* ('0.00K', '5.00M'): strip K/M and multiply — median $0, 97.5th pct $100, max $5M; the quantile table beats the histogram.
2. **Cleaning text.** Corpus vs document. Lowercase; strip non-letters (**`regex=True`** — see "what changed"); NLTK stop words: *"trained spotter reported hail up to the size of quarters…"* → *"trained spotter reported hail size quarters along heavy rain standing water yard"*. Top words hail 3,611 · reported 1,955 · size 1,862 · quarter 1,037 — **'hail' is the label hiding in the text**. Word cloud, `word_tokenize`, stemming (trained→train, heavy→heavi). `remove_stopwords` / `stem_words` are functions, reused next video.
3. **Bag of words.** Flood is now ONE function, `clean_text()`, and one line (ten copy-pasted cells became four). Stack → 7,634 rows. **Split first**, 60/20/20 = 4,580 / 1,527 / 1,527, vocabulary from train only. CountVectorizer: 4,984 words, **0.2% non-zero** (47,333 of ~23M cells). uni + bi + tri = 56,439 columns for 4,580 rows.
4. **BoW vs TF-IDF.** Say it: 2022 split X_TRAIN a second time, so the perfect 1.0 was memorized rows. Honest now: tree 0.993, forest 0.995, 8 misses of 1,527. Top words hail, flood, size, water, road. `forest_without()`: drop hail/flood/flash → **0.977** (it switches to size, water, road); drop the next tier → **0.944** (close, report, street, creek) — different vocabularies all the way down. TF-IDF (train, 4,580 × 5,054): tree 0.988, forest 0.997. `feature_importances_` has no direction — that's SHAP, 6b.
5. **Keras Tokenizer.** All 15 types; shuffle, `dropna` → 28,966. Imbalance: Thunderstorm Wind 13,178 … Dust Devil 6. LabelEncoder is alphabetical (Debris Flow = 0). **Split the text first** (20,276 / 8,690), `Tokenizer(num_words=10000)` fit on train — it keeps stop words — `texts_to_matrix(mode='tfidf')`. PCA's first component explains 3.7%; t-SNE on 6,606 rows — look for islands of one color.
6. **Dense net, 15 classes.** Baseline first: always Thunderstorm Wind = accuracy 0.454, **macro F1 0.042**. 1,011,715 params. EarlyStopping (patience 20) on `validation_split=0.2` of train (2022 validated on the test set): 23 epochs, keeps **epoch 3**; the loss curve is right under the fit and shows 20 epochs of overfitting. Not too big: 32/32 drops to 0.47; lr 0.0001 = smoother curve, same score. Test accuracy 0.93, **macro F1 0.67** — Dust Devil, Marine Hail, Marine High Wind, Marine Strong Wind all 0.00. argmax on probabilities (2022 rounded first → unsure rows became Debris Flow). Heatmap: 176 flash floods called Flood. Class weights: macro 0.67 → 0.69, accuracy 0.925 → 0.904 — the gain is two classes with 9–10 test cases, every big class slips; they can't create signal. **Merge** types under 50 narratives into 'Other (rare)' (15 → 12 classes): macro F1 0.85, but mostly because four 0.00 classes left the average — 'Other' finds 3 of 26. **Try it (last cell):** Keras can't drop stop words, so remove them before the Tokenizer — macro F1 0.67 → 0.66, accuracy 0.925 → 0.920. TF-IDF already does that job, and NLTK's list includes 'down' ('knocked down trees' → 'knocked trees'). Test your cleaning steps.
6b. **SHAP (new).** `feature_importances_` ranked words; SHAP gives each a **direction**. Base value = the prediction with every word hidden (0.80 for Hail); pushes add up to the prediction (0.98). `shap.Explainer(function, shap.maskers.Text(...))` works for the forest AND the network. Forest mistake: *"Heavy rainfall was also observed."* labelled Hail — no hail words, P 0.45. Global (50 narratives): flooding −4.2, water −3.7 vs hail +2.7. Network: 'flooded basements… small lakes' → Flood 0.91; Flood words river, stage, crested, melting; 'flash' pushes away; 'in', 'of' carry signal because the Tokenizer keeps stop words. (SHAP's TreeExplainer failed its own additivity check on this sklearn/NumPy stack, so the notebook deliberately uses the Text masker for both models.)
7. **Embedding layer.** Open on the line: M5.1 threw order away, M5.2 keeps it. One-hot (10,000 mostly-zero columns) vs an embedding (8 learned numbers per word; king − man + woman ≈ queen). `Embedding(1000, 64)` is a lookup table = 64,000 params. IMDB top 10k words, `pad_sequences` to the first **20** words. `Input(20)` → Embedding(10000, 8) 80,000 → Flatten 160 → Dense 161 = **80,161**. Validation peaks **0.76** (epoch 7), best val loss 0.498 at epoch 5. Flatten treats positions separately ("this movie is shit" vs "… the shit") — recurrent layers fix that in video 9.
8. **GloVe.** Tiny docs first: `one_hot(d, 50)` is a **hash** — 'excellent', 'weak' and 'poor' all land on 35, so the learned model is capped at **90%** (same input, opposite labels). Embedding(50, 8) = 400 params, total 433; `get_weights()` is a 50 × 8 lookup table. Borrow GloVe: real Tokenizer, 15 rows × 100 dims from 400,000 GloVe words, `trainable=False` → 1,500 frozen, 401 trainable → **100%**. Then 6_1's second half: raw IMDB from Stanford, 87,393 unique tokens → keep 10k, pad to 300; **150 train / 51 validation on purpose**. `summary()` says 1,960,065 trainable because it prints *before* the freeze — 1,000,000 are frozen after. 300 epochs: train 1.0, val loss best 0.98 (epoch 6) → 1.94; val acc 0.45–0.53. Learned-from-scratch peaks 0.59. **Don't call a winner** — with 51 validation reviews one review is 2 points; both are coin flips. The fix is more data (video 12 proves it).
9. **Embeddings + RNNs.** 100 words × 32 numbers *is* a 100-step time series with 32 features. Count: 320,000 + 2,080 = 322,080; `return_sequences` SimpleRNN(50) 4,150; four stacked 366,000. IMDB 25,000 / 25,000, top 10k words, cut at 100. SimpleRNN(64) 326,273 params, val acc bounces 0.71–0.81 → **0.789**. LSTM → GRU 1,338,849 params → **0.812**. 96% of the parameters are the embedding.
10. **Monster.** Shapes: 100 × 128 → Conv1D(64, 3) 98 × 64 → pool 49 × 64 → bidirectional 49 × 60 → GRU 20 → 1. **1,332,381 params — fewer than LSTM → GRU, and it wins: 0.842.** Best val loss 0.345 at epoch 4 of 10. This is the capstone's `build_monster()`.
11. **Capstone Pt 1.** Bluesky public API (no login): 100 per call + cursor, skip replies and reposts. ESPN has only 857 original posts ever; NYT 1,000 in ~3 weeks. Freeze the Sept 28 snapshot so everyone matches. Folder per account = labels alphabetical (espn.com = 0). 95% of posts ≤ 47 words → maxlen 50. Split first, tokenizer on train: 8,707 words. Baseline accuracy .538 but **macro F1 .350**.
12. **Capstone Pt 2.** Same monster, same test set: 100 posts per account **.854** (54 wrong of 372) → every post **.938** (23 wrong). Who wrote this — then the Senate sentence → ESPN at P = 0.00. **The shortcut**: ESPN posts are shorter and padding goes in front, so the zero count leaks length — Times posts cut to 10 words are recognized 35% of the time vs 92%; add one sentence and it flips to 0.99. Hand off to NPR.

## What changed since 2022 (things that look different on screen — say the ones in bold)

- **Seeds.** Every Keras notebook seeds at the top (`keras.utils.set_random_seed(5509)` + op determinism) and re-seeds right before each model build — the M4 convention. The trees use `random_state=5509`. The numbers above are what students get.
- **Loss curve after every fit**, inline, with the epoch count and best validation loss printed (6_1 first model, 6_2 monster, both GloVe fits were missing one).
- **Macro F1 next to accuracy** in EDA_ML_Storms and Tokenizer_FFNN, a **majority-class baseline** in Tokenizer_FFNN, and a test-set scoreboard cell after each of 6_2's three IMDB models.
- **Two 2022 bugs fixed (9/29), both of which flattered the models:** EDA_ML_Storms split X_TRAIN into test/validation (the 1.0 was memorized rows), and Tokenizer_FFNN rounded probabilities before argmax (unsure rows became Debris Flow). **Split first everywhere**, vocabularies fit on train only; Tokenizer_FFNN validates on 20% of train with **EarlyStopping**, never on the test set.
- **New cells:** `forest_without()` and the top-words chart (video 4), `clean_text()` for flood (video 3), class weights run for real + a labelled confusion heatmap (video 6), hail injury/death EDA collapsed to one cell. **New notebook:** `SHAP_for_Text_Storms` (6b).
- **The strip-punctuation line in EDA_ML_Storms now has `regex=True`.** Since pandas 2.0 `str.replace` is literal by default — the 2022 line silently removed nothing. Also `[^a-zA-Z ]` instead of `[^A-z ]`.
- **Keras 3:** `Input(shape=...)` layers replace `input_length=` / `input_shape=` (without them `summary()` shows no parameter counts and GloVe's `set_weights` crashes); the Tokenizer imports from `tensorflow.keras.preprocessing.text`.
- **No lambdas or list comprehensions:** `remove_stopwords()` and `stem_words()` in EDA_ML_Storms, a loop for `one_hot` in the GloVe example, `.apply(nltk.word_tokenize)`.
- t-SNE uses `max_iter` (scikit-learn removed `n_iter`); 6_1 reads the IMDB files in `sorted()` order so Colab picks the same 150 training reviews.
- **The Twitter capstone is archived** (`Module5/_archive_twitter/`): snscrape died when X closed scraping in 2023. Bluesky replaces it.

## Through-lines

- **Bag vs sequence.** Videos 1–6 destroy order on purpose; 7–12 keep it. The monster in video 10 is literally M4's tools reading words.
- **Macro F1, every time.** Accuracy 0.93 hid a 0.67 in video 6 (baseline: 0.454 hid 0.042); accuracy .538 hid .350 in video 11.
- **Ask what the model is looking at.** Top words + `forest_without()` (video 4), SHAP (6b), the zero-padding length shortcut (video 12). A perfect score is a question, not an answer.
- **More data beats a fancier model.** 100 → all posts: .854 → .938 with the same network (video 12).

## Guardrails (slips to not make twice)

- Split first, always: every M5 notebook now fits its vocabulary on train only — say it once in video 3 and once in video 5.
- Don't call video 4's 0.99 a great model — it's easy data (different vocabularies), and the old 1.0 was a split bug.
- `one_hot` is a hash, not a vocabulary — collisions are real (video 8).
- In 6_1 the pretrained-GloVe model trains on only 150 reviews — that's the point (small data), not a mistake.
- On a T4 a recurrent kernel can drift a hair from the stored CPU numbers — say "within a few hundredths", don't promise an exact match.

## Recording checklist

- [ ] Restart-and-run-all each notebook in Colab before recording (all seven verified locally on TF 2.21 / Keras 3.15 / pandas 3 / shap 0.48 on 2026-09-29)
- [ ] SHAP notebook: the 50-narrative and 40-narrative cells take about a minute each — run them before you hit record
- [ ] 🔴 cells must render as a bare red dot — anything visible means the comment leaked
- [ ] ≤8 min; split rather than rush
- [ ] Tag uploads `(IDL)` — captions auto-generate
- [ ] **Record Kaltura Capture at 1280×720** — kills the "CPU exceeded 99%" toast at the source
- [ ] Close the other Colab tabs — a background tab's "99%" progress lit up on camera in M3.3
- [ ] 6_1 downloads IMDB (~84 MB) and GloVe (~820 MB zip) from Stanford — run those cells *before* you hit record
