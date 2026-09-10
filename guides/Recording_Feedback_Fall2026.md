# Fall 2026 video review — what's working, what would make it land better for first-time grad students

**OPIM 5509 · Dr. Dave Wanik · reviewed from the 50 Fall 2026 (IDL) transcripts — Modules 1–3, 5 h 12 m, ~52,000 words**

## The numbers first

| | M1 (9 videos) | M2 (24) | M3 (17) |
| :-- | :-- | :-- | :-- |
| Runtime | 51 min | 2:31 | 1:50 |
| Pace (words/min) | 169 | 164 | 166 |
| "you know" per 1,000 words | 5.9 | 5.1 | 5.2 |
| "um / uh" per 1,000 | 1.4 | 1.8 | 0.5 |
| "kind of / sort of" per 1,000 | 2.0 | 1.8 | 2.4 |
| On-camera waits for code to finish | 0 | 0 | 6 |
| Self-corrections | 2 | 4 | 7 |

Read: the pace is right (160–170 wpm is conversational; the by-hand math videos correctly slow to ~145–150, the evaluation videos run 177–186 and are the ones to ease off). Filler is low for unscripted teaching — "you know" lands about once every 200 words, which is warmth, not a problem. The two things that *do* cost a first-time learner are below: cold opens with no goal, and numbers said differently than they appear on screen.

## What is already excellent — keep doing it

- **Everything is anchored to a number the student can reproduce.** 76.5 → 44.9 → 196; 78 → 140 → 32,988; 15 / 1,426 / 4,320; MAE 1.77 vs 1.76 vs persistence 2.02. That is the single best habit in these videos — it turns "watch" into "check."
- **You keep bridging back.** "Only the model line changes." "The sigmoid probability *is* predict_proba." "Conv1D is the convolution from Module 3, in one dimension." First-time learners need the scaffolding; you build it out loud.
- **You deflate the magic.** "A nonlinear weighted sum." "It's just four networks fit at once." "Neural networks aren't smart." This is the antidote to the hype students arrive with.
- **Honest results.** The stock video that barely beats 50% ("don't trade on it"), Fashion MNIST "absolutely destroying" shirts, the cat on the doghouse. Those are the moments students remember, because they're the ones a textbook never shows.
- **Beat the dumb model.** The 62% Titanic baseline, the 10% MNIST baseline, persistence on time series. Say it in every module; it's the most transferable professional habit in the course.
- **The analogies are gold.** Pasta maker, domino pooling, bank tellers for dropout, Plinko for a frozen base, deck of cards for the 3-D tensor, grandfather clock for divergence. Keep every one — and see item 5 below about landing the formal term right after.
- **Length discipline.** Almost everything is 4–8 minutes; the ≤8 rule is working.

## The seven changes with the biggest payoff (in priority order)

### 1. Bookend every video: a 10-second goal up top, a 20-second recap at the end
Most videos open mid-stream ("Moving right along", "Hey, now our augmented model fit") and end with "I'll stop here for now." A student watching the *first time* has no frame for what to listen for and no consolidation at the end. The fix is cheap and mechanical — put it in the 🔴 marker:
- **Open:** "By the end of this video you'll be able to ___ . The one number to watch is ___ ."
- **Close:** "Three things to remember: ___, ___, ___. Next video: ___."
M3.1 video 4 ("Where we are going") and M2 video 21's three-point summary are the models — do that everywhere.

### 2. Never wait on training on camera
Six moments in Module 3 are "is it done? … looks like it's just about wrapping up … I'll pause and let it finish." In an 8-minute video that's dead air, and it teaches the wrong lesson (that the interesting part is the progress bar). Pre-run every fit before you hit record (the notebooks now ship with stored outputs, so Run-all-then-record works), point at the finished curve, and *say* the cost: "this took 14 minutes on a T4." If you want the live feel, run the tiny model live and show the big one pre-baked.

### 3. Say numbers the way they appear on screen
The spoken slips that recur are all numeric-verbal: "2,640" for 20,640; "0 to 250" for 255; "the nine digits" for ten; "5% of the data" for 0.5%; the precision numerator (48 where it's 101). None of them break a lesson, but students transcribe what they *hear*. Rule: have the number on screen (the markers now carry every anchor) and read it off rather than from memory. The new module READMEs and the transcripts' errata sections already catch these for students — surface those errata notes in HuskyCT next to the videos.

### 4. Guard the four words the whole course turns on
- **validation vs. test** — you flag it yourself ("sometimes I mix up validation and test"). Your own definition in M2 video 19 is perfect: *test means never, ever seen by the model.* Say that sentence verbatim every time the word comes up in M4.
- **false positive vs. false negative** — the M3 confusion-matrix fumble ("am I mixing up my confusion matrix… I need another cup of coffee") is charming once, but FP/FN is exactly what a first-timer is trying to nail down. Keep the *recall starts with R → row* mnemonic on screen and read the cell off the matrix.
- **AUC** — M1 video 9 inverts it ("0.95 means your false positives are outrageous"). That's the one line in 50 videos worth a re-record or a pinned correction: AUC measures class separation, full stop.
- **epoch vs. step** — steps-per-epoch is where the ASR hears "epic" and where students blur the two; the M3 video that soft-codes it is exactly right.

### 5. Land the formal term right after each analogy
Idioms and analogies are the strength — *soup to nuts, Plinko, grandfather clock, bank tellers, a candy bar of information* — but a large share of this cohort is not a native English speaker, and each idiom is a small tax. Keep them; just close the loop: "…like Plinko — the convolutional base is **frozen**, `trainable = False`." Same for "the red dots" → "hidden units."

### 6. Add one "predict before you run" moment per video
The best pedagogical beats in the whole set are the ones where you predict the outcome before fitting: the mean-image-per-class that lets students *guess the confusion matrix* (M2 v23), "shirts and pullovers will collide" (M2 v24), "persistence will be brutal to beat" (M3/M4). Do it once per video as a deliberate pause: "Before I run this — will augmentation raise or lower the *training* accuracy? Say it out loud." That's retrieval practice, and it costs ten seconds.

### 7. Put the common-mistake traps where they live, on screen
The recurring student errors are already in the videos but only spoken: `input_shape` is features-only (M2 v12), clear the session *and* redefine the model (M2 v17/v21), fit the scaler on train only (M1 v7), folder names are the labels (M3 v6), the trailing-1 reshape for a univariate RNN (M4). Give each a red **🧨 common mistake** markdown cell in the notebook at the exact spot, so the student who skims the notebook later still hits it.

## Smaller things

- **Reference the book precisely.** "Your book" comes up ~5 times with no chapter. Put "Chollet §5.2" in the notebook header the video runs on, and point at it once.
- **The "I spun this up with Claude in five minutes" asides** are honest and modern — keep them, but pair each with the course standard you already set in M1: *you must still be able to read the output; "I fit a model and have no idea how it fit" is not allowed.* Otherwise a first-timer hears "the tool does it for you."
- **Say the runtime you're on** ("this is a T4; on CPU expect 10×") the first time in each module — it explains why their run looks different.
- **Unseeded fits** (Titanic, cats/dogs) mean students' numbers won't match yours; either seed them or say "yours will land within a few points — what must hold is that it beats 62%." The M2 README says this; the video should too.
- **Production:** record Kaltura Capture at 1280×720 (kills the "CPU exceeded 99%" toast that hit three M3.3 videos), close the other Colab tabs (a background tab's "99%" lit up on camera), keep the webcam feed — students like the two-feed player.
- **Use the metadata you now have.** Each video has a per-video skill sheet (summary + five timestamped chapters) and each section a skills checklist. Link them in HuskyCT beside the embeds, and load the chapter timestamps into Kaltura as chapters — that's what the key_points format was designed for. For a first-timer, "jump to the part where he counts the parameters" is worth more than any polish.

## A per-video template (for the M4 markers onward)

1. **Goal + the number to watch** (10 s)
2. **The idea, with the analogy — then the formal term** (1–2 min)
3. **Do it in the notebook, pre-run** — point at the finished output (3–4 min)
4. **Read the output honestly** — metric *and* picture; beat the baseline (1–2 min)
5. **One "predict before you run"** question, somewhere in 3–4 (10 s)
6. **Three things to remember + what's next** (20 s)

## Module notes

- **M1:** the AUC line (v9 ~7:00) is the only thing I'd pin a correction on; the boxplot whisker ("a standard deviation or two" → 1.5×IQR) is worth a caption note.
- **M2:** the by-hand arc (v1–10) is the strongest teaching in the course — the PEDW mnemonic, error attribution, "learn the entire dataset." The classification videos are where validation/test and FP/FN discipline matters most.
- **M3:** cats/dogs is much stronger this year (three generators, early stopping, `.keras`, report + confusion matrix, Grad-CAM). The six waits-on-training and the confusion-matrix fumble are the only things I'd change; the stock-video-style honesty in the transfer wrap-up ("the story to tell your manager") is exactly the framing grad students need.
