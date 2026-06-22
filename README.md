# TakeMeter — Classifying r/nba Discourse

A four-way text classifier for r/nba comments and post titles. The project compares a **fine-tuned DistilBERT** against a **zero-shot LLM baseline** (Groq `llama-3.3-70b-versatile`) on the same locked test set.

**Research question:** Can a fine-tuned text classifier distinguish between different kinds of r/nba discourse more effectively than a zero-shot large language model?

**Headline result:** On this dataset, no — the zero-shot baseline (66.67%) beat the fine-tuned model (40.0%). The fine-tuned model collapsed to the majority class. Details and analysis below.

---

## 1. Community Choice and Reasoning

I study public discussion from **r/nba**.

r/nba is a good fit for a discourse-classification task because a single thread mixes several distinct communicative styles: evidence-based basketball analysis, debate prompts and predictions, emotional reactions and jokes, and neutral factual updates or questions. These styles matter to participants because they determine whether a comment adds basketball reasoning, starts a discussion, shares information, or mainly expresses fandom. That natural variety gives a classifier meaningful, human-relevant boundaries to learn — and makes the "reasoned argument vs. unsupported take" distinction genuinely hard, which is the interesting part.

---

## 2. Label Taxonomy

The dataset uses four label strings: `analysis`, `unsupported_take`, `reaction_meme`, `information_question`.

### `analysis`
A comment makes a basketball claim and supports it with specific evidence, comparison, or reasoning that explains *why* the evidence supports the claim. A statistic alone is not enough — the text must use the evidence.

1. *"Andrew Wiggins last 7 games: 20/5/2 on 50/39/71 he is currently on a better stretch, still not what you except from your max contract guy but its drifting in a better direction"*
2. *"He finishes at the rim with a higher percentage than Lebron. But thats also probably because they overplay the 3 so his degree of difficulty isnt the same as players like Kyrie and Lebron."*

### `unsupported_take`
A **declarative** opinion, ranking, prediction, or comparison stated without enough reasoning to count as analysis.

1. *"Since '98 but whatever I guess. LeBron is well past Kobe at this point and anyone who denies it is simply delusional."*
2. *"What has Simmons accomplished besides winning ROY after a full year of NBA training? He can tear up the regular season all he wants but teams have already shown they know how to game plan against him in the playoffs when it matters."*

### `reaction_meme`
A comment primarily expressing humor, emotion, fandom, sarcasm, a rant, or casual conversation rather than a substantive claim.

1. *"Lol Raptor fans stop bitching so much. It's so fucking annoying going into a thread and just seeing FUCK THE REF comments"*
2. *"Man the Spurs are giving me hope again. After that Bulls loss I was ready to give up on this season. But we still have the GOAT coach and we have plenty of talent to work with. We just have to play with pride like we have been lately. GSG!"*

### `information_question`
A comment that neutrally reports a fact/stat/news item, **or** asks a question / poses a discussion prompt (including "who would you rather" debates and "what happened to X?" questions) without making a supported evaluative claim of its own.

1. *"[Charania] OKC's Raymond Felton and Dennis Schroder have been suspended one game by the NBA for altercation against the Bulls."*
2. *"Who would you rather have moving forward, Doncic or Ingram? Stats for the 2018-19 season Doncic: 18.4/6.7/4.9 55.8 TS% Ingram: 15.2/4.0/2.2 52.2 TS%"*

**Key precedence rule** (resolves most ties): question/prompt-form posts go to `information_question` even when evaluative; only *declarative* opinions go to `unsupported_take`. `analysis` requires claim + evidence + explanation; everything emotional/social falls to `reaction_meme`.

---

## 3. Data

### Source
Public r/nba post titles and comments from the **2018–19 NBA season** (roughly December 2018 – February 2019), collected with AI assistance and stored in `takemeter_rnba_200_clean.csv` with columns `text,label,notes`.

### Labeling process
1. Read each example fully before labeling.
2. Apply the precedence rules (analysis → information_question → unsupported_take → reaction_meme).
3. Record a note for genuinely ambiguous cases in the `notes` column.
4. Recheck every `analysis` example for claim + evidence + explanation.

AI was used to draft tentative labels for a candidate pool; **every final label was human-reviewed and corrected where needed** (see §9).

### Label distribution (200 examples)

| Label | Count | Share |
|---|---:|---:|
| `information_question` | 82 | 41.0% |
| `unsupported_take` | 68 | 34.0% |
| `reaction_meme` | 34 | 17.0% |
| `analysis` | 16 | 8.0% |
| **Total** | **200** | **100%** |

The plan targeted a 20% minimum per class. The achieved set fell short for `analysis` (8%) and `reaction_meme` (17%) — authentic r/nba sampling is dominated by news/questions and declarative takes, and reasoned analysis is genuinely rare. This imbalance is the root cause of the modeling result below.

### Three difficult-to-label examples

1. **Text:** *"Collin Sexton is shooting 40% from 3 through 31 games But apparently he doesn't know how to play basketball /s. He's been underrated as a rookie this year, and should really get more attention."*
   **Candidates:** `analysis` / `unsupported_take` → **Decision: `unsupported_take`.** It opens with a real stat, but the central act is an evaluation ("underrated," "should get more attention") with no explanation of how 40% supports the claim. An unexplained stat + bold claim is `unsupported_take`.

2. **Text:** *"The Utah Jazz have played more road games up to December 17th than any NBA team since the 1980 Kansas City Kings ... I'm confident the Jazz are much better than our record presently suggests."*
   **Candidates:** `analysis` / `information_question` / `reaction_meme` → **Decision: `analysis`.** Sits between a stat report and an emotional fan post, but the writer uses the schedule stat to support an explicit claim ("better than our record suggests"), crossing the claim + evidence + explanation bar.

3. **Text:** *"James Harden finishes with 47 (15 free throw points)! He is on a one-man mission to bring the rockets back from the depths of hell. He is crazy! Im so sad we traded him."*
   **Candidates:** `information_question` / `reaction_meme` → **Decision: `reaction_meme`.** It reports a real stat line, but the dominant purpose is emotional fandom. Labeling the main communicative purpose, the feeling is the point.

---

## 4. Fine-Tuning Approach

- **Base model:** `distilbert-base-uncased` (Hugging Face), fine-tuned in Google Colab from the project starter notebook.
- **Data split:** 70% train / 15% validation / 15% test from the single labeled CSV. **Test set = 30 examples** (locked before any evaluation).
- **Training setup:** 3 epochs, learning rate `2e-5`, batch size 16.

### Hyperparameter decision
I started with the notebook defaults because the dataset is small (200 rows), and planned to change a setting only for a concrete reason (rising validation loss, a training error, or poor minority-class recall). The run produced exactly such a signal: the model **collapsed to the majority class** (see §6). That is concrete evidence the defaults plus the imbalanced 200-row dataset were insufficient. The indicated fixes — class-weighted loss or minority oversampling, more data, and more epochs — would be the first changes on any retrain, and would be recorded here.

---

## 5. Baseline Description

- **Model:** Groq `llama-3.3-70b-versatile`, evaluated zero-shot on the **same locked 30-example test set**, before reviewing fine-tuned test performance.
- **Prompt:** included (a) the four label definitions, (b) the edge-case/precedence rules, (c) an instruction to output **only one** valid label string, using the exact CSV label names (`analysis`, `unsupported_take`, `reaction_meme`, `information_question`).
- **Collection:** each test text was sent individually; the model's single-label output was parsed and compared to the gold label. Quality gate: if >10% of outputs were unparseable, revise the format instruction and rerun. (Outputs parsed cleanly, so no rerun was needed.)

---

## 6. Evaluation Report

### Overall accuracy

| Model | Accuracy (n=30) |
|---|---:|
| Zero-shot baseline (`llama-3.3-70b-versatile`) | **66.67%** |
| Fine-tuned (`distilbert-base-uncased`) | **40.0%** |
| Improvement (fine-tuned − baseline) | **−26.67 pts** |

### Confusion matrix — fine-tuned model (rows = true, columns = predicted)

| true ＼ pred | analysis | unsupported_take | reaction_meme | information_question | **total** |
|---|---:|---:|---:|---:|---:|
| **analysis** | 0 | 1 | 0 | 2 | 3 |
| **unsupported_take** | 0 | 0 | 0 | 10 | 10 |
| **reaction_meme** | 0 | 0 | 0 | 5 | 5 |
| **information_question** | 0 | 0 | 0 | 12 | 12 |
| **predicted total** | 0 | 1 | 0 | 29 | 30 |

The model predicted `information_question` for **29 of 30** examples (one stray `unsupported_take`). It got all 12 true `information_question` right and **0** on the other three classes — textbook majority-class collapse.

### Per-class metrics — both models (Precision / Recall / F1)

| Label | n | Zero-shot baseline | Fine-tuned DistilBERT |
|---|---:|---|---|
| `analysis` | 3 | 1.00 / 0.67 / 0.80 | 0.00 / 0.00 / 0.00 |
| `unsupported_take` | 10 | 0.62 / 0.80 / 0.70 | 0.00 / 0.00 / 0.00 |
| `reaction_meme` | 5 | 0.38 / 0.60 / 0.46 | 0.00 / 0.00 / 0.00 |
| `information_question` | 12 | 1.00 / 0.58 / 0.74 | 0.41 / 1.00 / 0.59 |
| **Overall accuracy** | 30 | **0.667** | **0.400** |
| **Macro F1** | 30 | **0.68** | **0.15** |

The contrast is stark. The zero-shot baseline earns a non-trivial F1 on **all four** classes (macro F1 0.68), including the rare ones — it correctly recovers 2 of 3 `analysis` and 3 of 5 `reaction_meme`. The fine-tuned model scores **0.00 on three of four classes** (macro F1 0.15); its only non-zero F1 comes from the class it defaulted to.

### Three specific wrong predictions (fine-tuned)

1. **Text:** *"Since '98 but whatever I guess. LeBron is well past Kobe at this point and anyone who denies it is simply delusional."*
   **True:** `unsupported_take` → **Predicted:** `information_question`. A clear declarative opinion, but the model defaulted to the majority class and ignored the evaluative stance.

2. **Text:** *"Lol Raptor fans stop bitching so much. It's so fucking annoying going into a thread and just seeing FUCK THE REF comments"*
   **True:** `reaction_meme` → **Predicted:** `information_question`. Strong emotional/rant cues were not learned — only ~24 reaction examples appeared in training, too few to form the boundary.

3. **Text:** *"Andrew Wiggins last 7 games: 20/5/2 on 50/39/71 ... its drifting in a better direction"*
   **True:** `analysis` → **Predicted:** `information_question`. With only ~11 `analysis` examples in training, the model never separated "stat + explained claim" from "neutral stat report."

All three share one cause: **majority-class collapse driven by the 200-row, heavily imbalanced dataset**, not inconsistent labeling.

### Sample classifications (fine-tuned)

Five test posts run through the fine-tuned model. The confidence is the model's softmax probability for its predicted class. Note how flat these are — hovering around **0.27**, barely above the 0.25 a uniform guess across four classes would give. The model isn't confidently wrong; it's *unconfident and defaulting* to the majority class.

| # | Text (truncated) | True label | Predicted | Confidence | Correct? |
|---|---|---|---|---:|:---:|
| 1 | "[Charania] OKC's Raymond Felton and Dennis Schroder have been suspended one game..." | `information_question` | `information_question` | 0.28 | ✅ |
| 2 | "Since '98 but whatever I guess. LeBron is well past Kobe at this point..." | `unsupported_take` | `information_question` | 0.27 | ❌ |
| 3 | "Lol Raptor fans stop bitching so much..." | `reaction_meme` | `information_question` | 0.26 | ❌ |
| 4 | "Andrew Wiggins last 7 games: 20/5/2 on 50/39/71..." | `analysis` | `information_question` | 0.27 | ❌ |
| 5 | "What has Simmons accomplished besides winning ROY..." | `unsupported_take` | `information_question` | 0.27 | ❌ |

**Correct example explained (#1):** This is a `[Charania]` news report — a neutral factual update with no claim, opinion, or emotion — so `information_question` is genuinely the right label. But the model assigns it only **0.28** confidence, essentially a four-way coin flip. It is right for the wrong reason: it predicts `information_question` for nearly everything, and because that is both the majority class *and* the correct label here, the prediction lands. The near-uniform confidence is the tell that no real boundary was learned — a confident, well-trained model would put this far above 0.27.

---

## 7. Reflection — What the Model Learned vs. What I Intended

**Intended:** a model that distinguishes four kinds of r/nba discourse and, in particular, separates reasoned `analysis` from unsupported opinion and from emotional reactions.

**Actually learned:** essentially the class prior. With 200 rows and 41% of them `information_question`, the cheapest way for the model to lower training loss was to predict `information_question` almost always — which yields 40% test accuracy without learning any boundary. It learned "what's most common," not "what distinguishes the classes." The zero-shot LLM did better precisely because it relies on general language understanding rather than this tiny, skewed training signal. The lesson: with severe imbalance and very little data, fine-tuning a small model can *underperform* a strong zero-shot model, and accuracy alone hides this — per-class F1 (three classes at 0.00) is what exposes it.

---

## 8. Spec Reflection

**One way the spec helped:** The planning doc's locked taxonomy and ordered precedence rules gave a deterministic tie-breaker for the hardest, most common ambiguity — evaluative *questions* vs. declarative *takes*. Committing up front to "question/prompt form → `information_question`, declarative opinion → `unsupported_take`" kept labeling consistent across hundreds of borderline posts that otherwise would have been coin-flips.

**One way implementation diverged, and why:** The spec set a 20% minimum share per class, but the final dataset came in at 8% `analysis` and 17% `reaction_meme`. I chose to keep authentic sampling rather than synthesize or over-collect to hit the floor — reasoned analysis is genuinely scarce on r/nba, and forcing the quota would have meant inventing or cherry-picking unrepresentative examples. I accepted the imbalance as a documented limitation instead. (Label names also diverged from the template's original `reasoned_analysis`/`opinion_or_prompt`/… to the shorter strings actually used in the CSV and training code.) In hindsight the imbalance was the decisive factor in the model's collapse, so the trade-off — authenticity over balance — is the main thing I would revisit.

---

## 9. AI Usage

AI tools were used in three specific, disclosed ways. All AI output was reviewed and, where noted, overridden.

1. **Candidate collection + tentative pre-labeling (annotation assistance — disclosed).** I directed an AI tool to gather candidate r/nba post titles/comments and assign tentative labels for a candidate pool. **I reviewed every final label myself**, changed labels where the AI was wrong, and recorded difficult decisions in the `notes` column. Final labels are human-decided, not AI-decided.

2. **Label stress-testing.** I asked an AI to generate boundary comments sitting between `analysis` and `unsupported_take`. It produced borderline "stat + claim" examples; I reclassified them myself and, in response, **tightened the `analysis` rule** to require an explicit explanation of *how* a stat supports the claim.

3. **Error-pattern review.** I gave the AI the confusion matrix and 30-example results and asked it to identify failure patterns. It proposed "majority-class collapse to `information_question`." I **verified the counts against `evaluation_results.json` and the confusion matrix** (29/30 predictions; three classes at F1 = 0) before accepting and reporting that explanation.

**Annotation assistance disclosure:** Yes — AI generated tentative labels (instance 1). It did not have final say; all labels were human-reviewed and corrected.

---

## Files

| File | Description |
|---|---|
| `takemeter_rnba_200_clean.csv` | 200 labeled r/nba examples (`text,label,notes`) |
| `evaluation_results.json` | Overall accuracies, label map, test-set size |
| `planning.md` | Full project planning / spec document |
| Confusion-matrix image | Fine-tuned model confusion matrix (also reproduced as a table in §6) |
