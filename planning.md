# TakeMeter — Project Planning Document

> **Project status:** Completed. Labels finalized, model trained, and both models evaluated on the locked test set. Sections below reflect the dataset and results actually produced (`takemeter_rnba_200_clean.csv`, `evaluation_results.json`).

## 1. Project Overview

### Community
I will study public discussion from **r/nba**.

### Why this community is a good fit
r/nba is a useful setting for a discourse-classification task because it contains a mix of evidence-based basketball analysis, debate prompts and predictions, emotional reactions, jokes, factual updates, and questions. These styles matter to participants because they affect whether a comment adds basketball reasoning, starts a discussion, shares information, or mainly expresses fandom.

### Research question
Can a fine-tuned text classifier distinguish between different kinds of r/nba discourse more effectively than a zero-shot large language model?

---

## 2. Label Taxonomy

> The dataset uses four label strings: `analysis`, `unsupported_take`, `reaction_meme`, and `information_question`.

### Label 1: `analysis`
**Definition:** A comment makes a basketball claim and supports it with specific evidence, comparison, or basketball reasoning that explains why the evidence supports the claim.

**Clear examples:**
1. "Andrew Wiggins last 7 games: 20/5/2 on 50/39/71 he is currently on a better stretch, still not what you except from your max contract guy but its drifting in a better direction"
2. "He finishes at the rim with a higher percentage than Lebron. But thats also probably because they overplay the 3 so his degree of difficulty isnt the same as players like Kyrie and Lebron."

**Decision rule:** A statistic alone is not enough. The text must use the evidence to support or explain a claim.

---

### Label 2: `unsupported_take`
**Definition:** A comment makes a declarative, unsupported evaluation, ranking, prediction, or comparison — an opinion stated without enough reasoning to count as analysis.

**Clear examples:**
1. "Since '98 but whatever I guess. LeBron is well past Kobe at this point and anyone who denies it is simply delusional."
2. "What has Simmons accomplished besides winning ROY after a full year of NBA training? He can tear up the regular season all he wants but teams have already shown they know how to game plan against him in the playoffs when it matters."

**Decision rule:** Use `unsupported_take` for a *declarative* opinion. If the same evaluative content is phrased mainly as a question or a prompt for others' opinions (e.g. "Who would you rather have, A or B?"), it goes to `information_question` instead.

---

### Label 3: `reaction_meme`
**Definition:** A comment primarily expresses humor, emotion, fandom, sarcasm, a rant, a personal/community share, or casual conversation rather than making a substantive basketball claim.

**Clear examples:**
1. "Lol Raptor fans stop bitching so much. It's so fucking annoying going into a thread and just seeing FUCK THE REF comments"
2. "Man the Spurs are giving me hope again. After that Bulls loss I was ready to give up on this season. But we still have the GOAT coach and we have plenty of talent to work with. We just have to play with pride like we have been lately. GSG!"

**Decision rule:** Personal news, fan celebrations, rants, and jokes belong here unless the main purpose is a factual update or reasoned argument.

---

### Label 4: `information_question`
**Definition:** A comment neutrally reports a fact, statistic, or news item, **or** asks a question / poses a discussion prompt (including "who would you rather" debate prompts and "what happened to X?" questions), without making a supported evaluative claim of its own.

**Clear examples:**
1. "[Charania] OKC's Raymond Felton and Dennis Schroder have been suspended one game by the NBA for altercation against the Bulls."
2. "Who would you rather have moving forward, Doncic or Ingram? Stats for the 2018-19 season Doncic: 18.4/6.7/4.9 55.8 TS% Ingram: 15.2/4.0/2.2 52.2 TS%"

**Decision rule:** A neutral stat report is not analysis unless the writer explains what the information means or why it supports a claim. Question- or prompt-form posts land here even when they carry an evaluative angle, because the main act is asking rather than asserting.

---

## 3. Label Precedence and Edge-Case Rules

Use these rules in order when one example seems to fit more than one label:

1. **Reasoned analysis first:** Label `analysis` only when the post has a claim + evidence + explanation.
2. **Question / prompt / neutral report next:** Label `information_question` when the post mainly asks a question, poses a discussion prompt, or neutrally reports information.
3. **Declarative opinion next:** Label `unsupported_take` for declarative unsupported evaluations, rankings, or predictions.
4. **Reaction/social last:** Label `reaction_meme` for emotion, humor, personal shares, sarcasm, rants, or casual interaction.

### Anticipated hard cases

| Edge case | Possible labels | Rule I will use |
|---|---|---|
| A post gives one statistic and makes a bold claim | `analysis` / `unsupported_take` | Use `analysis` only when the text explains how the stat supports the claim. Otherwise use `unsupported_take`. |
| A question includes player stats | `information_question` / `unsupported_take` | Use `information_question` whenever the main act is a question or a prompt for others' opinions, even if it carries an evaluative angle. Use `unsupported_take` only for a declarative opinion. |
| A factual update ends with a joke | `information_question` / `reaction_meme` | Label the main communicative purpose. If the post mainly shares a fact, use `information_question`; if the joke or emotion is the point, use `reaction_meme`. |
| A personal or community story mentions basketball | `reaction_meme` / `information_question` | Use `reaction_meme` unless the central purpose is neutral reporting. |

### Difficult annotation decisions

1. **Example:** "Collin Sexton is shooting 40% from 3 through 31 games But apparently he doesn't know how to play basketball /s. He's been underrated as a rookie this year, and should really get more attention."
   **Possible labels:** `analysis` / `unsupported_take`
   **Final label:** `unsupported_take`
   **Why:** It opens with a real stat, but the central act is the evaluation ("underrated," "should get more attention") rather than an explanation of how the 40% supports a specific claim. Under the Section 3 rule, an unexplained stat plus a bold claim is `unsupported_take`.

2. **Example:** "The Utah Jazz have played more road games up to December 17th than any NBA team since the 1980 Kansas City Kings. Just saw this statistic on the Jazz local broadcast - absolutely insane. Nothing to excuse our poor play thus far, of course, but I'm confident the Jazz are much better than our record presently suggests."
   **Possible labels:** `analysis` / `information_question` / `reaction_meme`
   **Final label:** `analysis`
   **Why:** It sits between a neutral stat report and an emotional fan post, but the writer uses the schedule stat to support an explicit claim ("the Jazz are much better than our record suggests"), which crosses the claim + evidence + explanation bar for `analysis`.

3. **Example:** "James Harden finishes with 47 (15 free throw points)! He is on a one-man mission to bring the rockets back from the depths of hell. He is crazy! Im so sad we traded him."
   **Possible labels:** `information_question` / `reaction_meme`
   **Final label:** `reaction_meme`
   **Why:** It reports a real stat line, but the dominant purpose is emotional fandom ("one-man mission," "so sad we traded him"). Under the "label the main communicative purpose" rule, the feeling is the point, so it is `reaction_meme`.

---

## 4. Data Collection and Annotation Plan

### Data source
I, with the help of AI, collected public r/nba post titles and comments from the 2018–19 NBA season (roughly December 2018 through February 2019), and put them  all into a csv file.

### Dataset format
The final CSV includes:

```csv
text,label,notes
"Example comment",analysis,"Why this was difficult, if applicable"
```

### Dataset size and achieved distribution
I used **200 total examples**. The plan targeted a minimum 20% share per class; the achieved distribution did **not** meet that floor for two classes (see Known limitations).

| Label | Target count | Target min share | Achieved count | Achieved share |
|---|---:|---:|---:|---:|
| `analysis` | 40 | 20% | 16 | **8.0%** |
| `unsupported_take` | 60 | 30% | 68 | 34.0% |
| `reaction_meme` | 40 | 20% | 34 | **17.0%** |
| `information_question` | 60 | 30% | 82 | 41.0% |
| **Total** | **200** | **100%** | **200** | **100%** |

### Annotation process
1. Read each example fully before assigning a label.
2. Apply the precedence rules in Section 3.
3. Add a note for genuinely ambiguous examples (see the `notes` column).
4. Recheck every `analysis` example for claim + evidence + explanation.
5. Recheck the final label distribution. *(In practice `analysis` and `reaction_meme` finished below the 20% floor; this was not corrected before training and is recorded as a limitation.)*

### Quality checks before training
- Confirmed there are no duplicate texts.
- Confirmed all labels are valid strings from the taxonomy (`analysis`, `unsupported_take`, `reaction_meme`, `information_question`).
- Compared word-count distributions by label (medians: analysis 42, information_question 31, unsupported_take 30.5, reaction_meme 25.5) — `analysis` skews longest, a possible length signal to watch.
- Ran a word-count-only diagnostic to check it does not beat the majority-class baseline.
- Read a random sample of examples per label for consistency.

### Known limitations
- **Data vintage:** posts are from the 2018–19 NBA season, so player, team, and trade references are dated and tied to that period's storylines (e.g. AD trade rumors, Doncic's rookie year).
- **Source-specific language:** short Reddit excerpts with heavy slang, sarcasm, and `/s` markers; some posts are truncated, which removes context an annotator or model could use.
- **Severe class imbalance:** `analysis` is only 8% (16 examples) and `reaction_meme` 17% (34 examples), both below the intended 20% floor. With a 15% test split (30 examples), minority classes have very few test cases — `analysis` had only 3 in the test set.

---

## 5. Model and Training Plan

### Base model
I fine-tuned `distilbert-base-uncased` using the project starter notebook in Google Colab.

### Data split
The notebook created a **70% train / 15% validation / 15% test** split from the single labeled CSV (test set = 30 examples).

### Initial training settings
- Epochs: 3
- Learning rate: `2e-5`
- Batch size: 16
- Evaluation set: locked test split

### Hyperparameter decision
I started with the notebook defaults because the dataset is small. I planned to change a hyperparameter only for a concrete reason such as rising validation loss, a training error, or poor minority-class recall. In this run the fine-tuned model collapsed to the majority class (see Section 7), which is itself a concrete signal that the defaults plus the imbalanced 200-row dataset were insufficient; any retraining (class weighting, oversampling minority classes, more data, more epochs) would be recorded in the README.

---

## 6. Zero-Shot Baseline Plan

I evaluated Groq's `llama-3.3-70b-versatile` on the same locked 30-example test set before reviewing fine-tuned test performance.

### Baseline prompt requirements
The prompt:
- Includes the four label definitions.
- Includes the edge-case rules.
- Instructs the model to output only one valid label string.
- Uses the same label names as the CSV (`analysis`, `unsupported_take`, `reaction_meme`, `information_question`).

### Baseline quality check
If more than about 10% of outputs are unparseable, I revise the output-format instruction and rerun the baseline before proceeding.

---

## 7. Evaluation Plan

### Metrics
I report:
- Overall accuracy for the zero-shot baseline and fine-tuned model.
- Per-class precision, recall, and F1 for both models.
- A confusion matrix for the fine-tuned model.
- At least three fine-tuned-model errors with written analysis.

### Why accuracy is not sufficient
Accuracy can hide poor performance on a smaller or harder class. Per-class precision, recall, and F1 show whether the model learned each boundary, especially `analysis` versus `information_question` and `unsupported_take` versus `reaction_meme`.

### Results (locked 30-example test set)

| Model | Overall accuracy |
|---|---:|
| Zero-shot baseline (`llama-3.3-70b-versatile`) | **66.67%** |
| Fine-tuned (`distilbert-base-uncased`) | **40.0%** |
| Improvement (fine-tuned − baseline) | **−26.67 pts** |

**The fine-tuned model performed worse than the zero-shot baseline.** The confusion matrix shows it collapsed to the majority class: it predicted `information_question` for 29 of 30 test examples (and `unsupported_take` once), getting all 12 true `information_question` cases right and **0** correct on the other three classes.

Approximate per-class F1 for the fine-tuned model:

| Label | Recall | Precision | F1 |
|---|---:|---:|---:|
| `analysis` | 0/3 = 0.00 | 0.00 | 0.00 |
| `unsupported_take` | 0/10 = 0.00 | 0.00 | 0.00 |
| `reaction_meme` | 0/5 = 0.00 | — | 0.00 |
| `information_question` | 12/12 = 1.00 | 12/29 ≈ 0.41 | ≈ 0.58 |

### Fine-tuned-model error analysis (three examples)
1. A true `unsupported_take` ("LeBron is well past Kobe... anyone who denies it is delusional") was predicted `information_question` — the model defaulted to the majority class instead of recognizing a declarative opinion.
2. A true `reaction_meme` ("FUCK THE REF" rant) was predicted `information_question` — emotional/rant cues were ignored; with only ~24 reaction examples in training the class was undertrained.
3. A true `analysis` (claim + stat + explanation) was predicted `information_question` — with only ~11 training examples for `analysis`, the model never learned the boundary.

All three errors share one cause: **majority-class collapse driven by the 200-row, heavily imbalanced dataset**, not random labeling noise.

### Definition of success — outcome
I considered the fine-tuned classifier meaningfully useful only if:

1. It exceeds the zero-shot baseline on overall accuracy. → **Not met** (40.0% vs 66.67%).
2. No label has an F1 below **0.50** on the test set. → **Not met** (three classes at 0.00).
3. Its errors can be explained by identifiable boundary cases rather than obvious labeling inconsistency or a length shortcut. → **Partly met** — the errors are explained by a recognizable failure mode (majority-class collapse under class imbalance), not by inconsistent labels.
4. The results do not show evidence of data leakage or a trivial surface-feature shortcut. → **Met** — no leakage; the model underfit rather than exploiting a shortcut.

**Conclusion:** On this dataset the fine-tuned DistilBERT did *not* beat the zero-shot LLM. Because a 200-row dataset yields a tiny test set (some classes with only 3 examples), per-class metrics are interpreted cautiously. The most likely fixes are a larger and better-balanced dataset, class weighting or minority oversampling, and more training epochs.

---

## 8. AI Tool Plan and Disclosure

### Label stress-testing
I used an AI tool to generate boundary examples using my taxonomy, classified them myself, and revised the definitions only when I found a specific inconsistency.

### Annotation assistance
I used AI to generate tentative labels for a candidate text pool. I reviewed every final label myself, changed labels when needed, and recorded difficult decisions in the notes column.

### Failure analysis
After evaluating the models, I gave the misclassified examples and the confusion matrix to an AI tool to suggest possible patterns. I verified each suggested pattern by rereading the examples and confirmed the majority-class-collapse explanation against the actual confusion matrix before reporting it.

### Specific AI-use log

| Task | What I asked the AI to do | What it produced | What I changed, verified, or overrode |
|---|---|---|---|
| Label stress test | Generate 5–10 boundary comments that sit between `analysis` and `unsupported_take` | Borderline stat-plus-claim examples | Reclassified them myself; tightened the `analysis` rule to require an explicit explanation of how the stat supports the claim |
| Error-pattern review | Analyze the fine-tuned confusion matrix and 30-example results for failure patterns | Identified majority-class collapse to `information_question` (29/30 predictions) and tied it to class imbalance | Verified the counts against `evaluation_results.json` and the matrix; confirmed three of four classes had F1 = 0 before reporting |

---
