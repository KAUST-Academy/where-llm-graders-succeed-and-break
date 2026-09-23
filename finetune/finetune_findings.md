# Fine-tuning open graders — results

*Numbers regenerate from `finetune/pairwise_analysis.py`; workbooks in
`finetune/results/FT-*.xlsx` (Part I, single-exam study) and
`finetune/results/X-*.xlsx` (Part II, cross-exam study).*

**Part I (§1–8)** is the original single-exam study on the Computer Vision exam,
kept verbatim as the record behind the FT-* workbooks. **Part II (§9–17)** is
the cross-exam study added when the Introduction-to-AI dataset landed: the same
five models with adapters trained per-exam and pooled, evaluated on both exams.
The pooled adapters are the study's main adapters going forward.

---

## 1. Setup

**Split.** The 570 dual-graded students are split **456 train / 114 held-out**,
shuffled with a fixed seed (42), and pinned as explicit ID lists in
`data/split_meta.json`. Every run uses the *identical* split, so all numbers are
directly comparable. Training examples: **1,494** per (student, question).

**Training.** LoRA, bf16 (no quantization), 2 epochs, lr 2e-4 cosine, effective
batch 16, gradient checkpointing, **1×A100**. r=16, α=32, dropout 0.05.
`all-linear` targets except the 30B MoE (attention-only).

**Two supervision recipes** (identical prompts; only the output contract differs):

| recipe | target per (student, question) | supervision |
|---|---|---|
| **marks** | `{score, bonus}` | TA marks (per-question mean of TA1, TA2) |
| **bd** | `{score, bonus, task_breakdown[…]}` | Gemini D01, distilled (no reasoning field) |

**Models** (five, three families, ~4B-effective to 30B):
Qwen2.5-Coder-7B, Qwen2.5-Coder-14B, Qwen3-Coder-30B-A3B (MoE), Llama-3.1-8B,
Gemma-4-E4B.

---

## 2. Metrics

For each student: **error = AI total − TA average** (out of 35).

| metric | definition | captures |
|---|---|---|
| **MAE** | mean of \|error\| | how far off, ignoring direction |
| **Bias** | mean of *signed* error | direction: **−** = harsher than TAs, **+** = more generous |
| **Scatter** | std of error | inconsistency / noise |
| **Paired diff** | third-grader test (see `pairwise_finetune.md`) | is it as reliable as a human TA? |

**The headline test (paired third-grader).** Comparing the AI to the *average*
of two TAs is unfair — averaging cancels human noise. So we treat the AI as a
**third grader** and compare only 1-vs-1 disagreements, paired per student:

```
d_i = mean(|AI−TA1|_i , |AI−TA2|_i) − |TA1−TA2|_i
```

Reading the 95% CI of mean(d): **contains 0** → indistinguishable from a human
TA; **entirely < 0** → *better* than a human TA; **entirely > 0** → worse.
`pairwise_finetune.md` explains this in full and lists every run.

**Reference points** (114 held-out students): inter-grader floor `|TA1−TA2|` =
**2.828**; Gemini D01 ≈ **1.85 MAE**.

---

## 3. Core result: fine-tuning reaches (and sometimes beats) human parity

| model | recipe | base MAE | **FT MAE** | FT bias | FT scatter | paired verdict |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | marks | 6.680 | **1.968** | +0.98 | 2.67 | = TA |
| Qwen2.5-7B | bd | 4.766 | **1.966** | +0.67 | 2.67 | = TA |
| Qwen2.5-14B | marks | 4.039 | **1.868** | +0.92 | 2.62 | **BETTER than TA** |
| Qwen2.5-14B | bd | 3.503 | **2.004** | +0.34 | 2.80 | = TA |
| Qwen3-30B-A3B | marks | 4.839 | **1.775** | +0.52 | 2.56 | **BETTER than TA** |
| Qwen3-30B-A3B | bd | 4.528 | **2.136** | +0.91 | 2.91 | = TA |
| Llama-3.1-8B | marks | 7.993 | **2.002** | +0.34 | 2.67 | = TA |
| Llama-3.1-8B | bd | 6.962 | **2.131** | +0.38 | 2.84 | = TA |
| Gemma-4-E4B | marks | 4.238 | **1.903** | +0.36 | 2.66 | = TA |
| Gemma-4-E4B | bd | 3.962 | **2.354** | +1.12 | 3.00 | = TA |

**Every un-tuned base is significantly WORSE than a human TA** (paired +0.94 …
+5.43, every CI above 0). **Every fine-tune reaches at least parity**, and **two
configurations (14B-marks, 30B-marks) are significantly BETTER than a human TA** —
their whole 95% interval sits below zero (14B −0.457 [−0.89, −0.03]; 30B −0.556
[−0.98, −0.13]). i.e. they disagree with the TAs *less* than the TAs disagree
with each other.

Every fine-tune also lands **below the inter-grader floor (2.828)** and in the
neighborhood of Gemini D01 (~1.85), from ~1,500 examples.

---

## 4. Size and family barely matter

Fine-tuned MAE (marks), by model:

| Qwen-7B | Llama-8B | Gemma-4 | Qwen-14B | Qwen-30B |
|---|---|---|---|---|
| 1.968 | 2.002 | 1.903 | 1.868 | 1.775 |

All five models — 4B-effective to 30B, three families — converge to a tight
**1.78–2.00 (marks) / 1.97–2.35 (bd)** band. The 30B (1.775) edges the 7B
(1.968), but the gap is a fraction of the base-vs-tuned effect and the paired CIs
overlap heavily. **A 7B fine-tune is as good a grader as a 30B fine-tune**, and
the base model's starting quality does not predict the fine-tuned result (Llama
starts worst at 7.99 and lands at 2.00, right with everyone else).

---

## 5. Personas: fine-tuning immunizes against "be strict"

The paper's thesis — a strict-grader persona *breaks* an open model — holds
dramatically for base models and is **erased by fine-tuning**, across all five
models and **both recipes**.

**Base models collapse under "be strict"** (MAE, out of 35):

| model | neutral | **strict** | rigorous | exacting | base spread |
|---|---|---|---|---|---|
| Qwen-7B (marks) | 6.68 | 12.00 | 9.01 | 6.17 | 5.83 |
| Qwen-14B (marks) | 4.04 | **23.72** | 3.68 | 7.52 | **20.03** |
| Qwen-30B (marks) | 4.84 | 16.33 | 5.01 | 5.74 | 11.49 |
| Llama-8B (marks) | 7.99 | **25.28** | 12.28 | 12.16 | 17.28 |
| Gemma-4 (marks) | 4.24 | 10.06 | 4.66 | 6.06 | 5.82 |

Under "be strict," the bigger models dock **16–25 points out of 35** from every
student (bias −16 to −25). "Be strict" does not make them pickier; it makes them
catastrophically harsh.

**The fine-tuned models are flat** (same personas):

| model | neutral | strict | rigorous | exacting | **FT spread** |
|---|---|---|---|---|---|
| Qwen-7B (marks) | 1.97 | 1.94 | 1.98 | 1.93 | **0.049** |
| Qwen-14B (marks) | 1.87 | 1.95 | 1.83 | 1.91 | **0.122** |
| Qwen-30B (marks) | 1.78 | 1.88 | 1.77 | 1.82 | **0.112** |
| Llama-8B (marks) | 2.00 | 2.20 | 1.92 | 1.93 | **0.279** |
| Gemma-4 (marks) | 1.90 | 2.18 | 1.95 | 1.92 | **0.278** |

Persona sensitivity drops by **21–164×** (per-model ratios 119/164/103/62/21).
The strict persona, which multiplies a base model's error by ×1.8–×5.9, moves a
fine-tuned one by less than a third of a point (max spread 0.279).
The fine-tuned graders are anchored to the rubric hard enough that the persona
instruction stops mattering.

The **bd** recipe shows the same pattern, with a single exception:
**Llama-bd-strict** stays elevated (MAE 4.08, spread 1.99) — bd fine-tuning
*dampens* but does not fully immunize the weakest base. Its **marks** counterpart
(2.20, spread 0.28) *is* immunized, so the vulnerability is recipe-specific, not
model-specific. Full grid in `pairwise_finetune.md` and the persona workbooks.

Wording matters in the expected direction everywhere: **strict** (most
destructive) > **rigorous** > **exacting** (mildest) — evidence it is the
connotation of the word, not any persona, that does the damage. (`neutral` is
itself "strict but fair," so the baseline already carries mild framing.)

---

## 6. What fine-tuning changes

Un-tuned models grade **harsh** (negative bias, large scatter). Fine-tuning does
two things:

1. **Scatter falls** to a tight ~2.6–3.0 in every model — grading becomes
   consistent.
2. **Bias is corrected toward 0** — fine-tuned bias is **+0.34 … +1.12** (a
   slight residual generosity), versus base biases running to −25 under a strict
   persona.

---

## 7. marks vs bd

The recipe gap has largely closed. Across models:

- **marks slightly beats bd in 4 of 5 models** on MAE (e.g. 7B 1.968 vs 1.966 —
  a tie; 30B 1.775 vs 2.136; Gemma 1.903 vs 2.354).
- The two configurations that reach *BETTER than TA* are both **marks**.
- So the earlier "bd rescues a weak base" story does not survive: Llama marks
  (2.002) ≈ Llama bd (2.131).

Practically: **marks-only supervision (pure human labels, no model distillation)
is sufficient and marginally better.** bd's only remaining edge is producing a
per-task breakdown as a by-product.

---

## 8. Limitations

- **n = 114, single split, single seed.** CIs on MAE are ~±0.45 wide; the paired
  test is the load-bearing statistic. No k-fold / repeated seeds.
- **`bd` distils a model** (Gemini D01), so that arm measures distillation; only
  `marks` trains purely on human labels.
- **The 30B is not a clean size point** — MoE (vs dense) and attention-only LoRA.
- **Evaluation is not bitwise reproducible.** vLLM batching reorders reductions
  (~0.01 MAE drift). Gemma-4 was evaluated through HuggingFace generation in
  Part I because the vLLM of the time could not apply its adapter. *Stale as of
  Part II:* the current vLLM serves the gemma adapter fine — the Part II vLLM
  rerun reproduces the Part I HF numbers to within 0.1 across gemma's four
  re-run cells (marks-adapter cell: 1.881 vs 1.903), which
  also cross-validates the two serving stacks against each other.

---

## Reproduction

```bash
PY=/ibex/user/shaebiyy/conda-environments/gemma4-vllm/bin/python  

# data (deterministic; same seed -> same files)
$PY finetune/build_finetune_data.py  --seed 42 --train-frac 0.8   # marks -> data/
$PY finetune/build_breakdown_data.py                              # bd    -> data_bd/

# train (override the interpreter with PY=<python> if needed)
EPOCHS=2 NO_4BIT=1 SAVE_STEPS=25 bash finetune/run_finetune.sh <key>            # marks
EPOCHS=2 NO_4BIT=1 SAVE_STEPS=25 bash finetune/run_breakdown_finetune.sh <key>  # bd
#   keys: 7b | 14b | 30b | llama | gemma   (30B MoE targets auto-set)

# evaluate (vLLM for Qwen/Llama; HF for Gemma)
bash finetune/run_vllm_eval.sh <key> <marks|bd>     # 7b 14b 30b llama
bash finetune/run_hf_eval.sh   gemma  <marks|bd>    # gemma

# persona sweep for one model (base + FT × strict/rigorous/exacting)
bash finetune/persona_sweep.sh <key>              # bd
RECIPE=marks bash finetune/persona_sweep.sh <key> # marks

# stats (paired third-grader test + explanation) -> pairwise_finetune.md
$PY finetune/pairwise_analysis.py --md finetune/pairwise_finetune.md
```

---
---

# Part II — Cross-exam study: one grader for two exams (2026-08-30/31)

*Workbooks: `finetune/results/X-<TAG>[bd]-<trained-on>-on-<graded>[-<persona>].xlsx`
(200 files: 80 matrix/base cells + 120 persona cells). Every MAE below is
verified equal to `exams.run_summary`; adapters in `finetune/adapters/*-{intro,both}`.*

## 9. What changed and why

The repo gained a second dataset — the **Stage 2 Introduction-to-AI exam** — and
with it the question Part I could not ask: *does a fine-tuned grader survive a
change of exam?* Three training sets per model (cv, intro, pooled), each
evaluated on both exams' held-out students, plus the untuned base: an
8-condition grid per (model, recipe), 80 cells in all. **The pooled adapters are
now the study's main adapters**; every Part I experiment was rerun on them.

The two exams differ in nearly every surface property:

| | CV (Stage 3) | Intro (Stage 2) |
|---|---|---|
| students (dual-graded) | 570 | 1,038 |
| questions | Q1–Q4 | Q1–Q3 |
| marks | 35 + 13 bonus (separate columns) | 56 + 9 bonus (**folded** into each grade) |
| ground truth | `Practical_AI_exam_grades.xlsx` | `Practical_AI_exam_grades.csv` |
| rubric headers | `## Question 1:` | `## Q1:` |
| guidelines document | yes | **none** |
| missing submissions | none | 96 notebooks (~3%) |
| inter-grader floor \|TA1−TA2\| (held-out) | 2.828 | **5.187** |
| TA mean (held-out, eval convention) | 25.28 / 35 | 30.92 / 65 |

**Splits.** Intro gets its own pinned 80/20 split (seed 42): **830 train / 208
held-out** → 2,413 / 605 examples (`data_intro/split_meta.json`). The pooled
("both") training set is the **union of the two per-exam train splits** — never a
reshuffle of the union — so each exam's held-out students are identical across
every arm and every cell of a column is measured on the same students. Student
numbers restart at 1 per exam, so splits are exam-qualified everywhere
(`eval_students.json` is `{"cv": […], "intro": […]}` for pooled builds). Pooled
training: **3,907 examples** (1,494 cv + 2,413 intro); pooled eval 387 + 605.

**Folded bonus.** Intro's TAs recorded one grade per question with bonus folded
in (3.4/10.6/6.2% of TA marks exceed the base caps, confirming it). Training
targets recover a `(score, bonus)` pair by filling base marks first and
overflowing into bonus — exact for evaluation because agreement on intro is
measured on the **sum** (AI score + AI bonus vs TA grade), matching
`introduction_to_ai_grade_with_local.py`. CV stays score-only on both sides, as
in Part I. Marks above base+bonus (~0.5% of intro TA marks) are clamped, as the
root grader clamps.

**Prompts.** Same assembly as Part I with the exam's own label, rubric, caps and
reference solution; intro prompts carry **no guidelines block** (the exam shipped
none). The caps are written into every prompt — that is what makes cross-exam
inference coherent. `finetune/exams.py` holds the per-exam registry;
`--dataset {cv,intro,both}` threads through the builders, runners and eval.
Regenerating the cv data under the new code is **byte-identical** to the pinned
Part I data (1,494/387 rows, zero prompt or completion diffs; bd 1,494/386),
and `exams.run_summary` reproduces `common.compute_run_summary` exactly on D01
(MAE 1.642, n=570) — Part I numbers are untouched by construction.

**Training.** Identical recipe to Part I (2 epochs, lr 2e-4 cosine, r=16 α=32,
effective batch 16, seed 42). Dense models trained 4×A100 data-parallel with
grad-accum divided by world size so the global batch stays 16
(`training_args.bin` confirms eff_batch=16, world=4 for the new dense adapters);
gemma-4 trained 1×A100 like its Part I adapter after its MatFormer architecture
hit DDP's unused-parameter check on 4 GPUs (all four gemma reruns: mask check
ok, eff_batch 16, world 1). 20 new adapters (5 models × {marks, bd} × {intro,
both}); the 10 Part I cv adapters are reused unchanged — they never saw intro,
so they are already valid transfer subjects.

## 10. The transfer matrix

MAE vs TA average on the held-out students. Columns = what the adapter was
trained on. **marks**:

| model | CV: base | cv | intro | pooled | Intro: base | cv | intro | pooled |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | 6.60 | 1.93 | 3.22 | **1.84** | 6.48 | 6.37 | 3.67 | **3.40** |
| Qwen2.5-14B | 4.06 | 1.90 | 3.29 | **1.89** | 7.70 | 5.41 | 3.35 | **3.29** |
| Qwen3-30B-A3B | 4.76 | 1.80 | 2.59 | **1.75** | 8.60 | 5.22 | 3.67 | **3.44** |
| Gemma-4-E4B | 4.29 | 1.88 | 3.79 | **1.88** | 4.66 | 4.79 ‡ | 3.51 | 3.53 |
| Llama-3.1-8B | 7.99 | 1.99 | 4.95 | **2.01** | 12.30 | 5.02 | 3.48 | **3.37** |
| **mean** | 5.54 | 1.90 | 3.57 | **1.87** | 7.95 | 5.36 | 3.54 | **3.41** |

**bd**:

| model | CV: base | cv | intro | pooled | Intro: base | cv | intro | pooled |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | 4.94 | 1.97 | 3.55 | **1.94** | 7.36 | 5.37 | 4.37 | **4.19** |
| Qwen2.5-14B | 3.47 | 1.98 | 3.28 | **1.84** | 10.00 | 5.79 | 4.09 | **4.02** |
| Qwen3-30B-A3B | 4.58 | 2.15 | 3.01 | **1.99** | 9.58 | 8.21 | 4.28 | **4.07** |
| Gemma-4-E4B | 3.87 | 2.38 | 3.30 | **2.22** | 4.50 | 4.80 ‡ | 4.25 | **4.22** |
| Llama-3.1-8B | 6.78 | 2.15 | 4.46 | **2.04** | 12.37 | 6.61 | 4.22 | **4.25** |
| **mean** | 4.73 | 2.13 | 3.52 | **2.01** | 8.76 | 6.16 | 4.24 | **4.15** |

Three findings:

1. **Transfer is real and bidirectional.** An adapter improves grading on the
   exam it never saw: intro mean 7.95 → 5.36 under cv-training; cv 5.54 → 3.57
   under intro-training. It stops short of in-domain (3.54 / 1.90). intro→cv
   transfers relatively better, consistent with intro's 1.6× larger training
   set (a size confound, §16).
2. **Pooling is free.** The pooled adapter matches or beats each specialist *on
   the specialist's own exam* in all 8 (exam × recipe) mean columns, and
   per-model in nearly every row (cv marks 1.84 vs 1.93, 1.75 vs 1.80, …).
   Adding a second course's data costs the first course nothing.
3. **The exams are not on one scale.** CV is out of 35, intro out of 65
   (sum convention). Normalised, in-domain error is ≈5% on both — intro's larger
   absolute MAE reflects a harder exam (its TAs disagree 5.187 vs 2.828), not
   worse grading.

**‡ the one negative transfer cell.** Gemma-4-E4B is the only model cv-training
fails to help on intro (marks 4.79 vs base 4.66; bd 4.80 vs 4.50). It has by far
the strongest untuned base on intro, so there is little headroom; in-domain
labels still move it (3.51), and its cv-side behaviour is normal. Reported, not
hidden.

**Base-model mechanism.** The bases mis-calibrate in *opposite directions* per
exam: harsh on CV (bias −2.3…−5.9; Llama mean 19.78 vs TA 25.28) and generous on
intro (bias +2.8…+9.5; Llama 40.47 vs TA 30.92) — no fixed prompt-level offset
could repair both. Untuned Llama's cv std is 2.09 (it gives nearly every student
the same mark); fine-tuning revives the spread to ~7.3–7.5, most of the way to the grader
average's 8.0.

## 11. Third-grader test, pooled adapters

Same paired test as §2, both exams (inter-grader floors 2.828 / 5.187):

| run | n | \|AI−TAavg\| | \|AI−TA1\| | paired diff [95% CI] | verdict |
|---|---|---|---|---|---|
| pooled 7B marks, CV | 114 | 1.837 | 2.350 | −0.479 [−0.90, −0.05] | **BETTER than TA** |
| pooled 14B marks, CV | 114 | 1.892 | 2.382 | −0.464 [−0.89, −0.04] | **BETTER than TA** |
| pooled 30B marks, CV | 114 | 1.753 | 2.375 | −0.557 [−0.98, −0.13] | **BETTER than TA** |
| pooled Gemma marks, CV | 114 | 1.879 | 2.408 | −0.454 [−0.89, −0.02] | **BETTER than TA** |
| pooled Llama marks, CV | 114 | 2.007 | 2.537 | −0.390 [−0.85, +0.07] | = TA |
| pooled 7B marks, Intro | 208 | 3.400 | 4.186 | −1.114 [−1.73, −0.49] | **BETTER than TA** |
| pooled 14B marks, Intro | 208 | 3.294 | 4.107 | −1.126 [−1.75, −0.50] | **BETTER than TA** |
| pooled 30B marks, Intro | 208 | 3.445 | 4.341 | −1.043 [−1.65, −0.43] | **BETTER than TA** |
| pooled Gemma marks, Intro | 208 | 3.530 | 4.448 | −0.938 [−1.57, −0.30] | **BETTER than TA** |
| pooled Llama marks, Intro | 208 | 3.365 | 4.335 | −1.007 [−1.65, −0.36] | **BETTER than TA** |
| pooled 7B bd, CV | 114 | 1.944 | 2.408 | −0.450 [−0.90, −0.00] | **BETTER than TA** |
| pooled 14B bd, CV | 114 | 1.844 | 2.342 | −0.514 [−0.93, −0.10] | **BETTER than TA** |
| pooled 30B bd, CV | 114 | 1.986 | 2.575 | −0.365 [−0.81, +0.08] | = TA |
| pooled Gemma bd, CV | 114 | 2.221 | 2.526 | −0.273 [−0.71, +0.17] | = TA |
| pooled Llama bd, CV | 114 | 2.044 | 2.537 | −0.369 [−0.80, +0.06] | = TA |
| pooled 7B bd, Intro | 208 | 4.191 | 5.091 | −0.372 [−1.06, +0.31] | = TA |
| pooled 14B bd, Intro | 208 | 4.016 | 4.865 | −0.509 [−1.16, +0.14] | = TA |
| pooled 30B bd, Intro | 208 | 4.072 | 5.074 | −0.399 [−1.08, +0.28] | = TA |
| pooled Gemma bd, Intro | 208 | 4.221 | 5.088 | −0.372 [−1.05, +0.31] | = TA |
| pooled Llama bd, Intro | 208 | 4.248 | 5.049 | −0.308 [−0.98, +0.37] | = TA |

**The pooled marks adapter is significantly BETTER than a human TA in 9 of 10
(model × exam) cells** — including all five models on intro — and the bd adapter
reaches at least TA parity in all 10 (better in two: 7B-bd and 14B-bd on CV).
No cell reads worse-than-TA. Candour
note: intro's bar is lower because its TAs disagree more (5.187 vs 2.828); every
point estimate is negative regardless. Reproduce with
`pairwise_analysis.py --runs X-<TAG>[bd]-both-on-{cv,intro}` (the script is now
exam-aware, inferring the exam from the `-on-<exam>` filename tag).

## 12. Persona sweep on the pooled adapters — immunity transfers, and a new finding

All 3 harsh personas × 2 exams × {base, pooled} × 5 models × both recipes
(120 cells, job 51256325). MAE (bias) per persona, **marks on CV**:

| model | arm | neutral | strict | rigorous | exacting | worst drift |
|---|---|---|---|---|---|---|
| Qwen-7B | base | 6.60 (−5.9) | 12.00 (−11.9) | 9.07 (−8.9) | 6.07 (−5.5) | +5.40 |
| Qwen-7B | pooled | 1.84 (+0.9) | 1.84 (+0.8) | 1.85 (+0.8) | 1.81 (+0.7) | **+0.02** |
| Qwen-14B | base | 4.06 (−2.3) | 23.81 (−23.8) | 3.65 (−1.6) | 7.31 (−6.5) | +19.75 |
| Qwen-14B | pooled | 1.89 (+0.8) | 1.85 (−0.2) | 1.87 (+0.9) | 1.81 (+0.5) | **−0.02** |
| Qwen-30B | base | 4.76 (−2.3) | 17.28 (−17.2) | 4.96 (−3.0) | 5.76 (−5.1) | +12.51 |
| Qwen-30B | pooled | 1.75 (+0.6) | 1.80 (+0.4) | 1.79 (+0.7) | 1.82 (−0.7) | **+0.07** |
| Gemma-4 | base | 4.29 (−3.2) | 10.10 (−10.1) | 4.69 (−4.0) | 6.03 (−5.7) | +5.81 |
| Gemma-4 | pooled | 1.88 (+0.8) | 1.93 (−0.0) | 1.89 (+0.8) | 1.82 (+0.5) | **+0.05** |
| Llama-8B | base | 7.99 (−5.5) | 25.28 (−25.3) | 12.35 (−12.0) | 11.95 (−11.8) | +17.28 |
| Llama-8B | pooled | 2.01 (+0.9) | 1.96 (+0.4) | 1.93 (+0.7) | 1.91 (+0.8) | **−0.05** |

**marks on Intro**:

| model | arm | neutral | strict | rigorous | exacting | worst drift |
|---|---|---|---|---|---|---|
| Qwen-7B | base | 6.48 (+2.8) | 6.70 (−3.6) | 6.04 (+0.6) | 6.15 (+2.8) | +0.22 |
| Qwen-7B | pooled | 3.40 (+0.7) | 3.36 (+0.2) | 3.30 (+0.4) | 3.31 (−0.0) | **−0.04** |
| Qwen-14B | base | 7.70 (+5.9) | 8.83 (−7.3) | 7.71 (+5.8) | 6.33 (+2.4) | +1.13 |
| Qwen-14B | pooled | 3.29 (+0.3) | 3.61 (−1.9) | 3.23 (+0.0) | 3.28 (−0.7) | **+0.32** |
| Qwen-30B | base | 8.60 (+7.3) | 6.34 (−3.0) | 7.34 (+5.2) | 6.30 (+2.4) | −1.26 † |
| Qwen-30B | pooled | 3.45 (+0.5) | 3.73 (+0.4) | 3.58 (+0.4) | 3.72 (−1.8) | **+0.29** |
| Gemma-4 | base | 4.66 (−1.3) | 11.19 (−10.9) | 5.11 (−2.9) | 6.22 (−4.8) | +6.53 |
| Gemma-4 | pooled | 3.53 (+0.6) | 3.83 (−1.5) | 3.54 (+0.1) | 3.57 (−0.6) | **+0.30** |
| Llama-8B | base | 12.30 (+9.5) | 30.92 (−30.9) | 7.16 (−1.9) | 7.97 (+3.0) | +18.62 |
| Llama-8B | pooled | 3.37 (+0.2) | 3.39 (−1.3) | 3.26 (−0.4) | 3.26 (−0.6) | **+0.03** |

**bd on CV** (worst drifts): base +2.59 / +18.48 / +13.42 / +2.37 / +18.50;
pooled +0.08 / +0.02 / +0.39 / −0.02 / **+0.62** (Llama-bd-strict 2.67 vs Part
I's un-pooled 4.08 — pooling further dampens the one Part I exception).
**bd on Intro** (worst drifts): base −0.22 / −0.81 / −1.98 / +3.67 / +18.56;
pooled +0.04 / +0.06 / +0.01 / −0.38 / −0.20.

Findings:

1. **Immunity transfers.** The pooled adapters move by at most **+0.32 (marks) /
   +0.62 (bd)** across every persona, exam and model, while bases collapse by up
   to +19.75 (cv) and +18.62 (intro). Under *strict*, base Llama on intro
   reaches MAE 30.9 with bias −30.9 — it scores nearly every student zero.
2. **† The persona effect is not even sign-consistent** — the genuinely new
   result. On intro the strict persona sometimes *improves* a base (Qwen-30B
   8.60→6.34; several bd cells too), because the bases over-grade intro and the
   induced harshness partially cancels the mis-calibration. The same wording
   that destroys a model on one exam accidentally helps it on the other:
   personas are unusable as a calibration knob, not merely risky. Part I could
   not see this — it required a second exam where the base errs in the opposite
   direction.

## 13. Error structure of the pooled adapters

Part I §6 rerun for base → pooled, both exams. Tail = students mis-graded by
>10 marks on CV, >16 on intro (the 56/35-scaled equivalent). r = Pearson
correlation of per-student |error| with the exam's intrinsic ambiguity
|TA1−TA2| (positive ⇒ mistakes concentrate where the humans also disagreed).
**marks**:

| model | CV tail base→pooled | Intro tail | CV r base→pooled | Intro r |
|---|---|---|---|---|
| Qwen-7B | 25/114 → **0** | 13/208 → 3 | −0.19 → **+0.40** | +0.32 → **+0.47** |
| Qwen-14B | 8/114 → 1 | 17/208 → 3 | +0.37 → +0.46 | +0.27 → +0.45 |
| Qwen-30B | 3/114 → **0** | 30/208 → 2 | −0.03 → **+0.40** | +0.15 → **+0.47** |
| Gemma-4 | 4/114 → **0** | 5/208 → 3 | −0.17 → **+0.35** | +0.21 → **+0.47** |
| Llama-8B | **43/114 → 0** | **66/208 → 4** | −0.31 → **+0.41** | +0.01 → **+0.38** |

bd shows the same pattern (tails 0–6 pooled; r +0.36…+0.48; intro-bd pooled
keeps a residual generosity of +2.2…+2.6 bias where marks sits at +0.2…+0.7).
All three Part I claims replicate on both exams: the catastrophic tail is
eliminated, bias lands within a mark of zero (marks), and the pooled models'
errors become *human-like* — positive r everywhere, where bases sit at zero or
negative (blundering on submissions the TAs agreed on).

## 14. External corroboration

The repo's independently produced IA01 runs (root grader, different code path,
bd-with-reasoning contract) baseline four of five bases on intro. Restricted to
the same 208 held-out students, they agree with the Part II base cells in
ranking and magnitude: A07 7.17 vs 7.36, A14 9.22 vs 10.00, A30M 10.37 vs 9.58,
LL8B 12.43 vs 12.37 (IA01 vs our bd base). Residual gaps reflect the prompt
contract difference. **Gemma-4-E4B has no IA01 counterpart** (only gemma3-12b/27b
were run on intro), so its intro base rests on our runs alone. On the cv side,
every Part II cell re-run of a Part I configuration reproduces it within ±0.18
MAE (gemma ±0.09; Llama base exactly), across a different serving stack.

## 15. Provenance and integrity

- **Commits** `0be8b8d` (pipeline generalisation + matrix results) and `2d95f63`
  (code-review fixes + persona/pairwise infra). SLURM: trainings 51034982 (array,
  4×A100, conf-2026h2 QOS) + 51073166 (gemma 1-GPU rerun); evals 51079644,
  51089142, 51094613, 51094734; personas 51256325. All eval cells 5–18 min each.
- **Naming**: `X-<TAG>[bd]-<trained-on>-on-<graded>[-<persona>].xlsx`;
  trained-on ∈ {base, cv, intro, both}.
- **Parse failures**: 8 of ~40,000 generations (5/605 on 14B-bd-intro, 3 on
  30B-bd) — truncations at the 1,024-token cap; they score 0 and are logged per
  run (`!! N/M generations failed`).
- **A reporting bug caught and fixed mid-study**: an ad-hoc analysis script
  compared AI score against TA score+bonus on cv, inflating cv MAE by ≈ the mean
  bonus in one interim report. The workbooks and `run_summary` were never wrong;
  all 80 matrix cells now verify equal to `run_summary` (0 mismatches).
- **Concurrency fix worth knowing**: the eval sbatch cleanup originally ran an
  unscoped `pkill -f EngineCore`, which can kill *sibling* array tasks' vLLM
  servers on a shared node. Servers now start under `setsid` and cleanup kills
  exactly that process group (`2d95f63`).

## 16. Limitations added by Part II

- **Transfer directions are not size-matched** (456 vs 830 training students);
  "which direction transfers better" is confounded. A 456-student intro
  subsample control is designed but not run.
- **Token cap**: 1,024 max-new-tokens truncates ≈0.5% of intro-bd generations to
  score 0, slightly inflating those cells (kept for consistency across cells).
- **Gemma-4's intro base is uncorroborated** externally (§14).
- **Both exams are one institution's course sequence** — cross-institution
  transfer is untested.
- Intro's higher inter-grader disagreement (5.187) makes its third-grader bar
  easier; every intro verdict should be read next to that floor.

## 17. Reproduction (cross-exam)

```bash
PY=/ibex/user/shaebiyy/conda-environments/gemma4-vllm/bin/python

# data (deterministic; cv regenerates byte-identical to Part I)
$PY finetune/build_finetune_data.py  --dataset intro     # -> data_intro/
$PY finetune/build_finetune_data.py  --dataset both      # -> data_both/
$PY finetune/build_breakdown_data.py --dataset intro     # -> data_bd_intro/
$PY finetune/build_breakdown_data.py --dataset both      # -> data_bd_both/

# train the 20 new adapters (skips ones whose weights exist; gemma auto-1-GPU
# is NOT automatic -- resubmit gemma tasks with --gres=gpu:a100:1 if they hit
# the DDP unused-parameter error on multi-GPU)
sbatch finetune/train_cross_exam.sbatch

# the 80-cell matrix (tasks 0-9 cv row, 10-19 intro, 20-29 both;
# INCLUDE_BASE=1 adds the 20 untuned-base cells to the cv row)
INCLUDE_BASE=1 sbatch --array=0-9 finetune/cross_eval.sbatch
sbatch --array=10-29 finetune/cross_eval.sbatch

# persona sweep on the pooled adapters (120 cells, 12 evals per server load)
sbatch finetune/persona_cross_eval.sbatch

# third-grader test on the pooled adapters (exam inferred from -on-<exam>)
$PY finetune/pairwise_analysis.py --runs \
  X-A07-both-on-cv X-A07-both-on-intro   # ... etc for all tags/recipes
```
