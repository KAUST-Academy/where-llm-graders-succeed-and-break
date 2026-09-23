---
pretty_name: Introduction to AI (Stage 2) Practical Exam — Submissions and Dual TA Grades
license: cc-by-nc-4.0
task_categories:
- text-classification
- text-generation
language:
- en
tags:
- education
- automated-grading
- llm-evaluation
- prompt-sensitivity
- code-assessment
size_categories:
- 1K<n<10K
---

# Introduction to AI — Practical Exam Grading Dataset (Stage 2)

1,038 anonymized student submissions to a university-level practical AI exam,
each independently graded by **two** teaching assistants, plus the rubric and
the instructor reference solutions.

It is the second exam of *Where LLM Graders Succeed and Break: Evidence from
Two Computer-Science Exams*. The paper's 162 grader configurations on this
exam, the graders that produced them and the analysis code live in the project
repository, alongside the Computer Vision exam
(`KAUSTAcademy/AutoGrader_Computer_Vision`); this folder is the data itself.

The dataset was built to test how sensitive LLM graders are to the *prompt* they
are given — specifically, what happens when an instructor adds a plausible
"be strict" instruction to an otherwise well-behaved grading prompt.

## What is here

| path | contents |
|---|---|
| `Practical_AI_exam_grades.csv` | ground truth: two TA grades per student, per question and total |
| `submissions_extracted/<n>/Q{1,2,3}.ipynb` | the anonymized student notebooks |
| `Solutions/` | instructor reference solution per question |
| `stage_2_rubrics_ta.md` | the marking scheme the TAs graded against |
| `few_shot_examples.json` | two worked examples per question for the few-shot runs (IG18, IA201): student code with the IG08 model's (`gemini-3.1-pro-preview`) score, bonus and reasoning, not TA marks |

Students are identified only by a number (1–1038) that is stable across every
file. Notebook `submissions_extracted/2/Q2.ipynb` belongs to the student in row
`Number == 2` of the grades CSV.

## The exam and its scale

Three questions: **Q1** tabular regression (predict food delivery time),
**Q2** PyTorch (predict age from face images), **Q3** classification (find the
most predictive feature in an anonymized dataset).

Marks come from the rubric's task tables, not its header totals — the two
disagree in the distributed PDF, which is preserved here as `stage_2_rubrics_ta.md`:

| question | base | bonus |
|---|---|---|
| Q1 Regression | 23 | 3 |
| Q2 PyTorch | 14 | 3 |
| Q3 Classification | 19 | 3 |

Two instructor corrections were issued by email *during* grading, both affecting
Q3 only (Part 1 dropped a task and became 2 marks; Part 4 dropped a task and
became 5). Both are folded into the rubric file and marked ⚠.

## Ground truth quality

Every student is graded by two TAs, so the dataset carries its own noise floor:

| statistic | value |
|---|---|
| **floor: TA-vs-TA mean absolute difference (totals)** | **5.13** (95% bootstrap CI 4.80–5.48) |
| inter-TA correlation (totals) | 0.871 |

The floor is ~8% of the ~65-point scale. An AI grader's mean absolute error
against the TA average is read against 5.13, the distance between the two
TAs, not against zero.

## Anonymization

Student names, ID numbers, emails, and submission links are absent. Notebooks
were rebuilt from scratch keeping only code cells: markdown cells, outputs,
execution counts, and all per-cell metadata were dropped, so Colab
`executionInfo` blocks (which embed a Google account name and user id) do not
survive. Remaining identifiers inside code — names in variable names or
comments, emails, ID numbers, phone numbers, filesystem usernames — were
replaced with `ANON` (identifier-safe) or `[REDACTED]`. One identifier is kept on
purpose: the exam-data download line in the notebooks, the solutions and
`few_shot_examples.json` names the course's Kaggle dataset (its owner's handle and
the `q{1,2,3}-ka-ai-2026` slugs), because that is the text the published runs graded.

Scrubbing was verified two ways: an exhaustive scan against the full roster
(every student and TA name, email and ID, plus generic email/ID/phone patterns
and notebook-format regressions), and an LLM review of all ~25k comment and
string literals looking for identifiers the roster could not know about. The
second pass is what caught a reference to an instructor by first name and one
student who had loaded the exam data from a personal account.

TAs appear only as pseudonyms (`TA_1`, `TA_2`, …). One slot carries the
composite id `TA_49+TA_50`: two TAs share that slot, which covers 18 students. The pseudonym-to-person mapping is **not** published, so grading
behaviour cannot be attributed to any individual.

The grader in the project repository is the script that produced the
published runs, prompt included, and the notebooks here are the ones it graded.

## Known gaps, stated plainly

- 24 students from the original cohort are absent: they uploaded nothing, an
  unreadable archive, or files that were not the exam. The remaining 1,038 are
  numbered contiguously.
- 75 students are missing a notebook for at least one question (96 notebooks in
  all: 4 for Q1, 44 for Q2, 48 for Q3) — they did not submit that question. An
  absent `Q<n>.ipynb` means exactly that, and the grading code scores it 0.
- The rubric's header totals contradict its own task tables in all three
  questions. This is preserved rather than corrected; it is part of what a real
  grading prompt has to cope with.
- 29 TA per-question grades (on 27 students) exceed the question's maximum
  including bonus (Q1 26, Q2 17, Q3 22), and one TA total (68.45) exceeds 65.
- Two TA grades awarded credit for a question the student never submitted
  (8.0 and 10.0, each contradicted by the paired TA's 0). Both were set to 0
  and the totals recomputed: ground truth that credits work no grader can see
  would penalise every grader, human or machine.
- Nine rows record 0.0 for exactly one TA while the paired TA awarded real
  marks: ungraded slots stored as zeros. They are kept; without them the floor
  is 5.09 (95% CI 4.76–5.42) instead of 5.13, and no conclusion in the paper
  changes. A further nine rows carry 0.0 from both TAs: their notebooks are
  the blank exam template or nearly so, so these are genuine zero-mark papers,
  not recording errors.

## The LLM runs

The paper grades this exam with 162 configurations: 132 over 17 open-weights
models and 30 with closed models (23 Gemini, 6 OpenAI, 1 Anthropic), one
workbook per configuration in the project repository's
`introduction_to_ai_results/results/`, holding the AI score, bonus and reasoning per
question alongside the TA columns. The core comparison runs every open-weights
model twice over the full cohort with an identical prompt except for its
leading persona preamble:

- **neutral** — "You are a strict but fair teaching assistant."
- **strict** — "You are a HARSH teaching assistant. Award the MINIMUM defensible
  score for any task that is incomplete, buggy, or deviates from the rubric.
  Never give partial credit if the task does not run correctly."

Everything else is held fixed: reference solution included, per-task breakdown
requested, no chain-of-thought, temperature 0, one sample per question. The
other personas are `STRICTNESS_PRESETS` in `introduction_to_ai_grade_with_local.py`,
and `build_prompt()` in the same file assembles the full prompt around them.

`behaviour` classes: `refusal` (≥90% of students zeroed, awarded-total std
< 0.5), `near-refusal` (≥90% zeroed but still varying), `collapse` (MAE ≥ 15.7,
i.e. 3.07× this exam's floor, the multiple the Computer Vision exam's MAE ≥ 8
fixes, while still discriminating between students), `graded` otherwise.

## Reproducing

The grading and analysis code lives in the project repository, which expects
this folder at `introduction_to_ai_dataset/` beside it:

```bash
# serve a model with vLLM, then:
python introduction_to_ai_grade_with_local.py --all \
  --model <hf-model-id> --api-base http://localhost:8000/v1 \
  --max-output-tokens 8192 \
  --with-solution --no-reasoning --with-breakdown \
  --strictness strict --temperature 0.0 --runs 1 \
  --tag IA02_m-<model_tag>_sol1_gd0_rs0_bd1_str-strict_t00_n1

python analysis/introduction_to_ai_run_analysis.py   # MAE, CI, behaviour class per run
```

`introduction_to_ai_submit_persona_sweep.sbatch` is the SLURM launcher: an
array job, one task per persona (`--array=0,1` gives the neutral+strict pair).
`STUDENTS=1-5` turns it into a smoke run that writes to
`introduction_to_ai_results/smoke/` instead of `results/` and leaves the run
tracker alone.
Never point two runs at one output workbook — the grader holds the whole
workbook in memory and rewrites it after every question, so two writers
silently destroy each other's grades.

The published runs used vLLM 0.19.1. Grading is not bitwise reproducible:
generation is unconstrained, and JSON is parsed leniently.

## Licence and intended use

Released under **CC BY-NC 4.0** ([Creative Commons Attribution-NonCommercial 4.0
International](https://creativecommons.org/licenses/by-nc/4.0/)): use, share and adapt it for
non-commercial purposes with credit to the source. The student work was contributed with the
course instructors' permission and is released for research on automated grading. Do not
attempt to re-identify students or TAs.
