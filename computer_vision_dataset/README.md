---
pretty_name: KAUST Academy Stage 3 Practical AI Exam — Submissions and Dual TA Grades
license: cc-by-nc-4.0
language:
  - en
size_categories:
  - 1K<n<10K
task_categories:
  - text-classification
  - text-generation
tags:
  - automated-grading
  - llm-as-a-judge
  - code-assessment
  - education
  - jupyter-notebooks
  - human-annotation
  - inter-annotator-agreement
---

# KAUST Academy Stage 3 Practical AI Exam

570 student submissions to a four-question practical deep-learning exam, each independently
graded by **two** teaching assistants. It is the ground truth behind *Where LLM Graders Succeed
and Break: Evidence from Two Computer-Science Exams*, a study of LLM-based auto-grading, and it
exists to answer a question most auto-grading benchmarks cannot: **how good is good enough?**
Because every paper carries two independent human scores, the dataset ships with its own
measured human floor.

The headline use: on total score, two human TAs disagree by **2.61 / 35 points** on average.
A grader whose mean absolute error against the TA average is at or below 2.61 makes, in
aggregate, no more error than the two TAs make against each other. The comparison is not
like-for-like — averaging two TAs cancels part of their noise, which favours the grader under
test — so the accompanying paper decides parity with a paired third-grader test instead.

## What is here

| Path | Contents |
|---|---|
| `submissions_extracted/<1–570>/Q<n>.ipynb` | 1,881 student notebooks, one folder per student |
| `Practical_AI_exam_grades.xlsx` | Ground truth: per-question and total scores from both TAs |
| `Questions/Q<n>_stage3.ipynb` | The four exam questions as handed to students |
| `Solutions/` | Reference solution notebooks |
| `stage_3_rubrics_ta.md` | The per-question rubric the TAs graded against |
| `exam_guidelines_student.md` | Student-facing exam guidelines: allowed and prohibited resources for the practical exam (course materials and notes allowed; pre-written code and AI tools banned), the per-task partial-credit scale, and the rules of the separate theory exam |
| `few_shot_examples.json` | K=2 worked examples per question for few-shot prompting, from 5 students; their scores and rationales are the D01 model's (`gemini-3-flash-preview`) outputs, not TA marks |

Not every student attempted every question, so notebook counts vary:

| Notebooks submitted | Students |
|---|---|
| 4 | 180 |
| 3 | 383 |
| 2 | 5 |
| 1 | 2 |

Q4 is bonus-only, which is why just 180 of 570 students attempted it.

## The exam

35 points, plus 13 bonus points.

| Q | Topic | Points | Bonus |
|---|---|---|---|
| Q1 | Handwritten character classification via transfer learning | 12 | 4 |
| Q2 | Potato disease classification with a CNN | 11 | 4 |
| Q3 | Multi-class segmentation of underwater imagery | 12 | — |
| Q4 | Custom dataset for image colorization (bonus question) | — | 5 |

## Ground truth

`Practical_AI_exam_grades.xlsx` has one row per student and 22 columns: `Number`,
`TA_1_ID`, `TA_2_ID`, per-question score and bonus for each TA, per-TA totals, the
TA-averaged `Average Questions` and `Average Bonus`, and `Earned Points` (a 25-point course
grade: Average Questions/35×20 + Average Bonus/13×5).

- **570** students, all dual-graded
- **20** TAs in **10** fixed pairs; each pair graded ~55–60 students, and each student was
  graded by exactly one pair
- TA-averaged total: mean **26.04**, median **27.45**, range 0–35
- Per-question marks can be negative (the guidelines' −50% penalty) and a few exceed the
  question maximum, so per-TA totals run from −1.8 to 35.5. `TA 2 - Q4 Bonus` is blank for
  students 195 and 197; the workbook's formula totals treat blanks as 0.

### ⚠️ `TA_1_ID` and `TA_2_ID` are slots, not people

Because the pairs are fixed, slot 1 always holds one member of a pair and slot 2 the other —
but *which* person occupies slot 1 differs from pair to pair. A signed `TA_1 − TA_2`
difference pooled across pairs is therefore **uninterpretable**. Report the magnitude of
disagreement, or the correlation, or restrict signed analyses to within a single pair.

### Measured human floor

Pooled across all 570 dual-graded students:

| Metric | Value |
|---|---|
| Inter-grader MAE on total score | **2.61** / 35 (95% bootstrap CI [2.37, 2.85]) |
| Pearson *r* between the two TAs | 0.868 |
| Largest single-paper disagreement | 24.1 points |
| Inter-grader MAE on bonus | 1.71 over the 213 students to whom at least one TA awarded bonus (0.64 over all 570) |

Disagreement is concentrated but has a heavy tail — 184 papers differ by less than 1 point,
while 12 differ by 11 points or more:

| \|TA₁ − TA₂\| | Students |
|---|---|
| [0, 1) | 184 |
| [1, 2) | 105 |
| [2, 3) | 91 |
| [3, 5) | 95 |
| [5, 8) | 58 |
| [8, 11) | 25 |
| [11, 35) | 12 |

For reference, the strongest model configuration in the accompanying study
(`gemini-3-flash-preview`, neutral persona, temperature 0, default prompt with thinking off) reaches MAE
**1.64** [1.51, 1.79] against the TA average across all 570 students — below the human floor.

## Loading

The notebooks are plain files, so pull them directly rather than through `load_dataset`
(the Hub's dataset viewer does not render `.ipynb` or `.xlsx`):

```python
from huggingface_hub import snapshot_download
import pandas as pd

path = snapshot_download("KAUSTAcademy/AutoGrader_Computer_Vision", repo_type="dataset")
grades = pd.read_excel(f"{path}/Practical_AI_exam_grades.xlsx")

# One student's Q1 notebook
import json
nb = json.load(open(f"{path}/submissions_extracted/1/Q1.ipynb", encoding="utf-8"))
code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
```

To grab only the ground truth without the 1,881 notebooks:

```python
from huggingface_hub import hf_hub_download
xlsx = hf_hub_download("KAUSTAcademy/AutoGrader_Computer_Vision", "Practical_AI_exam_grades.xlsx",
                       repo_type="dataset")
```

## Collection and privacy

Submissions come from a real cohort sitting the KAUST Academy Stage 3 practical exam.
Grading was done by the course's TAs against `stage_3_rubrics_ta.md` as part of normal
assessment, not collected for research after the fact.

The data is pseudonymized before release. Students appear only as integers 1–570 and TAs only
as `TA_1`…`TA_20`. Student notebooks hold code cells only, and were scanned for email
addresses, local filesystem paths containing usernames, self-identifying comments ("submitted
by…"), and Google Drive or Colab account references; all four scans came back empty.

Even so, this is real student coursework. **Code is idiosyncratic and is not guaranteed to be
non-identifying** — an unusual variable name or comment could in principle be traced by
someone with access to the original cohort. Treat it as sensitive:

- Do not attempt to re-identify students or TAs.
- Report aggregates, not individual submissions — avoid singling out or quoting one student's
  work in anything you publish.
- The licence below permits non-commercial redistribution; these two requests are ethical
  conditions of use rather than licence terms, and we ask that you pass them on if you
  redistribute.

## Licence

Released under **CC BY-NC 4.0** ([Creative Commons Attribution-NonCommercial 4.0
International](https://creativecommons.org/licenses/by-nc/4.0/)). You may use, share and adapt
the contents, including the raw notebooks, for non-commercial purposes, provided you credit
the source (see the citation below). The student work was contributed with the course
instructors' permission and is released for research on automated grading.

## Known limitations

- **Scores are not a gold standard.** They are two human opinions, and the 2.61-point spread
  is the honest measure of how much they can differ. Use the TA average as the target and
  treat the disagreement as irreducible noise, not error to be optimized away.
- **Unbalanced questions.** Q4 has 180 attempts to Q1's 570; per-question conclusions about
  Q4 rest on a much smaller and self-selected sample.
- **One cohort, one exam, one language.** Everything comes from a single sitting of a single
  course in English. Generalization to other rubrics, subjects, or student populations is
  untested.
- **`few_shot_examples.json` overlaps the evaluation set and carries model labels.** Its
  examples come from 5 students who remain in the 570, and their scores and rationales are
  D01's (`gemini-3-flash-preview`) outputs, not TA grades, so few-shot runs that use it distil
  D01 in context rather than learn from held-out human exemplars. (The file's provenance
  strings give D01's MAE as 2.25; the D01 run reported in the paper scores 1.64.)
- **Student notebooks hold code cells only**: no markdown cells, stored outputs or execution
  counts, so a submission can be judged from its code alone, not from what it printed.

## Citation

```bibtex
@misc{habibullah2026llmgraderssucceedbreak,
      title={Where LLM Graders Succeed and Break: Evidence from Two Computer-Science Exams},
      author={Ali Habibullah and Yazan Alshoibi and Mohammad Alshiekh and Salman Khan and Naeemullah Khan},
      year={2026},
      eprint={2609.29333},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2609.29333},
}
```
