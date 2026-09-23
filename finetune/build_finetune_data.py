#!/usr/bin/env python3
"""Build LoRA fine-tuning data for the auto-grader.

Supervision signal = TA MARKS ONLY. For every (student, question) we take the
per-question TA scores from ``computer_vision_dataset/Practical_AI_exam_grades.xlsx`` (both TAs,
averaged) as the gold target. We deliberately DO NOT use any model-generated
reasoning or task breakdown as a target -- the model is fine-tuned purely to
predict the number of marks.

The split is *dynamic* and *seed-fixed*: students are shuffled with
``--seed`` and divided into train / eval by ``--train-frac``. The eval student
list is written out so ``eval_lora.py`` grades exactly the held-out students.

The prompt is assembled with the SAME rubric / solution / guidelines pieces the
real grader uses (imported from ``computer_vision_grade_with_local``), but the task instruction
asks for scores only -- matching what the fine-tuned model will be asked to
produce at inference time.

Outputs (under ``--out-dir``, default ``finetune/data``):
  * ``train.jsonl``        -- conversational prompt/completion rows for TRL.
  * ``eval.jsonl``         -- same shape for the held-out students (optional eval-on-text).
  * ``eval_students.json`` -- list[int] of held-out student numbers.
  * ``split_meta.json``    -- seed, fractions, counts (for reproducibility).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import openpyxl

# Reuse the exact context-assembly the real grader uses so training prompts
# match inference prompts byte-for-byte (minus the task instruction).
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))  # vendored common.py lives beside this script

from common import (  # noqa: E402
    QUESTION_META,
    STRICTNESS_PRESETS,
    SOLUTIONS_DIR,
    EXTRACTED_DIR,
    GRADES_XLSX,
    GUIDELINES,
    load_rubric_sections,
    notebook_to_text,
)
import exams  # noqa: E402
from exams import ExamSpec  # noqa: E402


def default_out_dir(dataset: str) -> Path:
    """cv keeps the original finetune/data path so the pinned split is stable."""
    return HERE / ("data" if dataset == "cv" else f"data_{dataset}")


DEFAULT_OUT = HERE / "data"

# Grading guidance copied VERBATIM from ``computer_vision_grade_with_local.build_prompt`` (the
# prompt behind the published L-* / D01 ablations), so the fine-tuning prompts
# stay continuous with the grader the paper describes. Both the marks and the bd
# prompt append this same block, byte-identical, which keeps the marks-vs-bd
# comparison clean: the only difference between them is the output contract.
# NOTE: verbatim copy -- if build_prompt's wording changes, mirror it here.
GRADING_GUIDANCE = [
    "GRADING SCALE (strict):",
    "  - 100% of a task: code is correct and runs.",
    "  - 25-75% of a task: code is attempted with syntax/logic mistakes.",
    "  - 0% of a task: NO CODE was written for that task. This is the ONLY case for 0.",
    "  - -50% of a task: code uses an entirely wrong approach OR is blind copy-paste "
    "    OR disregards explicit instructions (e.g. wrong dataset).",
    "IMPORTANT: A submission that is disorganized, fragmented, uses a different "
    "structure than the reference, or fails to run end-to-end is NOT automatically 0. "
    "Grade each task individually: if the student wrote code that attempts the task, "
    "award partial credit even when other parts are broken. Only assign 0 to a task "
    "when there is literally no code for it.",
]


def truncate_code(code: str, max_chars: int) -> str:
    """Keep head + tail of a long notebook so the prompt fits the train context.

    Notebooks tend to carry setup at the top and results/answers at the bottom,
    so we keep both ends and drop the middle."""
    if max_chars <= 0 or len(code) <= max_chars:
        return code
    head = int(max_chars * 0.6)
    tail = max_chars - head
    return (
        code[:head]
        + "\n\n--- [TRUNCATED FOR LENGTH; middle of submission omitted] ---\n\n"
        + code[-tail:]
    )


def build_scores_prompt(
    q: int,
    rubric_section: str,
    guidelines: str,
    student_code: str,
    solution_code: str | None,
    use_guidelines: bool,
    strictness: str,
    exam: ExamSpec = exams.CV,
) -> str:
    """Prompt that asks ONLY for {score, bonus} -- no reasoning, no breakdown.

    ``exam`` defaults to CV so existing callers keep byte-identical prompts.
    The caps are written into the prompt, which is what lets an adapter trained
    on one exam grade the other coherently.
    """
    max_score, max_bonus, _ = exam.question_meta[q]
    persona = STRICTNESS_PRESETS.get(strictness, STRICTNESS_PRESETS["neutral"])
    parts = [
        f"{persona} You are grading Question {q} of the {exam.label} exam.",
        "",
    ]
    if use_guidelines:
        parts += ["## EXAM GUIDELINES (provided to students)", guidelines, ""]
    parts += [f"## RUBRIC FOR QUESTION {q}", rubric_section, ""]
    if solution_code is not None:
        parts += [
            "## REFERENCE SOLUTION (instructor's notebook)",
            "```python",
            solution_code,
            "```",
            "",
        ]
    parts += [
        "## STUDENT SUBMISSION (the one you must grade)",
        "```python",
        student_code,
        "```",
        "",
        "## YOUR TASK",
        f"Grade strictly against the rubric. Max base score for Q{q} is {max_score}; "
        f"max bonus is {max_bonus}.",
        "Return ONLY a JSON object with two numeric fields: `score` (<= max base) "
        "and `bonus` (<= max bonus; 0 if there are no bonus tasks). "
        "Do NOT include any reasoning, explanation, or task breakdown.",
        'Output format: {"score": <number>, "bonus": <number>}',
        "OUTPUT RULES (strict):",
        "  - Output ONLY the JSON object. No prose before or after.",
    ]
    parts += GRADING_GUIDANCE
    return "\n".join(parts)


def _num(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) else None


def load_ta_targets(grades_xlsx: Path) -> dict[int, dict[int, dict[str, float]]]:
    """student -> q -> {'score': float, 'bonus': float} (TA1/TA2 averaged, clamped).

    A question's score/bonus is None-skipped only when NO TA value exists for a
    component that the question actually has (Q4 has no base score; Q3 has no
    bonus -- those are reported as 0.0, not skipped)."""
    wb = openpyxl.load_workbook(grades_xlsx, data_only=True)
    ws = wb.active
    headers = [c.value for c in ws[1]]
    col = {h: i for i, h in enumerate(headers) if h is not None}

    out: dict[int, dict[int, dict[str, float]]] = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        num = row[0]
        if not isinstance(num, (int, float)):
            continue
        student = int(num)
        per_q: dict[int, dict[str, float]] = {}
        for q in (1, 2, 3, 4):
            max_score, max_bonus, _ = QUESTION_META[q]

            def avg(field: str) -> float | None:
                vals = []
                for ta in (1, 2):
                    key = f"TA {ta} - Q{q} {field}"
                    if key in col:
                        v = _num(row[col[key]])
                        if v is not None:
                            vals.append(v)
                return sum(vals) / len(vals) if vals else None

            score = avg("Score") if max_score > 0 else 0.0
            bonus = avg("Bonus") if max_bonus > 0 else 0.0
            if max_score > 0 and score is None:
                # No usable TA mark for a question that needs one -> skip it.
                continue
            score = max(0.0, min(score if score is not None else 0.0, max_score))
            bonus = max(0.0, min(bonus if bonus is not None else 0.0, max_bonus))
            per_q[q] = {"score": round(score, 2), "bonus": round(bonus, 2)}
        if per_q:
            out[student] = per_q
    return out


def make_split(students: list[int], train_frac: float, seed: int) -> tuple[list[int], list[int]]:
    ordered = sorted(students)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n_train = round(len(ordered) * train_frac)
    train = sorted(ordered[:n_train])
    eval_ = sorted(ordered[n_train:])
    return train, eval_


def example_rows(
    students: list[int],
    targets: dict[int, dict[int, dict[str, float]]],
    rubric_sections: dict[int, str],
    guidelines: str,
    use_solution: bool,
    use_guidelines: bool,
    strictness: str,
    max_student_chars: int,
    exam: ExamSpec = exams.CV,
) -> list[dict]:
    """Conversational prompt/completion rows (TRL format). One row per (student, q).

    Rows carry their ``exam`` key so a pooled ('both') dataset stays traceable
    back to the exam each example came from.
    """
    solutions = exams.load_solutions(exam, use_solution, notebook_to_text)
    # intro ships no guidelines document; never emit an empty guidelines block.
    use_guidelines = use_guidelines and exam.has_guidelines

    rows: list[dict] = []
    n_skipped = 0
    for s in students:
        sdir = exam.extracted_dir / str(s)
        for q in exam.questions:
            tgt = targets.get(s, {}).get(q)
            if tgt is None:
                continue
            nb = sdir / f"Q{q}.ipynb"
            if not nb.exists():
                # No submission -> the grader scores 0; nothing to learn from
                # an empty code block, so skip (TA total still uses the 0).
                n_skipped += 1
                continue
            code = truncate_code(notebook_to_text(nb), max_student_chars)
            prompt = build_scores_prompt(
                q, rubric_sections[q], guidelines, code,
                solutions[q], use_guidelines, strictness, exam,
            )
            completion = json.dumps({"score": tgt["score"], "bonus": tgt["bonus"]})
            rows.append({
                "prompt": [{"role": "user", "content": prompt}],
                "completion": [{"role": "assistant", "content": completion}],
                "student": s,
                "question": q,
                "exam": exam.key,
            })
    if n_skipped:
        print(f"  ({exam.key}: skipped {n_skipped} (student,q) cells with no submission notebook)")
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=list(exams.DATASET_CHOICES), default="cv",
                    help="Which exam(s) to build. 'both' pools the two per-exam "
                         "80/20 splits, so each exam's held-out students are "
                         "exactly the ones its own single-exam build holds out.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Default: finetune/data for cv, finetune/data_<dataset> otherwise.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for the dynamic train/eval split (fixed -> reproducible).")
    ap.add_argument("--train-frac", type=float, default=0.8,
                    help="Fraction of students used for training (rest is held out).")
    ap.add_argument("--no-solution", action="store_true",
                    help="Exclude the reference solution from prompts (default: include).")
    ap.add_argument("--no-guidelines", action="store_true",
                    help="Exclude exam guidelines from prompts (default: include).")
    ap.add_argument("--strictness", choices=list(STRICTNESS_PRESETS), default="neutral")
    ap.add_argument("--max-student-chars", type=int, default=24000,
                    help="Cap student-code chars so prompts fit the train context "
                         "(head+tail kept). 0 = no cap.")
    args = ap.parse_args()

    if not (0.0 < args.train_frac < 1.0):
        ap.error("--train-frac must be in (0, 1)")

    out = args.out_dir or default_out_dir(args.dataset)
    use_solution = not args.no_solution
    use_guidelines = not args.no_guidelines

    train_rows: list[dict] = []
    eval_rows: list[dict] = []
    per_exam: dict[str, dict] = {}

    # Split each exam independently on the same seed, then pool. Student numbers
    # restart at 1 in each exam, so the split must stay exam-qualified -- pooling
    # the raw ints would collide 570 CV students onto 570 intro ones.
    for exam in exams.exams_for(args.dataset):
        rubric_sections = exams.load_rubric_sections(exam)
        guidelines = exams.load_guidelines(exam)
        targets = exams.load_ta_targets(exam)

        gradable = sorted(s for s in targets if (exam.extracted_dir / str(s)).is_dir())
        tr, ev = make_split(gradable, args.train_frac, args.seed)
        print(f"[{exam.key}] students with TA marks + submissions: {len(gradable)}")
        print(f"  seed={args.seed} train_frac={args.train_frac} "
              f"-> {len(tr)} train / {len(ev)} eval")

        print(f"[{exam.key}] building train rows...")
        tr_rows = example_rows(tr, targets, rubric_sections, guidelines,
                               use_solution, use_guidelines, args.strictness,
                               args.max_student_chars, exam)
        print(f"[{exam.key}] building eval rows...")
        ev_rows = example_rows(ev, targets, rubric_sections, guidelines,
                               use_solution, use_guidelines, args.strictness,
                               args.max_student_chars, exam)
        train_rows += tr_rows
        eval_rows += ev_rows
        per_exam[exam.key] = {
            "n_students_total": len(gradable),
            "n_train_students": len(tr), "n_eval_students": len(ev),
            "n_train_examples": len(tr_rows), "n_eval_examples": len(ev_rows),
            "questions": list(exam.questions),
            "max_base": exam.total_base, "max_bonus": exam.total_bonus,
            "used_guidelines": use_guidelines and exam.has_guidelines,
            "train_students": tr, "eval_students": ev,
        }

    # Single exam -> plain list[int], unchanged on disk so the pinned cv split
    # and everything reading it keep working. Pooled -> {exam: [students]}.
    if len(per_exam) == 1:
        (key,) = per_exam
        eval_students_payload = per_exam[key]["eval_students"]
        # Top-level lists kept for build_breakdown_data.py, which reads
        # meta["train_students"] / meta["eval_students"] verbatim.
        flat = {"train_students": per_exam[key]["train_students"],
                "eval_students": per_exam[key]["eval_students"]}
    else:
        eval_students_payload = {k: v["eval_students"] for k, v in per_exam.items()}
        flat = {"train_students": {k: v["train_students"] for k, v in per_exam.items()},
                "eval_students": {k: v["eval_students"] for k, v in per_exam.items()}}

    write_jsonl(out / "train.jsonl", train_rows)
    write_jsonl(out / "eval.jsonl", eval_rows)
    (out / "eval_students.json").write_text(
        json.dumps(eval_students_payload), encoding="utf-8")
    (out / "split_meta.json").write_text(json.dumps({
        "dataset": args.dataset,
        "exams": sorted(per_exam),
        "seed": args.seed,
        "train_frac": args.train_frac,
        "n_students_total": sum(v["n_students_total"] for v in per_exam.values()),
        "n_train_students": sum(v["n_train_students"] for v in per_exam.values()),
        "n_eval_students": sum(v["n_eval_students"] for v in per_exam.values()),
        "n_train_examples": len(train_rows),
        "n_eval_examples": len(eval_rows),
        "use_solution": use_solution,
        "use_guidelines": use_guidelines,
        "strictness": args.strictness,
        "max_student_chars": args.max_student_chars,
        **flat,
        "per_exam": per_exam,
    }, indent=2), encoding="utf-8")

    print(f"\nWrote ({args.dataset}):")
    print(f"  {out/'train.jsonl'}   ({len(train_rows)} examples)")
    print(f"  {out/'eval.jsonl'}    ({len(eval_rows)} examples)")
    print(f"  {out/'eval_students.json'}")
    print(f"  {out/'split_meta.json'}")


if __name__ == "__main__":
    main()
