#!/usr/bin/env python3
"""Build breakdown-style LoRA fine-tuning data by DISTILLING Gemini D01.

Unlike ``build_finetune_data.py`` (marks-only), this trains the model to emit
the FULL grading JSON the real ``bd1`` grader uses::

    {"score", "bonus", "task_breakdown": [{"task","awarded","max","note"}...],
     "reasoning"}

Supervision signal = **Gemini D01's own graded outputs** (model
``gemini-3-flash-preview``, sol1/gd1/rs0/bd1/neutral). D01 already graded every
student and its per-task breakdown is stored (as text) in the reasoning column
of its result workbook, so we reconstruct clean structured targets from there --
no new API calls. We distil D01 *faithfully* (score/bonus/breakdown copied as-is,
NOT rescaled to the TA marks) because D01 tracks the TA-average (MAE 1.64) more
closely than the two human TAs agree with each other (2.61): the TA labels are
the noisier signal, so imitating D01 gives the cleaner, higher target.

The train/eval split is taken verbatim from ``finetune/data/split_meta.json``
(same seed-fixed students as the marks-only run) so the two experiments are
compared on the identical held-out students.

Outputs (under ``--out-dir``, default ``finetune/data_bd``):
  * ``train.jsonl``  -- conversational prompt/completion rows for TRL.
  * ``eval.jsonl``   -- same shape for the held-out students (parity/inspection).
  * ``eval_students.json`` / ``split_meta.json`` -- copied through for eval_lora.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import openpyxl

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))  # vendored common.py lives beside this script

from common import (  # noqa: E402
    QUESTION_META,
    STRICTNESS_PRESETS,
    SOLUTIONS_DIR,
    EXTRACTED_DIR,
    GUIDELINES,
    RESULTS_DIR,
    build_prompt,
    load_rubric_sections,
    notebook_to_text,
)
from build_finetune_data import truncate_code, GRADING_GUIDANCE  # noqa: E402
import exams  # noqa: E402
from exams import ExamSpec  # noqa: E402


def default_out_dir(dataset: str) -> Path:
    """cv keeps finetune/data_bd so the existing bd split stays put."""
    return HERE / ("data_bd" if dataset == "cv" else f"data_bd_{dataset}")


def default_split_meta(dataset: str) -> Path:
    """bd reuses the marks split verbatim, so the two share held-out students."""
    return HERE / ("data" if dataset == "cv" else f"data_{dataset}") / "split_meta.json"


DEFAULT_OUT = HERE / "data_bd"
DEFAULT_SPLIT = HERE / "data" / "split_meta.json"
# The D01 teacher workbook is an ABLATION-STUDY result (an input we distill
# from), so it lives in the repo-root computer_vision_results/ -- NOT in finetune/results/,
# which only holds this pipeline's own outputs.
DEFAULT_TEACHER = (
    ROOT / "computer_vision_results" / "results"
    / "D01__D01_m-3-flash-preview_sol1_gd1_rs0_bd1_str-neutral_t00_n1.xlsx"
)

# "- <task>: <awarded>/<max>[ — <note>]"
_LINE = re.compile(r"^- (?P<task>.+?): (?P<aw>-?[\d.]+)/(?P<mx>-?[\d.]+)(?: [—-]+ (?P<note>.*))?$")


def parse_teacher(xlsx: Path, exam: ExamSpec = None) -> dict[tuple[int, int], dict]:
    """(student, q) -> {'score','bonus','task_breakdown','reasoning'} from the teacher run."""
    exam = exam or exams.CV
    wb = openpyxl.load_workbook(xlsx, data_only=True)
    ws = wb.active
    H = [c.value for c in ws[1]]
    col = {h: i for i, h in enumerate(H) if h is not None}

    def num(v):
        return float(v) if isinstance(v, (int, float)) else None

    out: dict[tuple[int, int], dict] = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        sid = row[0]
        if not isinstance(sid, (int, float)):
            continue
        s = int(sid)
        for q in exam.questions:
            rc = col.get(f"AI Q{q} Reasoning")
            sc = col.get(f"AI Q{q} Score")
            bc = col.get(f"AI Q{q} Bonus")
            if rc is None:
                continue
            txt = row[rc]
            if not txt or "Breakdown:" not in str(txt):
                continue  # no submission / no breakdown -> nothing to distil
            reasoning, _, block = str(txt).partition("Breakdown:")
            tb = []
            for ln in block.splitlines():
                ln = ln.strip()
                m = _LINE.match(ln)
                if not m:
                    continue
                item = {
                    "task": m.group("task").strip(),
                    "awarded": float(m.group("aw")),
                    "max": float(m.group("mx")),
                }
                if m.group("note"):
                    item["note"] = m.group("note").strip()
                tb.append(item)
            if not tb:
                continue
            score = num(row[sc]) if sc is not None else None
            bonus = num(row[bc]) if bc is not None else 0.0
            out[(s, q)] = {
                "score": round(score if score is not None else 0.0, 2),
                "bonus": round(bonus if bonus is not None else 0.0, 2),
                "task_breakdown": tb,
                "reasoning": reasoning.strip(),
            }
    return out


def build_bd_prompt(
    q: int,
    rubric_section: str,
    guidelines: str,
    student_code: str,
    solution_code: str | None,
    use_guidelines: bool,
    strictness: str,
    exam: ExamSpec = None,
) -> str:
    """Breakdown-ONLY grading prompt: asks for {score, bonus, task_breakdown}
    and explicitly NOT for reasoning/prose.

    Same context assembly as ``computer_vision_grade_with_local.build_prompt`` (guidelines,
    rubric, reference solution, student code) and the same grading scale, but the
    output contract drops the ``reasoning`` field -- matching the no-reasoning
    distil targets. Shared by the data builder and ``eval_lora.py`` so train and
    inference prompts stay identical."""
    exam = exam or exams.CV
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
        "Return a JSON object with: `score` (number, <= max base), `bonus` (number, "
        "<= max bonus; 0 if no bonus tasks), and `task_breakdown` (array of per-task "
        "awards).",
        "`task_breakdown` MUST be non-empty and MUST contain ONE entry for EVERY task "
        "row in the rubric above (including bonus rows), each as "
        '{"task": <string>, "awarded": <number>, "max": <number>}. Sum the `awarded` '
        "values to get `score` (base tasks) and `bonus` (bonus tasks). Decide each task "
        "ONCE; do not revise.",
        "Do NOT include a `reasoning` field or any prose.",
        "OUTPUT RULES (strict):",
        "  - Output ONLY the JSON object. No prose before or after.",
        "  - Each task_breakdown entry appears EXACTLY ONCE. Never repeat a task.",
    ]
    # Same guidance block as the marks prompt and the published ablation prompt.
    parts += GRADING_GUIDANCE
    return "\n".join(parts)


def example_rows(students, teacher, rubric_sections, guidelines, solutions,
                 use_guidelines, strictness, max_student_chars, questions=(1, 2, 3, 4),
                 with_reasoning=False, exam: ExamSpec = None):
    exam = exam or exams.CV
    use_guidelines = use_guidelines and exam.has_guidelines
    rows, n_skip_nb, n_skip_teacher = [], 0, 0
    for s in students:
        sdir = exam.extracted_dir / str(s)
        for q in questions:
            tgt = teacher.get((s, q))
            if tgt is None:
                n_skip_teacher += 1
                continue
            nb = sdir / f"Q{q}.ipynb"
            if not nb.exists():
                n_skip_nb += 1
                continue
            code = truncate_code(notebook_to_text(nb), max_student_chars)
            if with_reasoning:
                prompt = build_prompt(
                    q=q,
                    rubric_section=rubric_sections[q],
                    guidelines=guidelines,
                    student_code=code,
                    solution_code=solutions[q],
                    use_breakdown=True,
                    use_guidelines=use_guidelines,
                    strictness=strictness,
                    exam=exam,
                )
            else:
                prompt = build_bd_prompt(
                    q, rubric_sections[q], guidelines, code,
                    solutions[q], use_guidelines, strictness, exam,
                )
            target = {
                "score": tgt["score"],
                "bonus": tgt["bonus"],
                "task_breakdown": tgt["task_breakdown"],
            }
            if with_reasoning:
                target["reasoning"] = tgt["reasoning"]
            completion = json.dumps(target, ensure_ascii=False)
            rows.append({
                "prompt": [{"role": "user", "content": prompt}],
                "completion": [{"role": "assistant", "content": completion}],
                "student": s,
                "question": q,
                "exam": exam.key,
            })
    if n_skip_nb or n_skip_teacher:
        print(f"  ({exam.key}: skipped {n_skip_nb} with no notebook, "
              f"{n_skip_teacher} with no teacher breakdown)")
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
                    help="Which exam(s) to build; 'both' pools the per-exam splits.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Default: finetune/data_bd for cv, finetune/data_bd_<dataset> otherwise.")
    ap.add_argument("--split-meta", type=Path, default=None,
                    help="Reuse the marks-only train/eval split for a fair comparison. "
                         "Default: the matching finetune/data[_<dataset>]/split_meta.json.")
    ap.add_argument("--teacher-xlsx", type=Path, default=None,
                    help="Teacher workbook to distil breakdowns from. Default: the "
                         "exam's bd1 neutral 3-flash-preview run. Single-exam only.")
    ap.add_argument("--no-guidelines", action="store_true")
    ap.add_argument("--strictness", default="neutral")
    ap.add_argument("--max-student-chars", type=int, default=24000)
    ap.add_argument("--train-questions", default=None,
                    help="Questions to put in TRAIN rows, e.g. '1,2,4' for a "
                         "leave-Q3-out generalization test.")
    ap.add_argument("--eval-questions", default=None,
                    help="Questions to put in EVAL rows, e.g. '3'.")
    ap.add_argument("--with-reasoning", action="store_true",
                    help="Also distil D01's `reasoning` prose into the target. Default "
                         "OFF: targets are breakdown-only {score, bonus, task_breakdown} "
                         "and the prompt explicitly forbids a reasoning field.")
    ap.add_argument("--students", choices=["split", "all"], default="split",
                    help="'split' = use split_meta train/eval students (holds out "
                         "students too). 'all' = every student for both (isolates the "
                         "question variable; the LOQO protocol).")
    args = ap.parse_args()

    out = args.out_dir or default_out_dir(args.dataset)
    split_meta_path = args.split_meta or default_split_meta(args.dataset)
    spanned = exams.exams_for(args.dataset)
    if args.teacher_xlsx and len(spanned) > 1:
        ap.error("--teacher-xlsx applies to a single exam; omit it for --dataset both")
    # --students all (the LOQO protocol) never uses the split; only require the
    # marks split when we actually hold students out.
    if args.students == "split" and not split_meta_path.exists():
        ap.error(f"{split_meta_path} not found -- run build_finetune_data.py "
                 f"--dataset {args.dataset} first, or pass --students all.")
    meta = (json.loads(split_meta_path.read_text())
            if split_meta_path.exists() else {})
    use_guidelines = not args.no_guidelines
    train_rows, eval_rows, per_exam = [], [], {}

    for exam in spanned:
        teacher_xlsx = args.teacher_xlsx or exam.teacher_xlsx
        if not teacher_xlsx.exists():
            ap.error(f"teacher workbook not found: {teacher_xlsx}")

        # Questions default to whatever the exam actually has (cv 1-4, intro 1-3);
        # an explicit --train/--eval-questions still drives the LOQO protocol.
        tq = ([int(x) for x in args.train_questions.split(",") if x.strip()]
              if args.train_questions else list(exam.questions))
        eq = ([int(x) for x in args.eval_questions.split(",") if x.strip()]
              if args.eval_questions else list(exam.questions))

        teacher = parse_teacher(teacher_xlsx, exam)
        print(f"[{exam.key}] parsed {len(teacher)} (student,q) breakdowns "
              f"from {teacher_xlsx.name}")

        if args.students == "all":
            all_students = sorted({s for (s, _q) in teacher})
            tr = ev = all_students
            print(f"[{exam.key}] students: ALL {len(all_students)} for both "
                  f"(LOQO: train Q{tq} -> test Q{eq}).")
        else:
            # Detect the meta's shape from its CONTENT, not from --dataset: a
            # mismatched --split-meta must error loudly, never silently build a
            # wrong or empty dataset. Preference order: per_exam[this exam],
            # then flat lists, then pooled flat dicts keyed by exam.
            if exam.key in meta.get("per_exam", {}):
                src = meta["per_exam"][exam.key]
            else:
                src = meta
            tr, ev = src.get("train_students"), src.get("eval_students")
            if isinstance(tr, dict):
                tr, ev = tr.get(exam.key), (ev or {}).get(exam.key)
            if not (isinstance(tr, list) and isinstance(ev, list) and tr and ev):
                ap.error(f"{split_meta_path} has no usable train/eval student "
                         f"lists for exam '{exam.key}' -- was it built with a "
                         f"different --dataset?")
            print(f"[{exam.key}] split from {split_meta_path.name}: "
                  f"{len(tr)} train / {len(ev)} eval students (train Q{tq} -> test Q{eq}).")

        rubric_sections = exams.load_rubric_sections(exam)
        guidelines = exams.load_guidelines(exam)
        solutions = exams.load_solutions(exam, True, notebook_to_text)

        print(f"[{exam.key}] target: breakdown"
              f"{' + reasoning' if args.with_reasoning else ' ONLY (no reasoning)'}")
        tr_rows = example_rows(tr, teacher, rubric_sections, guidelines, solutions,
                               use_guidelines, args.strictness, args.max_student_chars,
                               questions=tq, with_reasoning=args.with_reasoning, exam=exam)
        ev_rows = example_rows(ev, teacher, rubric_sections, guidelines, solutions,
                               use_guidelines, args.strictness, args.max_student_chars,
                               questions=eq, with_reasoning=args.with_reasoning, exam=exam)
        train_rows += tr_rows
        eval_rows += ev_rows
        per_exam[exam.key] = {
            "teacher_xlsx": teacher_xlsx.name,
            "train_questions": tq, "eval_questions": eq,
            "n_train_students": len(tr), "n_eval_students": len(ev),
            "n_train_examples": len(tr_rows), "n_eval_examples": len(ev_rows),
            "used_guidelines": use_guidelines and exam.has_guidelines,
            "eval_students": ev,
        }

    if len(per_exam) == 1:
        (key,) = per_exam
        eval_students_payload = per_exam[key]["eval_students"]
    else:
        eval_students_payload = {k: v["eval_students"] for k, v in per_exam.items()}

    write_jsonl(out / "train.jsonl", train_rows)
    write_jsonl(out / "eval.jsonl", eval_rows)
    (out / "eval_students.json").write_text(
        json.dumps(eval_students_payload), encoding="utf-8")
    stale = {"per_exam"} | ({"train_students", "eval_students"}
                            if args.students == "all" else set())
    (out / "split_meta.json").write_text(json.dumps({
        **{k: v for k, v in meta.items() if k not in stale},
        "dataset": args.dataset,
        "target": ("breakdown-distill" if args.with_reasoning
                   else "breakdown-only-distill"),
        "with_reasoning": args.with_reasoning,
        "students_mode": args.students,
        "n_train_examples": len(train_rows),
        "n_eval_examples": len(eval_rows),
        "per_exam": per_exam,
    }, indent=2), encoding="utf-8")

    print(f"\nWrote:")
    print(f"  {out/'train.jsonl'}   ({len(train_rows)} examples)")
    print(f"  {out/'eval.jsonl'}    ({len(eval_rows)} examples)")
    print(f"  {out/'split_meta.json'}")


if __name__ == "__main__":
    main()
