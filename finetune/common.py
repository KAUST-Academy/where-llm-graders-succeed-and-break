"""Vendored from computer_vision_grade_with_local.py (2026-07-22) so finetune/ is
self-contained: no imports from repo-root modules -> clean merges.
Includes the compute_run_summary fix (TA totals rebuilt from per-question
columns when the xlsx total cells are formulas). If root prompts ever
change, this copy does NOT follow them -- frozen for the paper."""
from __future__ import annotations

import json
import logging
import os
import re
import statistics
from pathlib import Path
from typing import Optional
import openpyxl

from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent          # repo root: read-only original-data inputs live here
RESULTS_DIR = HERE / "results"   # finetune outputs stay inside finetune/

def _parse_json_lenient(text: str) -> dict:
    """Try strict json -> non-strict (allow control chars) -> json_repair.
    Raises json.JSONDecodeError if every layer fails."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError as e:
        if not _HAS_JSON_REPAIR:
            raise
        try:
            repaired = json_repair.loads(text)
            if isinstance(repaired, dict):
                return repaired
            raise json.JSONDecodeError("json_repair returned non-dict", text, 0)
        except Exception:  # noqa: BLE001
            raise e


GUIDELINES = ROOT / "computer_vision_dataset" / "exam_guidelines_student.md"


RUBRIC = ROOT / "computer_vision_dataset" / "stage_3_rubrics_ta.md"


SOLUTIONS_DIR = ROOT / "computer_vision_dataset" / "Solutions"


EXTRACTED_DIR = ROOT / "computer_vision_dataset" / "submissions_extracted"


GRADES_XLSX = ROOT / "computer_vision_dataset" / "Practical_AI_exam_grades.xlsx"


STRICTNESS_PRESETS = {
    "strict": (
        "You are a HARSH teaching assistant. Award the MINIMUM defensible score "
        "for any task that is incomplete, buggy, or deviates from the rubric. "
        "Never give partial credit if the task does not run correctly."
    ),
    "rigorous": (
        "You are a RIGOROUS teaching assistant. Apply the rubric with academic "
        "precision. Award points strictly according to the rubric criteria; do "
        "not inflate scores out of sympathy for effort or partial attempts."
    ),
    "exacting": (
        "You are an EXACTING teaching assistant. Demand demonstrable correctness. "
        "Award credit only when the implementation precisely matches the rubric "
        "requirement; small deviations from the expected behavior are penalized."
    ),
    "neutral": (
        "You are a strict but fair teaching assistant."
    ),
    "lenient": (
        "You are a GENEROUS teaching assistant. Give the student the benefit of "
        "the doubt. Award the MAXIMUM defensible score whenever a reasonable "
        "attempt at the task is visible, even if the code is buggy or incomplete."
    ),
}


QUESTION_META = {
    1: (12.0, 4.0, "Q1_stage3_Finetuning_Solution.ipynb"),
    2: (11.0, 4.0, "Q2_stage3_CNN_Solution.ipynb"),
    3: (12.0, 0.0, "Q3_stage3_Segmentation_Solution.ipynb"),
    4: (0.0, 5.0, "Q4_stage3_Colorization_Solution.ipynb"),
}


def ai_headers(exam=None) -> list[str]:
    """AI output columns. exam=None -> cv, so existing callers are unchanged.
    intro has three questions, so its workbook must not gain an AI Q4 column."""
    if exam is None:
        import exams as _exams
        exam = _exams.CV
    cols: list[str] = []
    for q in exam.questions:
        cols.append(f"AI Q{q} Score")
        if exam.question_meta[q][1] > 0:
            cols.append(f"AI Q{q} Bonus")
        cols.append(f"AI Q{q} Reasoning")
    cols += ["AI Total Score", "AI Total Bonus"]
    return cols


def load_rubric_sections() -> dict[int, str]:
    text = RUBRIC.read_text(encoding="utf-8")
    parts = re.split(r"(?m)^(##\s+.*)$", text)
    sections: dict[int, str] = {}
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        m = re.match(r"##\s+(?:Bonus\s+)?Question\s+(\d+)", header, re.IGNORECASE)
        if not m:
            continue
        q = int(m.group(1))
        sections[q] = header + "\n" + body.strip()
    missing = [q for q in (1, 2, 3, 4) if q not in sections]
    if missing:
        raise RuntimeError(f"Rubric sections missing for: {missing}")
    return sections


def notebook_to_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    out: list[str] = []
    for idx, cell in enumerate(nb.get("cells", []), start=1):
        src = "".join(cell.get("source", []))
        ctype = cell.get("cell_type", "code")
        out.append(f"--- Cell {idx} [{ctype}] ---\n{src}\n")
    return "\n".join(out)


def render_few_shot_block(examples: list[dict], q: int, use_breakdown: bool) -> str:
    if not examples:
        return ""
    lines = ["## WORKED EXAMPLES (calibrate to these prior gradings)"]
    for i, ex in enumerate(examples, 1):
        prov = ex.get("provenance", {})
        lines += [
            "",
            f"### Example {i}",
            f"(Sourced from a prior submission with strong TA agreement; "
            f"TA-avg total = {prov.get('ta_avg_total', '?')}/35.)",
            "",
            "Student code:",
            "```python",
            ex["code"],
            "```",
            "",
            "Correct JSON response for this example:",
            "```json",
        ]
        obj = {
            "score": ex["score"],
            "bonus": ex.get("bonus", 0.0),
            "reasoning": ex["reasoning"],
        }
        if use_breakdown:
            obj["task_breakdown"] = (
                "[…omitted in this example; you MUST still produce one for the "
                "live grading below.]"
            )
        lines.append(json.dumps(obj, indent=2))
        lines.append("```")
    lines.append("")
    return "\n".join(lines)


def atomic_save(wb: openpyxl.Workbook, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    wb.save(tmp)
    os.replace(tmp, path)


def ensure_ai_columns(ws, exam=None) -> dict[str, int]:
    headers = [c.value for c in ws[1]]
    name_to_col: dict[str, int] = {h: i + 1 for i, h in enumerate(headers) if h}
    next_col = (max(name_to_col.values()) + 1) if name_to_col else 1
    for h in ai_headers(exam):
        if h not in name_to_col:
            ws.cell(row=1, column=next_col, value=h)
            name_to_col[h] = next_col
            next_col += 1
    return name_to_col


def build_student_row_index(ws) -> dict[int, int]:
    idx: dict[int, int] = {}
    for row in ws.iter_rows(min_row=2):
        v = row[0].value
        if v is None:
            continue
        try:
            idx[int(v)] = row[0].row
        except (TypeError, ValueError):
            continue
    return idx


SCHEMA_WITH_BREAKDOWN = {
    "type": "object",
    "properties": {
        "score": {"type": "number"},
        "bonus": {"type": "number"},
        "task_breakdown": {
            "type": "array",
            "minItems": 1,
            "maxItems": 30,
            "items": {
                "type": "object",
                "properties": {
                    "task":    {"type": "string", "maxLength": 200},
                    "awarded": {"type": "number"},
                    "max":     {"type": "number"},
                    "note":    {"type": "string", "maxLength": 300},
                },
                "required": ["task", "awarded", "max"],
            },
        },
        "reasoning": {"type": "string", "maxLength": 1500},
    },
    "required": ["score", "bonus", "task_breakdown", "reasoning"],
}


def build_prompt(
    q: int,
    rubric_section: str,
    guidelines: str,
    student_code: str,
    solution_code: Optional[str],
    use_breakdown: bool = True,
    use_guidelines: bool = True,
    strictness: str = "neutral",
    few_shot_examples: Optional[list[dict]] = None,
    exam=None,
) -> str:
    # exam defaults to cv so existing callers keep byte-identical prompts.
    if exam is None:
        import exams as _exams
        exam = _exams.CV
    max_score, max_bonus, _ = exam.question_meta[q]
    persona = STRICTNESS_PRESETS.get(strictness, STRICTNESS_PRESETS["neutral"])
    parts = [
        f"{persona} You are grading Question {q} of the {exam.label} exam.",
        "",
    ]
    if use_guidelines:
        parts += [
            "## EXAM GUIDELINES (provided to students)",
            guidelines,
            "",
        ]
    parts += [
        f"## RUBRIC FOR QUESTION {q}",
        rubric_section,
        "",
    ]
    if solution_code is not None:
        parts += [
            "## REFERENCE SOLUTION (instructor's notebook)",
            "```python",
            solution_code,
            "```",
            "",
        ]
    if few_shot_examples:
        parts += [render_few_shot_block(few_shot_examples, q, use_breakdown)]
    parts += [
        "## STUDENT SUBMISSION (the one you must grade)",
        "```python",
        student_code,
        "```",
        "",
        "## YOUR TASK",
        f"Grade strictly against the rubric. Max base score for Q{q} is {max_score}; "
        f"max bonus is {max_bonus}.",
    ]
    if use_breakdown:
        parts.append(
            "Return a JSON object with: `score` (number, <= max base), `bonus` (number, "
            "<= max bonus; 0 if no bonus tasks), `task_breakdown` (array of per-task "
            "awards), and `reasoning` (one short paragraph, <=200 words, explaining the "
            "score)."
        )
        parts.append(
            "`task_breakdown` MUST be non-empty and MUST contain ONE entry for EVERY "
            "task row in the rubric above (including bonus rows). Sum the `awarded` "
            "values to get `score` (base tasks) and `bonus` (bonus tasks). Decide each "
            "task ONCE; do not revise."
        )
    else:
        parts.append(
            "Return a JSON object with ONLY: `score` (number, <= max base), `bonus` "
            "(number, <= max bonus; 0 if no bonus tasks), and `reasoning` (one short "
            "paragraph, <=200 words). Do NOT include a task_breakdown."
        )
    parts += [
        "OUTPUT RULES (strict):",
        "  - Output ONLY the JSON object. No prose before or after.",
        "  - Each task_breakdown entry appears EXACTLY ONCE. Never repeat a task.",
        "  - Do NOT recalculate, re-sum, or revise scores inside `reasoning`.",
        "  - `reasoning` is <=200 words, one paragraph, no bullet lists.",
        "  - If you find yourself writing 'wait', 'let me reconsider', 'actually', "
        "    or restating per-task subtotals: STOP and close the JSON.",
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
    return "\n".join(parts)


def compute_run_summary(output_xlsx: Path) -> dict:
    out = {"mean_total_score": "", "std_total_score": "",
           "MAE_vs_TA_avg": "", "n_graded": 0}
    if not output_xlsx.exists():
        return out
    try:
        wb = openpyxl.load_workbook(output_xlsx, data_only=True)
        ws = wb.active
        headers = [c.value for c in ws[1]]
        def col(name):
            return headers.index(name) + 1 if name in headers else None
        c_ai = col("AI Total Score")
        c_ta1 = col("TA 1 - Total Score (out of 35)")
        c_ta2 = col("TA 2 - Total Score (out of 35)")
        # Per-question TA Score columns, used to reconstruct a TA total when the
        # "TA n - Total Score" cells are formulas with no cached value (openpyxl
        # data_only reads those as None, which otherwise blanks the MAE).
        c_ta_q = {ta: [col(f"TA {ta} - Q{q} Score") for q in (1, 2, 3, 4)]
                  for ta in (1, 2)}
        if c_ai is None:
            return out

        def ta_total(r, ta: int, direct_col):
            v = r[direct_col - 1] if direct_col else None
            if isinstance(v, (int, float)):
                return float(v)
            parts = [r[c - 1] for c in c_ta_q[ta] if c]
            parts = [p for p in parts if isinstance(p, (int, float))]
            return float(sum(parts)) if parts else None

        scores: list[float] = []
        diffs: list[float] = []
        for r in ws.iter_rows(min_row=2, values_only=True):
            ai_val = r[c_ai - 1]
            if ai_val in (None, ""):
                continue
            try:
                ai = float(ai_val)
            except (TypeError, ValueError):
                continue
            scores.append(ai)
            ta_vals = [t for t in (ta_total(r, 1, c_ta1), ta_total(r, 2, c_ta2))
                       if t is not None]
            if ta_vals:
                diffs.append(ai - sum(ta_vals) / len(ta_vals))
        out["n_graded"] = len(scores)
        if scores:
            out["mean_total_score"] = round(statistics.fmean(scores), 3)
            if len(scores) >= 2:
                out["std_total_score"] = round(statistics.stdev(scores), 3)
        if diffs:
            out["MAE_vs_TA_avg"] = round(sum(abs(d) for d in diffs) / len(diffs), 3)
    except Exception as e:  # noqa: BLE001
        logging.warning("Failed to summarize %s: %s", output_xlsx, e)
    return out


