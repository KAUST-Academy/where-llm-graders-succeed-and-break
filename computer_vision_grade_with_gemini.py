#!/usr/bin/env python3
"""Grade student notebook submissions with Gemini 3 Flash Lite (Vertex AI).

For each student in ``computer_vision_dataset/submissions_extracted/<Number>/`` and each
question (Q1-Q4), the script sends Gemini:

  1. ``computer_vision_dataset/exam_guidelines_student.md``   (always)
  2. The rubric section for that question     (always)
  3. The reference solution notebook          (toggle at startup)
  4. The student's submission notebook        (always)

It asks the model to return a structured grade + reasoning, then writes
those into ``computer_vision_dataset/Practical_AI_exam_grades.xlsx``. The workbook is saved after
every question so the run can be resumed after a crash: any question
whose score cell already has a value is skipped.

Run for student #1 first to verify, then run for ``--all``.
"""
from __future__ import annotations

import argparse
import csv
import errno
import json
import logging
import os
import re
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import openpyxl
from openpyxl.utils.exceptions import InvalidFileException
from zipfile import BadZipFile
from google import genai
from google.genai import types
from google.oauth2 import service_account

ROOT = Path(__file__).resolve().parent
KEY_FILE = ROOT / "vertex_key.json"
GUIDELINES = ROOT / "computer_vision_dataset" / "exam_guidelines_student.md"
RUBRIC = ROOT / "computer_vision_dataset" / "stage_3_rubrics_ta.md"
SOLUTIONS_DIR = ROOT / "computer_vision_dataset" / "Solutions"
EXTRACTED_DIR = ROOT / "computer_vision_dataset" / "submissions_extracted"
GRADES_XLSX = ROOT / "computer_vision_dataset" / "Practical_AI_exam_grades.xlsx"
LOG_FILE = ROOT / "gemini_grading.log"
PROMPTS_DIR = ROOT / "gemini_prompts"
RESPONSES_DIR = ROOT / "gemini_responses"
VARIANCE_CSV = ROOT / "gemini_variance.csv"
LOGS_DIR = ROOT / "logs"
RESULTS_DIR = ROOT / "computer_vision_results" / "results"
VARIANCE_DIR = ROOT / "computer_vision_results" / "variance"
ABLATION_TRACKER = ROOT / "computer_vision_results" / "ablation_runs.xlsx"
FEW_SHOT_BANK = ROOT / "computer_vision_dataset" / "few_shot_examples.json"

DEFAULT_MODEL = "gemini-flash-lite-latest"  # Gemini 3 Flash Lite on Vertex
LOCATION = "global"

# Vertex AI model IDs verified available in this project (as of 2026-05):
#   gemini-3.1-flash-lite          (Gemini 3.1 Flash Lite, stable, cheapest)
#   gemini-flash-lite-latest       (alias -> Gemini 3.1 Flash Lite, default)
#   gemini-3-flash-preview         (Gemini 3 Flash, mid-tier)
#   gemini-3.1-pro-preview         (Gemini 3.1 Pro, highest quality, slowest/most expensive)
#   gemini-2.5-pro                 (older generation, kept for comparison)
#   gemini-2.5-flash               (older generation)
#
# To list every model your project can reach:
#   for m in client.models.list(): print(m.name)

STRICTNESS_PRESETS = {
    "strict": (
        "You are a HARSH teaching assistant. Award the MINIMUM defensible score "
        "for any task that is incomplete, buggy, or deviates from the rubric. "
        "Never give partial credit if the task does not run correctly."
    ),
    "neutral": (
        "You are a strict but fair teaching assistant."
    ),
    "lenient": (
        "You are a GENEROUS teaching assistant. Give the student the benefit of "
        "the doubt. Award the MAXIMUM defensible score whenever a reasonable "
        "attempt at the task is visible, even if the code is buggy or incomplete."
    ),
    # Alternate strict-flavored personas used by the P-series ablations to test
    # whether the L-C01 collapse on Qwen2.5-Coder-32B is wording-specific or
    # a general fragility to strict-style prompting.
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
}

# Map question -> (max_score, max_bonus, solution filename)
QUESTION_META = {
    1: (12.0, 4.0, "Q1_stage3_Finetuning_Solution.ipynb"),
    2: (11.0, 4.0, "Q2_stage3_CNN_Solution.ipynb"),
    3: (12.0, 0.0, "Q3_stage3_Segmentation_Solution.ipynb"),
    4: (0.0, 5.0, "Q4_stage3_Colorization_Solution.ipynb"),
}

# Excel header names we add to the right of the existing columns.
def ai_headers() -> list[str]:
    cols: list[str] = []
    for q in (1, 2, 3, 4):
        cols.append(f"AI Q{q} Score")
        if QUESTION_META[q][1] > 0:
            cols.append(f"AI Q{q} Bonus")
        cols.append(f"AI Q{q} Reasoning")
    cols += ["AI Total Score", "AI Total Bonus"]
    return cols


class _DropThoughtSignatureWarnings(logging.Filter):
    """Drop the Gemini SDK's noisy 'non-text parts in the response: thought_signature'
    warnings. They fire on every Gemini 3 reply and add zero information."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "non-text parts in the response" not in msg


def setup_logging(log_file: Path) -> None:
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
    drop_filter = _DropThoughtSignatureWarnings()
    for h in handlers:
        h.addFilter(drop_filter)
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


# ---------------------------------------------------------------------------
# Rubric parsing
# ---------------------------------------------------------------------------

def load_rubric_sections() -> dict[int, str]:
    """Return {q -> markdown text} for each '## Question N' or '## Bonus Question N' section."""
    text = RUBRIC.read_text(encoding="utf-8")
    # Split on lines starting with '## ' (h2). Keep headers.
    parts = re.split(r"(?m)^(##\s+.*)$", text)
    # parts looks like: [pre, header1, body1, header2, body2, ...]
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


# ---------------------------------------------------------------------------
# Notebook -> plain code text
# ---------------------------------------------------------------------------

def load_few_shot_examples(q: int, n: int) -> list[dict]:
    """Return up to n worked-example gradings for question q from
    `few_shot_examples.json`. If the file is missing or has too few entries
    we return whatever is available — the caller decides whether to abort."""
    if n <= 0:
        return []
    if not FEW_SHOT_BANK.exists():
        raise FileNotFoundError(
            f"--few-shot {n} requested but {FEW_SHOT_BANK} is missing. "
            "Run `python computer_vision_build_few_shot_examples.py` to materialize it."
        )
    bank = json.loads(FEW_SHOT_BANK.read_text(encoding="utf-8"))
    return bank.get(f"Q{q}", [])[:n]


def render_few_shot_block(examples: list[dict], q: int, use_breakdown: bool) -> str:
    """Render K worked examples for question q as a single markdown block to
    splice into the prompt right before the actual student submission."""
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
            # We don't have ground-truth task breakdowns in the bank; include a
            # short note so the model still emits a breakdown for the live call
            # without us fabricating per-task awards.
            obj["task_breakdown"] = (
                "[…omitted in this example; you MUST still produce one for the "
                "live grading below.]"
            )
        lines.append(json.dumps(obj, indent=2))
        lines.append("```")
    lines.append("")
    return "\n".join(lines)


def notebook_to_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    out: list[str] = []
    for idx, cell in enumerate(nb.get("cells", []), start=1):
        src = "".join(cell.get("source", []))
        ctype = cell.get("cell_type", "code")
        out.append(f"--- Cell {idx} [{ctype}] ---\n{src}\n")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Excel helpers (with resume)
# ---------------------------------------------------------------------------

def atomic_save(wb: openpyxl.Workbook, path: Path) -> None:
    """Save the workbook to a temp file in the same dir, then atomically rename.
    Prevents corruption when the process is killed mid-save."""
    tmp = path.with_name(path.name + ".tmp")
    wb.save(tmp)
    os.replace(tmp, path)


def init_output_workbook(template: Path, dest: Path) -> int:
    """Clone the master workbook for a fresh run: blank the AI columns, and keep
    everything else — including the values behind the TA formula columns.

    Mirrors computer_vision_grade_with_local.init_output_workbook. openpyxl never re-emits a
    formula's cached result, so a bare load_workbook -> save round-trip empties
    every formula cell (`TA {1,2} - Total Score (out of 35)`, the TA
    total-bonus columns, the workbook averages). shutil.copy2 preserves those
    cached values; the round-trip right after is what destroys them, leaving a
    workbook whose TA ground truth reads as NaN in run_analysis.

    Returns the number of formula cells frozen.
    """
    import shutil
    shutil.copy2(template, dest)
    wb = openpyxl.load_workbook(dest)
    ws = wb.active
    hdrs = [c.value for c in ws[1]]
    ai_cols = {i + 1 for i, h in enumerate(hdrs) if h and str(h).startswith("AI ")}
    cached = openpyxl.load_workbook(template, data_only=True).active
    frozen = 0
    unresolved = 0
    for r in range(2, ws.max_row + 1):
        for c in range(1, ws.max_column + 1):
            cell = ws.cell(row=r, column=c)
            if c in ai_cols:
                cell.value = None
            elif isinstance(cell.value, str) and cell.value.startswith("="):
                cell.value = cached.cell(row=r, column=c).value
                frozen += 1
                if cell.value is None:
                    unresolved += 1
    if unresolved:
        logging.warning(
            "%d formula cells in %s carry no cached value; they stay empty and "
            "the analysis will read them as missing.", unresolved, template,
        )
    atomic_save(wb, dest)
    return frozen


def safe_load_workbook(path: Path, template: Path) -> openpyxl.Workbook:
    """Load `path` as an xlsx. If it's missing or corrupted (e.g. killed mid-save),
    re-create it from `template` with AI cells cleared, then load that."""
    try:
        return openpyxl.load_workbook(path)
    except (BadZipFile, InvalidFileException, KeyError) as e:
        logging.warning(
            "Output workbook %s is corrupted (%s: %s); re-initializing from %s.",
            path, type(e).__name__, e, template,
        )
        # Same formula-freezing path as a fresh clone: recovering from
        # corruption must not quietly produce an un-analysable workbook.
        n_frozen = init_output_workbook(template, path)
        logging.info("Re-initialized %s (%d formula cells frozen)", path, n_frozen)
        return openpyxl.load_workbook(path)


def ensure_ai_columns(ws) -> dict[str, int]:
    """Make sure AI columns exist in the header row. Return {header -> 1-based col index}."""
    headers = [c.value for c in ws[1]]
    name_to_col: dict[str, int] = {h: i + 1 for i, h in enumerate(headers) if h}
    # Use the rightmost *non-empty* header column, not ws.max_column, which can
    # include trailing empty columns left over from prior edits and would leave
    # a confusing blank gap between the existing data and the new AI columns.
    next_col = (max(name_to_col.values()) + 1) if name_to_col else 1
    for h in ai_headers():
        if h not in name_to_col:
            ws.cell(row=1, column=next_col, value=h)
            name_to_col[h] = next_col
            next_col += 1
    return name_to_col


def build_student_row_index(ws) -> dict[int, int]:
    """Build {student_number -> row_index} once so per-student lookup is O(1)."""
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


def find_student_row(ws, number: int) -> Optional[int]:
    for row in ws.iter_rows(min_row=2):
        if row[0].value == number:
            return row[0].row
    return None


# ---------------------------------------------------------------------------
# Gemini call
# ---------------------------------------------------------------------------

SCHEMA_WITH_BREAKDOWN = {
    "type": "OBJECT",
    "properties": {
        "score": {"type": "NUMBER", "description": "Base score for the question, before bonus."},
        "bonus": {"type": "NUMBER", "description": "Bonus points (0 if no bonus task for this question)."},
        "task_breakdown": {
            "type": "ARRAY",
            "minItems": 1,
            "maxItems": 30,
            "items": {
                "type": "OBJECT",
                "properties": {
                    "task": {"type": "STRING", "maxLength": 200},
                    "awarded": {"type": "NUMBER"},
                    "max": {"type": "NUMBER"},
                    "note": {"type": "STRING", "maxLength": 300},
                },
                "required": ["task", "awarded", "max"],
            },
        },
        "reasoning": {
            "type": "STRING",
            "maxLength": 1500,
            "description": "Concise explanation of the grading decisions, <=200 words.",
        },
    },
    "required": ["score", "bonus", "task_breakdown", "reasoning"],
}

SCHEMA_NO_BREAKDOWN = {
    "type": "OBJECT",
    "properties": {
        "score": {"type": "NUMBER", "description": "Base score for the question, before bonus."},
        "bonus": {"type": "NUMBER", "description": "Bonus points (0 if no bonus task for this question)."},
        "reasoning": {
            "type": "STRING",
            "maxLength": 1500,
            "description": "Concise explanation of the grading decisions, <=200 words.",
        },
    },
    "required": ["score", "bonus", "reasoning"],
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
) -> str:
    max_score, max_bonus, _ = QUESTION_META[q]
    persona = STRICTNESS_PRESETS.get(strictness, STRICTNESS_PRESETS["neutral"])
    parts = [
        f"{persona} You are grading Question {q} of the KAUST Stage 3 Practical AI exam.",
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
        parts += [
            render_few_shot_block(few_shot_examples, q, use_breakdown),
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


def get_client() -> genai.Client:
    creds = service_account.Credentials.from_service_account_file(
        str(KEY_FILE),
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    with open(KEY_FILE, "r", encoding="utf-8") as f:
        project_id = json.load(f)["project_id"]
    return genai.Client(
        vertexai=True,
        project=project_id,
        location=LOCATION,
        credentials=creds,
    )


def _supports_thinking_budget_zero(model: str) -> bool:
    """Only Gemini-3.x flash/pro model IDs accept thinking_budget=0; older
    models (e.g. gemini-2.5-pro) reject it and always think."""
    m = model.lower()
    return "flash-lite" in m or "3-flash" in m or "3.1" in m


def grade_question(
    client: genai.Client,
    prompt: str,
    use_reasoning: bool,
    use_breakdown: bool,
    temperature: float,
    model: str = DEFAULT_MODEL,
) -> tuple[dict, str]:
    """Call Gemini. Returns (parsed_json, raw_text)."""
    config_kwargs = dict(
        response_mime_type="application/json",
        response_schema=SCHEMA_WITH_BREAKDOWN if use_breakdown else SCHEMA_NO_BREAKDOWN,
        temperature=temperature,
        max_output_tokens=65535,
    )
    # Thinking toggle. Gemini 3.x flash/pro accept thinking_budget=0 to fully
    # disable thinking — without this, --no-reasoning still lets the model
    # silently burn the 65535 output cap thinking before emitting JSON, which
    # is what causes spurious MAX_TOKENS failures on long prompts. Older
    # models (gemini-2.5-pro) reject thinking_budget=0 entirely, so we only
    # opt-in for known Gemini-3.x model IDs. NOTE: for those older models
    # --no-reasoning is therefore a NO-OP — the model still thinks at its
    # default (dynamic) budget; main() warns about this at startup.
    if use_reasoning:
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=-1)
    elif _supports_thinking_budget_zero(model):
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    config = types.GenerateContentConfig(**config_kwargs)

    last_err: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            # If the model ran out of tokens mid-JSON, json.loads will throw an
            # opaque error. Detect and surface the real cause first.
            finish_reason = None
            try:
                finish_reason = resp.candidates[0].finish_reason
            except (AttributeError, IndexError, TypeError):
                pass
            if finish_reason is not None and str(finish_reason).endswith("MAX_TOKENS"):
                raise RuntimeError(
                    f"Gemini response truncated at max_output_tokens "
                    f"({config_kwargs['max_output_tokens']}); JSON is incomplete. "
                    "Consider disabling reasoning or increasing the cap."
                )
            return json.loads(resp.text), resp.text
        except Exception as e:  # noqa: BLE001
            last_err = e
            # MAX_TOKENS at temp=0 is deterministic; retrying just burns time.
            # Bail immediately so the script moves on to the next question.
            if "truncated at max_output_tokens" in str(e):
                logging.error(
                    "Gemini hit MAX_TOKENS on attempt %d; not retrying (deterministic).",
                    attempt,
                )
                break
            logging.warning("Gemini call failed (attempt %d/3): %s", attempt, e)
            time.sleep(2 * attempt)
    raise RuntimeError(f"Gemini call failed: {last_err}")


# ---------------------------------------------------------------------------
# Per-student driver
# ---------------------------------------------------------------------------

def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_") or "run"


def _stem(student_number: int, q: int, run_tag: str, run_idx: int, runs: int) -> str:
    base = f"student_{student_number}_Q{q}"
    if run_tag:
        base += f"__{_slug(run_tag)}"
    if runs > 1:
        base += f"__run{run_idx}"
    return base


def grade_student(
    client: genai.Client,
    student_number: int,
    rubric_sections: dict[int, str],
    guidelines: str,
    use_solution: bool,
    use_reasoning: bool,
    use_breakdown: bool,
    use_guidelines: bool,
    strictness: str,
    model: str,
    temperature: float,
    runs: int,
    run_tag: str,
    wb: openpyxl.Workbook,
    ws,
    cols: dict[str, int],
    row_index: dict[int, int],
    output_xlsx: Path,
    save_io: bool,
    variance_csv: Path,
    few_shot: int = 0,
) -> None:
    student_dir = EXTRACTED_DIR / str(student_number)
    if not student_dir.is_dir():
        logging.warning("Student %s: no extracted folder at %s", student_number, student_dir)
        return

    row = row_index.get(student_number)
    if row is None:
        logging.warning("Student %s: no row in grades workbook", student_number)
        return

    variance_mode = runs > 1
    totals_score = 0.0
    totals_bonus = 0.0
    graded_any = False
    failed_qs: list[int] = []

    for q in (1, 2, 3, 4):
        score_col = cols[f"AI Q{q} Score"]
        reason_col = cols[f"AI Q{q} Reasoning"]
        bonus_col = cols.get(f"AI Q{q} Bonus")

        # In variance mode we always re-grade (the whole point is to sample).
        if not variance_mode:
            existing = ws.cell(row=row, column=score_col).value
            if existing not in (None, ""):
                logging.info("Student %s Q%d: already graded (score=%s), skipping",
                             student_number, q, existing)
                try:
                    totals_score += float(existing)
                except (TypeError, ValueError):
                    logging.warning(
                        "Student %s Q%d: existing score %r is not numeric; "
                        "excluding from AI Total Score.",
                        student_number, q, existing,
                    )
                if bonus_col:
                    b = ws.cell(row=row, column=bonus_col).value
                    if b not in (None, ""):
                        try:
                            totals_bonus += float(b)
                        except (TypeError, ValueError):
                            logging.warning(
                                "Student %s Q%d: existing bonus %r is not numeric; "
                                "excluding from AI Total Bonus.",
                                student_number, q, b,
                            )
                graded_any = True
                continue

        nb_path = student_dir / f"Q{q}.ipynb"
        if not nb_path.exists():
            logging.info("Student %s Q%d: no submission notebook, score=0",
                         student_number, q)
            if not variance_mode:
                ws.cell(row=row, column=score_col, value=0)
                if bonus_col:
                    ws.cell(row=row, column=bonus_col, value=0)
                ws.cell(row=row, column=reason_col, value="No submission found.")
                atomic_save(wb, output_xlsx)
            continue

        student_code = notebook_to_text(nb_path)
        solution_code = None
        if use_solution:
            sol_path = SOLUTIONS_DIR / QUESTION_META[q][2]
            if sol_path.exists():
                solution_code = notebook_to_text(sol_path)
            else:
                logging.warning("Solution missing for Q%d: %s", q, sol_path)

        few_shot_examples = load_few_shot_examples(q, few_shot) if few_shot > 0 else None
        prompt = build_prompt(
            q=q,
            rubric_section=rubric_sections[q],
            guidelines=guidelines,
            student_code=student_code,
            solution_code=solution_code,
            use_breakdown=use_breakdown,
            use_guidelines=use_guidelines,
            strictness=strictness,
            few_shot_examples=few_shot_examples,
        )

        run_scores: list[float] = []
        run_bonuses: list[float] = []
        last_reasoning = ""

        for run_idx in range(1, runs + 1):
            stem = _stem(student_number, q, run_tag, run_idx, runs)
            if save_io:
                prompt_path = PROMPTS_DIR / f"{stem}.txt"
                prompt_path.parent.mkdir(parents=True, exist_ok=True)
                prompt_path.write_text(prompt, encoding="utf-8")
            logging.info(
                "Student %s Q%d run %d/%d: calling %s "
                "(prompt %d chars; solution=%s guidelines=%s reasoning=%s breakdown=%s "
                "strictness=%s temp=%s)",
                student_number, q, run_idx, runs, model, len(prompt),
                bool(solution_code), use_guidelines, use_reasoning, use_breakdown,
                strictness, temperature,
            )
            logging.debug("Student %s Q%d run %d PROMPT:\n%s",
                          student_number, q, run_idx, prompt)

            try:
                result, raw_text = grade_question(
                    client, prompt, use_reasoning, use_breakdown, temperature, model,
                )
            except Exception as e:  # noqa: BLE001
                logging.error("Student %s Q%d run %d: grading failed: %s",
                              student_number, q, run_idx, e)
                continue

            if save_io:
                resp_path = RESPONSES_DIR / f"{stem}.json"
                resp_path.parent.mkdir(parents=True, exist_ok=True)
                resp_path.write_text(raw_text, encoding="utf-8")
            logging.debug("Student %s Q%d run %d RESPONSE:\n%s",
                          student_number, q, run_idx, raw_text)

            raw_score = float(result.get("score", 0) or 0)
            raw_bonus = float(result.get("bonus", 0) or 0)
            q_max_score, q_max_bonus, _ = QUESTION_META[q]
            score = max(0.0, min(raw_score, q_max_score))
            bonus = max(0.0, min(raw_bonus, q_max_bonus))
            if score != raw_score or bonus != raw_bonus:
                logging.warning(
                    "Student %s Q%d run %d: model returned out-of-range "
                    "score=%s bonus=%s; clamped to score=%s bonus=%s",
                    student_number, q, run_idx,
                    raw_score, raw_bonus, score, bonus,
                )
            reasoning = result.get("reasoning", "") or ""
            breakdown = result.get("task_breakdown") or []
            if breakdown:
                reasoning = (
                    reasoning
                    + "\n\nBreakdown:\n"
                    + "\n".join(
                        f"- {b.get('task','?')}: {b.get('awarded','?')}/{b.get('max','?')}"
                        + (f" — {b['note']}" if b.get('note') else "")
                        for b in breakdown
                    )
                )

            logging.info("Student %s Q%d run %d: score=%s bonus=%s",
                         student_number, q, run_idx, score, bonus)
            run_scores.append(score)
            run_bonuses.append(bonus)
            last_reasoning = reasoning

            # Append a row to the variance CSV regardless of mode -- handy log.
            append_variance_row(
                variance_csv,
                student_number, q, run_idx, run_tag,
                model, use_solution, use_reasoning, use_breakdown,
                use_guidelines, strictness, temperature,
                score, bonus,
            )

        if not run_scores:
            logging.error(
                "Student %s Q%d: all %d run(s) failed; no score recorded.",
                student_number, q, runs,
            )
            failed_qs.append(q)
            continue

        if variance_mode:
            mean_s = statistics.fmean(run_scores)
            mean_b = statistics.fmean(run_bonuses)
            # Sample stdev (n-1). For a single successful run, stdev is undefined.
            if len(run_scores) >= 2:
                std_s = statistics.stdev(run_scores)
                std_b = statistics.stdev(run_bonuses)
                std_s_str = f"{std_s:.3f}"
                std_b_str = f"{std_b:.3f}"
            else:
                std_s_str = "n/a"
                std_b_str = "n/a"
            logging.info(
                "Student %s Q%d VARIANCE (n=%d): score mean=%.3f std=%s range=[%s,%s]; "
                "bonus mean=%.3f std=%s range=[%s,%s]",
                student_number, q, len(run_scores),
                mean_s, std_s_str, min(run_scores), max(run_scores),
                mean_b, std_b_str, min(run_bonuses), max(run_bonuses),
            )
            # In variance mode we do NOT overwrite the production excel cells.
            continue

        score = run_scores[-1]
        bonus = run_bonuses[-1]
        ws.cell(row=row, column=score_col, value=score)
        if bonus_col:
            ws.cell(row=row, column=bonus_col, value=bonus)
        ws.cell(row=row, column=reason_col, value=last_reasoning)
        atomic_save(wb, output_xlsx)

        totals_score += score
        totals_bonus += bonus
        graded_any = True

    if graded_any and not variance_mode:
        if failed_qs:
            # A total missing one question would masquerade downstream as a
            # (much lower) full 35-pt total and corrupt the run's mean/MAE.
            # Leave the totals blank; a resume re-attempts the failed Qs and
            # fills the totals once every question has a score.
            # NB: assign .value directly — ws.cell(..., value=None) is a no-op.
            ws.cell(row=row, column=cols["AI Total Score"]).value = None
            ws.cell(row=row, column=cols["AI Total Bonus"]).value = None
            logging.warning(
                "Student %s: Q%s permanently failed this run; totals left "
                "blank (resume the run to retry and fill them).",
                student_number, failed_qs,
            )
        else:
            ws.cell(row=row, column=cols["AI Total Score"], value=totals_score)
            ws.cell(row=row, column=cols["AI Total Bonus"], value=totals_bonus)
        atomic_save(wb, output_xlsx)


def append_variance_row(
    csv_path: Path,
    student: int, q: int, run_idx: int, run_tag: str,
    model: str, use_solution: bool, use_reasoning: bool, use_breakdown: bool,
    use_guidelines: bool, strictness: str, temperature: float,
    score: float, bonus: float,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        # Check size AFTER opening in append mode -- avoids the race where
        # two parallel runs both see "file doesn't exist" and write headers.
        needs_header = f.tell() == 0
        w = csv.writer(f)
        if needs_header:
            w.writerow([
                "timestamp", "tag", "student", "question", "run",
                "model", "use_solution", "use_reasoning", "use_breakdown",
                "use_guidelines", "strictness", "temperature",
                "score", "bonus",
            ])
        w.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            run_tag, student, q, run_idx,
            model, int(use_solution), int(use_reasoning), int(use_breakdown),
            int(use_guidelines), strictness, temperature,
            score, bonus,
        ])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_student_list(spec: str) -> list[int]:
    """Parse "1,2,5-10" -> [1, 2, 5, 6, 7, 8, 9, 10]."""
    out: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(chunk))
    return out


def _ask(prompt: str, default: str = "") -> str:
    """input() that returns the default on EOF instead of crashing."""
    try:
        return input(prompt).strip()
    except EOFError:
        logging.warning("stdin closed while prompting %r; using default %r.", prompt, default)
        return default


def _yn(question: str, default_no: bool = True) -> bool:
    suffix = "[y/N]" if default_no else "[Y/n]"
    while True:
        try:
            ans = input(f"{question} {suffix}: ").strip().lower()
        except EOFError:
            # Non-interactive stdin (e.g. piped, nohup): take the default.
            logging.warning(
                "stdin closed while prompting %r; using default (%s).",
                question, "no" if default_no else "yes",
            )
            return not default_no
        if ans == "":
            return not default_no
        if ans in ("n", "no"):
            return False
        if ans in ("y", "yes"):
            return True


# ---------------------------------------------------------------------------
# Ablation tracker (computer_vision_results/ablation_runs.xlsx) auto-update
# ---------------------------------------------------------------------------

@contextmanager
def _file_lock(target: Path, timeout_s: float = 60.0, poll_s: float = 0.25):
    """Cross-platform best-effort lock so parallel runs don't trample the tracker xlsx."""
    lock_path = target.with_suffix(target.suffix + ".lock")
    deadline = time.time() + timeout_s
    fd = None
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            break
        except FileExistsError:
            if time.time() > deadline:
                raise TimeoutError(f"Timed out acquiring lock on {lock_path}")
            time.sleep(poll_s)
        except OSError as e:
            if e.errno == errno.EEXIST:
                if time.time() > deadline:
                    raise TimeoutError(f"Timed out acquiring lock on {lock_path}")
                time.sleep(poll_s)
            else:
                raise
    try:
        yield
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.unlink(lock_path)
        except OSError:
            pass


def update_ablation_row(tracker_path: Path, run_tag: str, updates: dict) -> None:
    """Read tracker, find row by run_tag, write the given column updates, save."""
    if not tracker_path.exists() or not run_tag:
        return
    try:
        with _file_lock(tracker_path):
            wb = openpyxl.load_workbook(tracker_path)
            ws = wb.active
            headers = [c.value for c in ws[1]]
            try:
                tag_col = headers.index("run_tag") + 1
            except ValueError:
                logging.warning("Tracker %s has no 'run_tag' column; skipping update.",
                                tracker_path)
                return
            target_row = None
            for r in ws.iter_rows(min_row=2):
                if r[tag_col - 1].value == run_tag:
                    target_row = r[0].row
                    break
            if target_row is None:
                logging.warning("Tracker has no row for tag=%s; skipping update.", run_tag)
                return
            for key, val in updates.items():
                if key not in headers:
                    continue
                ws.cell(row=target_row, column=headers.index(key) + 1).value = val
            atomic_save(wb, tracker_path)
    except Exception as e:  # noqa: BLE001
        logging.warning("Failed to update ablation tracker (tag=%s): %s", run_tag, e)


def compute_run_summary(output_xlsx: Path) -> dict:
    """Read the per-run result xlsx and return {mean_total_score, std_total_score,
    MAE_vs_TA_avg, n_graded}."""
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
        if c_ai is None:
            return out
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
            t1 = r[c_ta1 - 1] if c_ta1 else None
            t2 = r[c_ta2 - 1] if c_ta2 else None
            ta_vals = [x for x in (t1, t2) if isinstance(x, (int, float))]
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


def main() -> None:
    parser = argparse.ArgumentParser()
    student_group = parser.add_mutually_exclusive_group()
    student_group.add_argument("--student", type=int, help="Grade a single student by Number")
    student_group.add_argument("--students", type=str, default=None,
                        help='Comma-separated student numbers or ranges, e.g. "1,2,5" or "1-10"')
    student_group.add_argument("--all", action="store_true", help="Grade every student found in submissions_extracted")
    parser.add_argument("--output-xlsx", type=str, default=None,
                        help="Path to the workbook to write into (default: Practical_AI_exam_grades.xlsx). "
                             "If it doesn't exist, the default workbook is copied to that path first.")
    parser.add_argument("--save-io", action="store_true",
                        help="Also save every prompt to gemini_prompts/ and every response to gemini_responses/ "
                             "(off by default).")
    parser.add_argument("--with-solution", dest="with_solution", action="store_true",
                        help="Include reference solution (non-interactive)")
    parser.add_argument("--no-solution", dest="no_solution", action="store_true",
                        help="Exclude reference solution (non-interactive)")
    parser.add_argument("--with-reasoning", dest="with_reasoning", action="store_true",
                        help="Enable Gemini thinking/reasoning (non-interactive)")
    parser.add_argument("--no-reasoning", dest="no_reasoning", action="store_true",
                        help="Disable Gemini thinking/reasoning (non-interactive)")
    parser.add_argument("--with-breakdown", dest="with_breakdown", action="store_true",
                        help="Ask for per-task breakdown in the response (default)")
    parser.add_argument("--no-breakdown", dest="no_breakdown", action="store_true",
                        help="Only ask for score + reasoning, no per-task breakdown")
    parser.add_argument("--with-guidelines", dest="with_guidelines", action="store_true",
                        help="Include exam_guidelines_student.md in the prompt (default)")
    parser.add_argument("--no-guidelines", dest="no_guidelines", action="store_true",
                        help="Exclude exam_guidelines_student.md from the prompt")
    parser.add_argument("--strictness", choices=list(STRICTNESS_PRESETS.keys()),
                        default=None,
                        help="Grader persona: strict / neutral (default) / lenient")
    parser.add_argument("--model", type=str, default=None,
                        help=f"Vertex model ID (default: {DEFAULT_MODEL}). "
                             "Try gemini-flash-latest, gemini-2.5-pro, gemini-2.5-flash.")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature (default 0.0). Bump to 0.3-0.7 for variance studies.")
    parser.add_argument("--runs", type=int, default=None,
                        help="Number of times to grade each question (default 1). >1 enables variance mode: "
                             "results are written to gemini_variance.csv and the excel is NOT touched.")
    parser.add_argument("--tag", type=str, default="",
                        help="Optional run tag; namespaces saved prompts/responses and labels CSV rows.")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Path to log file (default: logs/<tag>.log if --tag is set, "
                             "else gemini_grading.log). Required separate file per parallel run.")
    parser.add_argument("--variance-csv", type=str, default=None,
                        help="Path to variance CSV (default: variance/<tag>.csv if --tag is set, "
                             "else gemini_variance.csv). Required separate file per parallel run.")
    parser.add_argument("--few-shot", dest="few_shot", type=int, default=0,
                        help="N>0 prepends N worked-example gradings per question "
                             "(loaded from few_shot_examples.json). Default 0.")
    parser.add_argument("--tracker", type=str, default=None,
                        help=f"Path to the ablation tracker xlsx to back-fill "
                             f"with status/metrics on completion. Default: "
                             f"computer_vision_results/{ABLATION_TRACKER.name}.")
    parser.add_argument("--debug", action="store_true",
                        help="Verbose log: also dumps full prompts/responses into the log file")
    args = parser.parse_args()

    # Resolve log + variance paths. If --tag is set and the user didn't override,
    # derive per-run paths so parallel runs don't fight over the same files.
    if args.log_file:
        log_file = Path(args.log_file).resolve()
    elif args.tag:
        log_file = (LOGS_DIR / f"{args.tag}.log").resolve()
    else:
        log_file = LOG_FILE

    if args.variance_csv:
        variance_csv = Path(args.variance_csv).resolve()
    elif args.tag:
        variance_csv = (VARIANCE_DIR / f"{args.tag}.csv").resolve()
    else:
        variance_csv = VARIANCE_CSV

    setup_logging(log_file)
    logging.info("Log file = %s", log_file)
    logging.info("Variance CSV = %s", variance_csv)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.with_solution:
        use_solution = True
    elif args.no_solution:
        use_solution = False
    else:
        use_solution = _yn("Include the reference solution in the prompt?", default_no=True)

    if args.with_reasoning:
        use_reasoning = True
    elif args.no_reasoning:
        use_reasoning = False
    else:
        use_reasoning = _yn("Enable Gemini reasoning (thinking)?", default_no=True)

    if args.with_breakdown:
        use_breakdown = True
    elif args.no_breakdown:
        use_breakdown = False
    else:
        use_breakdown = _yn("Ask for per-task breakdown in the response?", default_no=False)

    if args.with_guidelines:
        use_guidelines = True
    elif args.no_guidelines:
        use_guidelines = False
    else:
        use_guidelines = _yn("Include exam_guidelines_student.md in the prompt?", default_no=False)

    if args.strictness is not None:
        strictness = args.strictness
    else:
        raw = _ask("Strictness [strict / neutral / lenient, default neutral]: ").lower()
        strictness = raw if raw in STRICTNESS_PRESETS else "neutral"

    if args.model is not None:
        model = args.model
    else:
        raw = _ask(f"Model [default {DEFAULT_MODEL}]: ")
        model = raw or DEFAULT_MODEL

    if args.temperature is not None:
        temperature = args.temperature
    else:
        raw = _ask("Temperature [default 0.0]: ")
        temperature = float(raw) if raw else 0.0

    if args.runs is not None:
        runs = args.runs
    else:
        raw = _ask("Number of runs per question [default 1]: ")
        runs = int(raw) if raw else 1
    if runs < 1:
        runs = 1
    if runs > 50:
        logging.warning("runs=%d is unusually large; each run is a paid API call.", runs)

    run_tag = args.tag

    logging.info("Model = %s", model)
    logging.info("Reference solution will be %s in prompts", "INCLUDED" if use_solution else "EXCLUDED")
    logging.info("Exam guidelines will be %s in prompts", "INCLUDED" if use_guidelines else "EXCLUDED")
    logging.info("Reasoning/thinking is %s", "ENABLED" if use_reasoning else "DISABLED")
    if not use_reasoning and not _supports_thinking_budget_zero(model):
        logging.warning(
            "Model %s does not accept thinking_budget=0: --no-reasoning is a "
            "NO-OP and the model will still think at its default dynamic "
            "budget. Label any such run as reasoning-ENABLED.", model,
        )
    logging.info("Per-task breakdown is %s", "INCLUDED" if use_breakdown else "EXCLUDED")
    logging.info("Strictness = %s", strictness)
    logging.info("Few-shot examples per question = %d", args.few_shot)
    if args.few_shot > 0 and not FEW_SHOT_BANK.exists():
        raise SystemExit(
            f"--few-shot {args.few_shot} requires {FEW_SHOT_BANK}; "
            "run `python computer_vision_build_few_shot_examples.py` first."
        )
    logging.info("Temperature = %s; runs per question = %d; tag = %r", temperature, runs, run_tag)
    if runs > 1:
        logging.info("VARIANCE MODE: excel will not be modified; results -> %s", variance_csv)

    rubric_sections = load_rubric_sections()
    guidelines = GUIDELINES.read_text(encoding="utf-8")

    # Resolve output workbook BEFORE the (expensive) Vertex client init so
    # any local copy/path problems fail fast without spending an auth round-trip.
    if args.output_xlsx:
        output_xlsx = Path(args.output_xlsx).resolve()
    elif args.tag:
        output_xlsx = (RESULTS_DIR / f"{args.tag}.xlsx").resolve()
    else:
        output_xlsx = GRADES_XLSX
    if output_xlsx != GRADES_XLSX and not output_xlsx.exists():
        output_xlsx.parent.mkdir(parents=True, exist_ok=True)
        # Strips pre-existing AI cells so this ablation grades from scratch, and
        # freezes formula cells so the clone keeps its TA totals.
        _n_frozen = init_output_workbook(GRADES_XLSX, output_xlsx)
        logging.info(
            "Initialized fresh output workbook %s (AI cells cleared, %d formula "
            "cells frozen to their cached values)", output_xlsx, _n_frozen,
        )
    logging.info("Writing grades to %s", output_xlsx)

    tracker_path = Path(args.tracker).resolve() if args.tracker else ABLATION_TRACKER
    if args.tracker and not tracker_path.exists():
        logging.warning("--tracker %s does not exist; status/metrics will not be back-filled.",
                        tracker_path)

    # Stamp tracker as started (idempotent; no-op if tag missing or no tracker).
    update_ablation_row(tracker_path, run_tag, {
        "status": "running",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    client = get_client()
    logging.info("Vertex client initialized")

    wb = safe_load_workbook(output_xlsx, GRADES_XLSX)
    ws = wb.active
    cols = ensure_ai_columns(ws)
    atomic_save(wb, output_xlsx)
    row_index = build_student_row_index(ws)

    if args.students:
        targets = parse_student_list(args.students)
    elif args.student is not None:
        targets = [args.student]
    elif args.all:
        targets = sorted(int(p.name) for p in EXTRACTED_DIR.iterdir()
                         if p.is_dir() and p.name.isdigit())
    else:
        targets = [1]
        logging.info("No target specified; defaulting to student #1.")

    final_status = "done"
    try:
        for n in targets:
            logging.info("=== Grading student #%d ===", n)
            try:
                grade_student(
                    client, n, rubric_sections, guidelines,
                    use_solution, use_reasoning, use_breakdown,
                    use_guidelines, strictness, model,
                    temperature, runs, run_tag,
                    wb, ws, cols, row_index,
                    output_xlsx, args.save_io,
                    variance_csv,
                    few_shot=args.few_shot,
                )
            except Exception as e:  # noqa: BLE001
                logging.exception("Student %s: unhandled error: %s", n, e)
    except BaseException:
        final_status = "failed"
        raise
    finally:
        # Always back-fill the tracker, even if we crashed mid-way.
        summary = compute_run_summary(output_xlsx) if runs == 1 else {}
        updates = {
            "status": final_status,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        updates.update(summary)
        update_ablation_row(tracker_path, run_tag, updates)

    logging.info("Done.")


if __name__ == "__main__":
    main()
