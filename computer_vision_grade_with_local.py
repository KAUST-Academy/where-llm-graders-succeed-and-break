#!/usr/bin/env python3
"""Grade student notebook submissions with a LOCAL model served by vLLM
(OpenAI-compatible API).

Same prompt, schema, resume logic, and CLI surface as ``computer_vision_grade_with_gemini.py``.
Differences:

  * Uses the ``openai`` SDK pointed at a vLLM server (``--api-base``).
  * Generation is UNCONSTRAINED. A ``guided_json`` key is still sent in
    ``extra_body``, but vLLM removed that parameter in 0.12.0 and its OpenAI
    layer is declared ``extra="allow"``, so from 0.12 onward it is accepted,
    logged as ignored, and dropped -- no grammar is applied. Every run in this
    project executed on 0.19.1 or later and was therefore unconstrained; the
    JSON comes from prompt compliance alone, which is why ``_parse_json_lenient``
    exists. A controlled A/B found enabling grammar-constrained decoding changes
    no score while costing ~6.9x in latency, so the key is left in place as
    inert rather than removed, to keep older runs bit-reproducible.
  * The ``--with-reasoning`` flag toggles Qwen3-style chat-template thinking
    via ``extra_body["chat_template_kwargs"] = {"enable_thinking": True}``.
    For models without that template kwarg, vLLM silently ignores it.

Tracker, log, and variance paths default to per-tag files
(``local_logs/<tag>.log``, ``local_variance/<tag>.csv``) so multiple parallel
runs against different vLLM servers don't fight each other.
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
from openai import OpenAI

# Optional: graceful JSON repair when models emit nearly-valid output (missing
# commas, unescaped control chars, etc.). Falls back to a strict-only path if
# the lib isn't installed.
try:
    import json_repair  # pip install json-repair
    _HAS_JSON_REPAIR = True
except ImportError:
    json_repair = None  # type: ignore
    _HAS_JSON_REPAIR = False


def _parse_json_lenient(text: str) -> dict:
    """Try strict json -> non-strict (allow control chars) -> json_repair.
    Raises json.JSONDecodeError if every layer fails or if the top-level
    value is not an object (callers index into it with .get)."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if parsed is None:
        try:
            parsed = json.loads(text, strict=False)
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
    if not isinstance(parsed, dict):
        raise json.JSONDecodeError(
            f"top-level JSON is {type(parsed).__name__}, not an object", text, 0)
    return parsed

ROOT = Path(__file__).resolve().parent
GUIDELINES = ROOT / "computer_vision_dataset" / "exam_guidelines_student.md"
RUBRIC = ROOT / "computer_vision_dataset" / "stage_3_rubrics_ta.md"
SOLUTIONS_DIR = ROOT / "computer_vision_dataset" / "Solutions"
EXTRACTED_DIR = ROOT / "computer_vision_dataset" / "submissions_extracted"
GRADES_XLSX = ROOT / "computer_vision_dataset" / "Practical_AI_exam_grades.xlsx"
LOG_FILE = ROOT / "local_grading.log"
PROMPTS_DIR = ROOT / "local_prompts"
RESPONSES_DIR = ROOT / "local_responses"
VARIANCE_CSV = ROOT / "local_variance.csv"
LOGS_DIR = ROOT / "local_logs"
RESULTS_DIR = ROOT / "computer_vision_results" / "results"
VARIANCE_DIR = ROOT / "computer_vision_results" / "variance"
ABLATION_TRACKER = ROOT / "computer_vision_results" / "ablation_runs.xlsx"
FEW_SHOT_BANK = ROOT / "computer_vision_dataset" / "few_shot_examples.json"

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"
DEFAULT_API_BASE = "http://localhost:8000/v1"
DEFAULT_API_KEY = "EMPTY"  # vLLM ignores this but the SDK requires it.

# Qwen variants verified for this study (served via vLLM on 4x A100):
#   Qwen/Qwen2.5-Coder-7B-Instruct       (7B,   ~15 GB, small baseline)
#   Qwen/Qwen2.5-Coder-14B-Instruct      (14B,  ~28 GB)
#   Qwen/Qwen2.5-Coder-32B-Instruct      (32B,  ~64 GB)
#   Qwen/Qwen3-Coder-30B-A3B-Instruct    (30B MoE, 3B active, fast inference)
#   Qwen/Qwen3-Coder-Next                (80B MoE, 3B active, 256K ctx)

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
    # --- Minimal-pair variants (L-MP series, added 2026-07-19) ---
    # Word-vs-policy attribution for the strict collapse: mpstrict/mprigorous/
    # mpfair keep the strict persona's POLICY sentences verbatim and vary ONLY
    # the headword; mpnoclause keeps HARSH but removes the no-partial-credit
    # clause (the sentence that contradicts the scaffold's mandated partial-
    # credit scale); mpnharsh keeps the neutral frame and swaps its one
    # adjective. Additive only -- do not modify the five presets above.
    "mpstrict": (
        "You are a STRICT teaching assistant. Award the MINIMUM defensible score "
        "for any task that is incomplete, buggy, or deviates from the rubric. "
        "Never give partial credit if the task does not run correctly."
    ),
    "mprigorous": (
        "You are a RIGOROUS teaching assistant. Award the MINIMUM defensible score "
        "for any task that is incomplete, buggy, or deviates from the rubric. "
        "Never give partial credit if the task does not run correctly."
    ),
    "mpfair": (
        "You are a FAIR teaching assistant. Award the MINIMUM defensible score "
        "for any task that is incomplete, buggy, or deviates from the rubric. "
        "Never give partial credit if the task does not run correctly."
    ),
    "mpnoclause": (
        "You are a HARSH teaching assistant. Award the MINIMUM defensible score "
        "for any task that is incomplete, buggy, or deviates from the rubric."
    ),
    "mpnharsh": (
        "You are a harsh but fair teaching assistant."
    ),
    # --- 2x2 policy-sentence ablation (L-PS series, added 2026-07-21) ---
    # Factorial over the two POLICY sentences of the 'strict' preset, holding the
    # frame "You are a HARSH teaching assistant." fixed:
    #   S1 = "Award the MINIMUM defensible score for any task that is incomplete,
    #         buggy, or deviates from the rubric."
    #   S2 = "Never give partial credit if the task does not run correctly."
    #   neither -> mpframe | S1 only -> mpnoclause | S2 only -> mps2only | both -> mpharsh
    # mpharsh is textually IDENTICAL to 'strict'. It exists so the both-cell is run
    # on the same stack and date as the other three cells: the legacy strict runs
    # (e.g. L-C01, May 2026) are on an unrecorded stack and are not comparable.
    # Additive only -- do not modify the five presets above.
    "mpframe": (
        "You are a HARSH teaching assistant."
    ),
    "mps2only": (
        "You are a HARSH teaching assistant. Never give partial credit if the "
        "task does not run correctly."
    ),
    "mpharsh": (
        "You are a HARSH teaching assistant. Award the MINIMUM defensible score "
        "for any task that is incomplete, buggy, or deviates from the rubric. "
        "Never give partial credit if the task does not run correctly."
    ),
}

QUESTION_META = {
    1: (12.0, 4.0, "Q1_stage3_Finetuning_Solution.ipynb"),
    2: (11.0, 4.0, "Q2_stage3_CNN_Solution.ipynb"),
    3: (12.0, 0.0, "Q3_stage3_Segmentation_Solution.ipynb"),
    4: (0.0, 5.0, "Q4_stage3_Colorization_Solution.ipynb"),
}


def ai_headers() -> list[str]:
    cols: list[str] = []
    for q in (1, 2, 3, 4):
        cols.append(f"AI Q{q} Score")
        if QUESTION_META[q][1] > 0:
            cols.append(f"AI Q{q} Bonus")
        cols.append(f"AI Q{q} Reasoning")
    cols += ["AI Total Score", "AI Total Bonus"]
    return cols


def setup_logging(log_file: Path) -> None:
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


# ---------------------------------------------------------------------------
# Rubric parsing
# ---------------------------------------------------------------------------

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


def load_few_shot_examples(q: int, n: int) -> list[dict]:
    """Return up to n worked-example gradings for question q from
    `few_shot_examples.json`. Raises if the file is missing and n>0."""
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


# ---------------------------------------------------------------------------
# Excel helpers (with resume + atomic save)
# ---------------------------------------------------------------------------

def atomic_save(wb: openpyxl.Workbook, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    wb.save(tmp)
    os.replace(tmp, path)


def init_output_workbook(template: Path, dest: Path) -> int:
    """Clone the master workbook for a fresh run: blank the AI columns, and keep
    everything else — including the values behind the TA formula columns.

    openpyxl never re-emits a formula's cached result, so a bare
    load_workbook -> save round-trip silently empties every formula cell:
    `TA {1,2} - Total Score (out of 35)`, the TA total-bonus columns, and the
    workbook averages. shutil.copy2 preserves those cached values; the
    round-trip immediately after is what destroys them.

    Those cells are the analysis pipeline's TA ground truth, so blanking them
    is what made run_analysis report MAE/bias/CI = NaN — silently, with
    n_valid = 0 — for all 92 workbooks created after cc66e2e. Freeze each
    formula to its cached value so the clone stands on its own.

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
    headers = [c.value for c in ws[1]]
    name_to_col: dict[str, int] = {h: i + 1 for i, h in enumerate(headers) if h}
    next_col = (max(name_to_col.values()) + 1) if name_to_col else 1
    for h in ai_headers():
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


# ---------------------------------------------------------------------------
# Local LLM call (OpenAI-compatible / vLLM)
# ---------------------------------------------------------------------------
#
# vLLM's structured-output JSON schema is *JSON Schema*, not the Gemini-flavoured
# variant (uppercase types). So the schemas below use lowercase "object" /
# "string" / "number" / "array", and `maxLength` works directly.

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

SCHEMA_NO_BREAKDOWN = {
    "type": "object",
    "properties": {
        "score":     {"type": "number"},
        "bonus":     {"type": "number"},
        "reasoning": {"type": "string", "maxLength": 1500},
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


def get_client(api_base: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=api_base, api_key=api_key)


def grade_question(
    client: OpenAI,
    prompt: str,
    use_reasoning: bool,
    use_breakdown: bool,
    temperature: float,
    model: str,
    max_output_tokens: int = 4096,
) -> tuple[dict, str]:
    """Call the local vLLM server. Returns (parsed_json, raw_text)."""
    schema = SCHEMA_WITH_BREAKDOWN if use_breakdown else SCHEMA_NO_BREAKDOWN

    extra_body: dict = {
        # vLLM strict structured output via outlines/lm-format-enforcer.
        "guided_json": schema,
        # Disable top_k explicitly so model-bundled generation_config.json
        # (e.g. Qwen3-Coder-Next's top_k=40) can't sneak in and bias sampling.
        # vLLM treats top_k=-1 as "no truncation".
        "top_k": -1,
    }
    # Qwen3-style thinking toggle via chat-template kwarg; ignored by models
    # whose chat template doesn't accept it.
    if use_reasoning:
        extra_body["chat_template_kwargs"] = {"enable_thinking": True}
    else:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}

    last_err: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=1.0,            # pin top_p to neutral so the model's
                                      # bundled generation_config (e.g.
                                      # Qwen3-Coder-Next: top_p=0.95) can't
                                      # narrow the distribution during
                                      # temperature>0 variance runs.
                max_tokens=max_output_tokens,
                extra_body=extra_body,
            )
            choice = resp.choices[0]
            text = choice.message.content or ""
            if choice.finish_reason == "length":
                raise RuntimeError(
                    f"Local model response truncated at max_tokens "
                    f"({max_output_tokens}); JSON is incomplete."
                )
            return _parse_json_lenient(text), text
        except Exception as e:  # noqa: BLE001
            last_err = e
            # Both truncation and JSON-parse errors are deterministic at temp=0;
            # retrying just burns time. Bail immediately.
            if "truncated at max_tokens" in str(e):
                logging.error(
                    "Local model hit max_tokens on attempt %d; not retrying (deterministic).",
                    attempt,
                )
                break
            if isinstance(e, json.JSONDecodeError) and temperature == 0.0:
                logging.error(
                    "Local model produced invalid JSON on attempt %d at temp=0; "
                    "not retrying (deterministic). Error: %s", attempt, e,
                )
                break
            logging.warning("Local call failed (attempt %d/3): %s", attempt, e)
            time.sleep(2 * attempt)
    raise RuntimeError(f"Local call failed: {last_err}")


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
    client: OpenAI,
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
    max_output_tokens: int,
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
            try:
                result, raw_text = grade_question(
                    client, prompt, use_reasoning, use_breakdown,
                    temperature, model, max_output_tokens,
                )
            except Exception as e:  # noqa: BLE001
                logging.error("Student %s Q%d run %d: grading failed: %s",
                              student_number, q, run_idx, e)
                continue

            if save_io:
                resp_path = RESPONSES_DIR / f"{stem}.json"
                resp_path.parent.mkdir(parents=True, exist_ok=True)
                resp_path.write_text(raw_text, encoding="utf-8")

            if result.get("score") is None:
                # A defaulted 0.0 must be distinguishable from a model-emitted
                # one in the logs: persona-induced format drift dropping the
                # key would otherwise manufacture the flat-zero collapse
                # signature silently (dissection F13).
                logging.warning(
                    "Student %s Q%d run %d: response has no 'score' key "
                    "(keys=%s); recording 0.0 by default",
                    student_number, q, run_idx, sorted(result)[:8],
                )
            try:
                raw_score = float(result.get("score", 0) or 0)
                raw_bonus = float(result.get("bonus", 0) or 0)
            except (TypeError, ValueError) as e:
                logging.error(
                    "Student %s Q%d run %d: malformed score/bonus "
                    "(score=%r bonus=%r): %s; treating run as failed",
                    student_number, q, run_idx,
                    result.get("score"), result.get("bonus"), e,
                )
                continue
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
            if not isinstance(breakdown, list):
                breakdown = [breakdown]
            if breakdown:
                # GLM-4-9B emits bare floats as breakdown items; the awarded
                # points already live in 'score', so a malformed item only
                # costs detail in the reasoning trail. Pre-fix, b.get() on a
                # float crashed the whole student (L-W lost 4-5 per persona).
                reasoning = (
                    reasoning
                    + "\n\nBreakdown:\n"
                    + "\n".join(
                        f"- {b.get('task','?')}: {b.get('awarded','?')}/{b.get('max','?')}"
                        + (f" — {b['note']}" if b.get('note') else "")
                        if isinstance(b, dict) else f"- {b}"
                        for b in breakdown
                    )
                )

            logging.info("Student %s Q%d run %d: score=%s bonus=%s",
                         student_number, q, run_idx, score, bonus)
            run_scores.append(score)
            run_bonuses.append(bonus)
            last_reasoning = reasoning

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
# CLI utilities
# ---------------------------------------------------------------------------

def parse_student_list(spec: str) -> list[int]:
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
            return not default_no
        if ans == "":
            return not default_no
        if ans in ("n", "no"):
            return False
        if ans in ("y", "yes"):
            return True


# ---------------------------------------------------------------------------
# Tracker (computer_vision_results/ablation_runs.xlsx) auto-update
# ---------------------------------------------------------------------------

@contextmanager
def _file_lock(target: Path, timeout_s: float = 60.0, poll_s: float = 0.25):
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
                logging.warning("Tracker %s has no 'run_tag' column.", tracker_path)
                return
            target_row = None
            for r in ws.iter_rows(min_row=2):
                if r[tag_col - 1].value == run_tag:
                    target_row = r[0].row
                    break
            if target_row is None:
                logging.warning("Tracker has no row for tag=%s.", run_tag)
                return
            for key, val in updates.items():
                if key not in headers:
                    continue
                ws.cell(row=target_row, column=headers.index(key) + 1).value = val
            atomic_save(wb, tracker_path)
    except Exception as e:  # noqa: BLE001
        logging.warning("Failed to update tracker (tag=%s): %s", run_tag, e)


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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    student_group = parser.add_mutually_exclusive_group()
    student_group.add_argument("--student", type=int, help="Grade a single student by Number")
    student_group.add_argument("--students", type=str, default=None,
                        help='Comma-separated student numbers or ranges, e.g. "1,2,5" or "1-10"')
    student_group.add_argument("--all", action="store_true", help="Grade every student found in submissions_extracted")
    parser.add_argument("--output-xlsx", type=str, default=None,
                        help="Per-run results workbook (default: Practical_AI_exam_grades.xlsx).")
    parser.add_argument("--save-io", action="store_true",
                        help="Also save prompts/responses under local_prompts/ and local_responses/.")
    parser.add_argument("--with-solution", dest="with_solution", action="store_true")
    parser.add_argument("--no-solution", dest="no_solution", action="store_true")
    parser.add_argument("--with-reasoning", dest="with_reasoning", action="store_true",
                        help="Toggle Qwen3 chat-template thinking on. Ignored by Qwen2.5-Coder.")
    parser.add_argument("--no-reasoning", dest="no_reasoning", action="store_true")
    parser.add_argument("--with-breakdown", dest="with_breakdown", action="store_true")
    parser.add_argument("--no-breakdown", dest="no_breakdown", action="store_true")
    parser.add_argument("--with-guidelines", dest="with_guidelines", action="store_true")
    parser.add_argument("--no-guidelines", dest="no_guidelines", action="store_true")
    parser.add_argument("--strictness", choices=list(STRICTNESS_PRESETS.keys()), default=None)
    parser.add_argument("--model", type=str, default=None,
                        help=f"HuggingFace model ID served by vLLM (default: {DEFAULT_MODEL}).")
    parser.add_argument("--api-base", type=str, default=DEFAULT_API_BASE,
                        help=f"Base URL of the vLLM server (default: {DEFAULT_API_BASE}).")
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY,
                        help="API key (vLLM ignores it but the SDK requires one).")
    parser.add_argument("--max-output-tokens", type=int, default=4096,
                        help="Max tokens per response (default 4096). Local models truncate "
                             "much sooner than Gemini's 65k; 4096 is plenty for JSON + reasoning.")
    parser.add_argument("--wait-for-server", type=int, default=300,
                        help="Seconds to wait for the vLLM server to become reachable at "
                             "startup (default 300 = 5 min). Useful when launching this "
                             "script right after `serve_qwen.sh` while the model is still "
                             "loading. Set to 0 to fail immediately if the server is down.")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--variance-csv", type=str, default=None)
    parser.add_argument("--few-shot", dest="few_shot", type=int, default=0,
                        help="N>0 prepends N worked-example gradings per question "
                             "(loaded from few_shot_examples.json). Default 0.")
    parser.add_argument("--tracker", type=str, default=None,
                        help=f"Path to the ablation tracker xlsx to back-fill "
                             f"with status/metrics on completion. Default: "
                             f"computer_vision_results/{ABLATION_TRACKER.name}.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

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
        use_reasoning = _yn("Enable Qwen reasoning (thinking) mode?", default_no=True)

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

    model = args.model or _ask(f"Model [default {DEFAULT_MODEL}]: ") or DEFAULT_MODEL

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

    run_tag = args.tag

    logging.info("Model = %s   API base = %s   max_output_tokens = %d",
                 model, args.api_base, args.max_output_tokens)
    logging.info("Reference solution will be %s in prompts", "INCLUDED" if use_solution else "EXCLUDED")
    logging.info("Exam guidelines will be %s in prompts", "INCLUDED" if use_guidelines else "EXCLUDED")
    logging.info("Reasoning/thinking is %s", "ENABLED" if use_reasoning else "DISABLED")
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

    if args.output_xlsx:
        output_xlsx = Path(args.output_xlsx).resolve()
    elif args.tag:
        output_xlsx = (RESULTS_DIR / f"{args.tag}.xlsx").resolve()
    else:
        output_xlsx = GRADES_XLSX
    if output_xlsx != GRADES_XLSX and not output_xlsx.exists():
        output_xlsx.parent.mkdir(parents=True, exist_ok=True)
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

    update_ablation_row(tracker_path, run_tag, {
        "status": "running",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    client = get_client(args.api_base, args.api_key)
    # vLLM takes 1-3 min to load big models. Poll the server up to
    # `wait_for_server_s` seconds before giving up, so we can start the grading
    # script in the same Jupyter session right after launching `serve_qwen.sh`
    # without timing the moment.
    wait_for_server_s = args.wait_for_server
    poll_s = 10
    start = time.time()
    while True:
        try:
            client.models.list()
            logging.info("vLLM server reachable at %s", args.api_base)
            break
        except Exception as e:  # noqa: BLE001
            elapsed = time.time() - start
            if elapsed > wait_for_server_s:
                logging.error("vLLM server at %s still unreachable after %ds: %s",
                              args.api_base, int(elapsed), e)
                raise
            logging.info("vLLM not ready yet (%s); retrying in %ds (elapsed %ds / %ds)",
                         type(e).__name__, poll_s, int(elapsed), wait_for_server_s)
            time.sleep(poll_s)

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
                    variance_csv, args.max_output_tokens,
                    few_shot=args.few_shot,
                )
            except Exception as e:  # noqa: BLE001
                logging.exception("Student %s: unhandled error: %s", n, e)
    except BaseException:
        final_status = "failed"
        raise
    finally:
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
