#!/usr/bin/env python3
"""Evaluate a fine-tuned grader through a vLLM server (fast path).

Same inputs/outputs as ``eval_lora.py`` -- reads the held-out split, writes a
result workbook in the standard ablation shape into ``computer_vision_results/results/`` and
prints MAE vs the TA average -- but instead of HuggingFace ``generate()`` at
batch size 1, it fires many requests CONCURRENTLY at a vLLM server.

Why this is ~10-30x faster:
  * vLLM continuous-batches concurrent requests (hence ``--concurrency``; a
    sequential client gets none of the benefit).
  * ``--enable-prefix-caching`` on the server makes the ~18k-char shared prefix
    (guidelines + rubric + reference solution) nearly free after the first hit.
  * ``structured_outputs.json`` enforces the schema, so responses parse first try.

The server (see ``serve_lora.sh``) can host the base model AND the LoRA adapter
at once, so ``--model`` just selects which one to grade with:
    --model Qwen/Qwen2.5-Coder-7B-Instruct   -> base   (baseline)
    --model ft                               -> adapter (fine-tuned)

Example:
    python finetune/eval_vllm.py --model ft --prompt-style breakdown \
        --eval-students finetune/data_bd/eval_students.json --tag FT-A07bd-lora
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openpyxl
from openai import OpenAI

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))  # vendored common.py lives beside this script

from common import (  # noqa: E402
    QUESTION_META,
    SOLUTIONS_DIR,
    EXTRACTED_DIR,
    GRADES_XLSX,
    GUIDELINES,
    RESULTS_DIR,
    SCHEMA_WITH_BREAKDOWN,
    load_rubric_sections,
    notebook_to_text,
    ensure_ai_columns,
    build_student_row_index,
    atomic_save,
    compute_run_summary,
    _parse_json_lenient,
    build_prompt,
)
from build_finetune_data import build_scores_prompt, truncate_code  # noqa: E402
from build_breakdown_data import build_bd_prompt  # noqa: E402
import exams  # noqa: E402

DEFAULT_API_BASE = "http://localhost:8000/v1"

# vLLM structured outputs take plain JSON Schema (lowercase types).
SCHEMA_MARKS = {
    "type": "object",
    "properties": {"score": {"type": "number"}, "bonus": {"type": "number"}},
    "required": ["score", "bonus"],
}

# Breakdown-ONLY: no `reasoning` key, matching build_breakdown_data.py defaults.
# Deliberately free of maxLength/minItems/maxItems: some structured-output
# backends (xgrammar) don't support those keywords, and they're only guardrails --
# the MAE reads the top-level score/bonus, never the breakdown contents.
SCHEMA_BREAKDOWN = {
    "type": "object",
    "properties": {
        "score": {"type": "number"},
        "bonus": {"type": "number"},
        "task_breakdown": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "awarded": {"type": "number"},
                    "max": {"type": "number"},
                },
                "required": ["task", "awarded", "max"],
            },
        },
    },
    "required": ["score", "bonus", "task_breakdown"],
}

SCHEMAS = {
    "marks": SCHEMA_MARKS,
    "breakdown": SCHEMA_BREAKDOWN,
    "breakdown_reasoning": SCHEMA_WITH_BREAKDOWN,
}
AUTO_TOKENS = {"marks": 64, "breakdown": 1024, "breakdown_reasoning": 1536}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="Model name AS SERVED by vLLM: the base HF id for the "
                         "baseline, or the --lora-modules name (e.g. 'ft') for the "
                         "fine-tuned adapter.")
    ap.add_argument("--api-base", default=DEFAULT_API_BASE)
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--dataset", choices=list(exams.EXAMS), default="cv",
                    help="Which exam to GRADE. Independent of what the adapter "
                         "was trained on -- that is the whole point: pass "
                         "--dataset intro with a cv-trained adapter to measure "
                         "transfer.")
    ap.add_argument("--eval-students", type=Path, default=None,
                    help="Default: the matching finetune/data[_<dataset>]/"
                         "eval_students.json. A pooled 'both' file is accepted; "
                         "the --dataset exam's slice is taken from it.")
    ap.add_argument("--tag", required=True, help="Run tag -> finetune/results/<tag>.xlsx")
    ap.add_argument("--output-xlsx", type=Path, default=None)
    ap.add_argument("--prompt-style", choices=list(SCHEMAS), default="marks",
                    help="Must match how the adapter was trained.")
    ap.add_argument("--questions", default=None,
                    help="Default: every question the --dataset exam has.")
    ap.add_argument("--concurrency", type=int, default=32,
                    help="In-flight requests. This is what lets vLLM batch; "
                         "raise for more throughput if the server keeps up.")
    ap.add_argument("--max-new-tokens", type=int, default=0, help="0 = auto per style.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--no-solution", action="store_true")
    ap.add_argument("--no-guidelines", action="store_true")
    ap.add_argument("--strictness", default="neutral")
    ap.add_argument("--max-student-chars", type=int, default=24000)
    ap.add_argument("--wait-for-server", type=int, default=1800,
                    help="Seconds to wait for vLLM to come up (big models load slowly).")
    return ap.parse_args()


def make_prompt(style, q, rubric, guidelines, code, solution, use_guidelines,
                strictness, exam=None):
    exam = exam or exams.CV
    if style == "breakdown":
        return build_bd_prompt(q, rubric, guidelines, code, solution,
                               use_guidelines, strictness, exam)
    if style == "breakdown_reasoning":
        return build_prompt(q=q, rubric_section=rubric, guidelines=guidelines,
                            student_code=code, solution_code=solution,
                            use_breakdown=True, use_guidelines=use_guidelines,
                            strictness=strictness, exam=exam)
    return build_scores_prompt(q, rubric, guidelines, code, solution,
                               use_guidelines, strictness, exam)


def _extract_json(text: str) -> dict:
    """Parse a model response that should be a JSON object.

    Falls back to slicing out the first '{'..last '}' so a markdown fence
    (```json ... ```) or a prose preamble still parses. Raises if nothing works."""
    try:
        return _parse_json_lenient(text)
    except Exception:
        a, b = text.find("{"), text.rfind("}")
        if 0 <= a < b:
            return _parse_json_lenient(text[a:b + 1])
        raise


def grade_one(client, model, prompt, schema, max_tokens, temperature) -> dict:
    """One vLLM call with schema-guided decoding. Returns {} on hard failure."""
    extra_body = {
        # vLLM >= 0.16 renamed the structured-output field. The old `guided_json`
        # is NOT rejected -- it is silently IGNORED ("fields present in the
        # request but ignored"), which drops schema enforcement without any
        # error. Use `structured_outputs.json`, per
        # vllm/entrypoints/openai/chat_completion/protocol.py.
        "structured_outputs": {"json": schema},
        "top_k": -1,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    raw = ""
    for attempt in range(1, 4):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=1.0,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
            choice = resp.choices[0]
            raw = choice.message.content or ""
            if choice.finish_reason == "length":
                # Truncated JSON is unparseable; retrying at temp=0 is pointless.
                print(f"    ! truncated at max_tokens={max_tokens}", flush=True)
                return {}
            return _extract_json(raw)
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                # Show what actually came back -- a parse error here almost always
                # means the schema was not enforced, not that the model is broken.
                snippet = raw[:160].replace("\n", "\\n") if raw else "<empty>"
                print(f"    ! failed after 3 attempts: {type(e).__name__}: {e}"
                      f" | raw={snippet!r}", flush=True)
                return {}
            time.sleep(2 * attempt)
    return {}


def main() -> None:
    args = parse_args()
    style = args.prompt_style
    schema = SCHEMAS[style]
    max_tokens = args.max_new_tokens or AUTO_TOKENS[style]
    exam = exams.get_exam(args.dataset)
    questions = (sorted(int(x) for x in args.questions.split(",") if x.strip())
                 if args.questions else list(exam.questions))
    bad = [q for q in questions if q not in exam.question_meta]
    if bad:
        raise SystemExit(f"{exam.key} has no question(s) {bad}; it has Q{list(exam.questions)}")

    students_path = args.eval_students or (
        HERE / ("data" if args.dataset == "cv" else f"data_{args.dataset}")
        / "eval_students.json")
    eval_students = exams.load_eval_students(students_path, exam)

    print(f"vLLM eval | model={args.model} style={style} max_tokens={max_tokens} "
          f"concurrency={args.concurrency}")
    print(f"  grading {exam.key}: {exams.describe(exam)}")
    print(f"  students={len(eval_students)} questions={questions} "
          f"from {students_path}")

    rubric_sections = exams.load_rubric_sections(exam)
    guidelines = exams.load_guidelines(exam)
    use_solution = not args.no_solution
    # intro ships no guidelines file; never emit an empty guidelines block.
    use_guidelines = (not args.no_guidelines) and exam.has_guidelines
    solutions = exams.load_solutions(exam, use_solution, notebook_to_text)

    client = OpenAI(base_url=args.api_base, api_key=args.api_key)
    deadline = time.time() + args.wait_for_server
    while True:
        try:
            names = [m.id for m in client.models.list().data]
            print(f"  server up; serving: {names}")
            if args.model not in names:
                print(f"  WARNING: '{args.model}' not in served models {names}; "
                      f"the request will likely 404.")
            break
        except Exception as e:  # noqa: BLE001
            if time.time() > deadline:
                raise SystemExit(f"vLLM not reachable at {args.api_base}: {e}")
            time.sleep(10)

    # ---- build the work list -------------------------------------------------
    jobs = []          # (student, q, prompt)
    missing = []       # (student, q) with no notebook -> score 0
    for s in eval_students:
        sdir = exam.extracted_dir / str(s)
        for q in questions:
            nb = sdir / f"Q{q}.ipynb"
            if not nb.exists():
                missing.append((s, q))
                continue
            code = truncate_code(notebook_to_text(nb), args.max_student_chars)
            jobs.append((s, q, make_prompt(style, q, rubric_sections[q], guidelines,
                                           code, solutions[q], use_guidelines,
                                           args.strictness, exam)))
    print(f"  {len(jobs)} generations to run ({len(missing)} cells have no submission)")

    # ---- fire them concurrently (this is what makes vLLM batch) --------------
    t0 = time.time()
    done = [0]

    def work(job):
        s, q, prompt = job
        res = grade_one(client, args.model, prompt, schema, max_tokens, args.temperature)
        done[0] += 1
        n = done[0]
        if n % 25 == 0 or n == len(jobs):
            el = time.time() - t0
            rate = n / el if el else 0
            eta = (len(jobs) - n) / rate if rate else 0
            print(f"    {n}/{len(jobs)}  {rate:.2f} gen/s  ETA {eta/60:.1f} min", flush=True)
        return (s, q, res)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(work, jobs))
    print(f"  generation done in {(time.time()-t0)/60:.1f} min")

    # ---- write the workbook (single-threaded; openpyxl is not thread-safe) ---
    out_xlsx = (args.output_xlsx.resolve() if args.output_xlsx
                else (RESULTS_DIR / f"{args.tag}.xlsx").resolve())
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    exams.init_results_workbook(exam, out_xlsx)
    wb = openpyxl.load_workbook(out_xlsx)
    ws = wb.active
    hdrs = [c.value for c in ws[1]]
    for r in range(2, ws.max_row + 1):
        for i, h in enumerate(hdrs):
            if h and str(h).startswith("AI "):
                ws.cell(row=r, column=i + 1).value = None
    cols = ensure_ai_columns(ws, exam)
    row_index = build_student_row_index(ws)

    per_student: dict[int, dict[int, tuple[float, float]]] = {}
    n_failed = 0
    for s, q, res in results:
        max_score, max_bonus, _ = exam.question_meta[q]
        if not res:
            n_failed += 1
        raw_s = float(res.get("score", 0) or 0)
        raw_b = float(res.get("bonus", 0) or 0)
        per_student.setdefault(s, {})[q] = (
            max(0.0, min(raw_s, max_score)),
            max(0.0, min(raw_b, max_bonus)),
        )
    for s, q in missing:
        per_student.setdefault(s, {})[q] = (0.0, 0.0)

    for s, qmap in per_student.items():
        row = row_index.get(s)
        if row is None:
            continue
        total_s = total_b = 0.0
        for q, (sc, bo) in sorted(qmap.items()):
            ws.cell(row=row, column=cols[f"AI Q{q} Score"], value=sc)
            bcol = cols.get(f"AI Q{q} Bonus")
            if bcol:
                ws.cell(row=row, column=bcol, value=bo)
            ws.cell(row=row, column=cols[f"AI Q{q} Reasoning"], value=f"(vLLM FT {style})")
            total_s += sc
            total_b += bo
        ws.cell(row=row, column=cols["AI Total Score"], value=total_s)
        ws.cell(row=row, column=cols["AI Total Bonus"], value=total_b)
    atomic_save(wb, out_xlsx)

    summary = exams.run_summary(exam, out_xlsx)
    print("\n== Eval summary (held-out split) ==")
    print(f"  model        : {args.model}")
    print(f"  n_graded     : {summary['n_graded']}")
    print(f"  mean total   : {summary['mean_total_score']}")
    print(f"  MAE vs TA-avg: {summary['MAE_vs_TA_avg']}")
    if n_failed:
        print(f"  !! {n_failed}/{len(jobs)} generations failed -> counted as 0; "
              f"MAE is optimistic-biased. Re-run or raise --max-new-tokens.")
    print(f"  output       : {out_xlsx}")

    # Per-question MAE (the metric for a leave-one-question-out test).
    wbq = openpyxl.load_workbook(out_xlsx, data_only=True)
    wsq = wbq.active
    Hq = [c.value for c in wsq[1]]
    def qcol(name):
        return Hq.index(name) + 1 if name in Hq else None
    # intro folds bonus into one TA Grade per question, so the AI side must add
    # its bonus there; cv keeps Score/Bonus separate and compares score-only.
    combined = exam.ta_mode == "combined"
    ta_field = "Grade" if combined else "Score"
    print("  per-question MAE (AI Qk vs TA Qk avg):")
    for q in questions:
        c_ai = qcol(f"AI Q{q} Score")
        c_aib = qcol(f"AI Q{q} Bonus") if combined else None
        c_ta = [qcol(f"TA {t} - Q{q} {ta_field}") for t in (1, 2)]
        if c_ai is None:
            continue
        diffs = []
        for r in wsq.iter_rows(min_row=2, values_only=True):
            v = r[c_ai - 1]
            if v in (None, ""):
                continue
            av = float(v)
            if c_aib and isinstance(r[c_aib - 1], (int, float)):
                av += float(r[c_aib - 1])
            tv = [r[c - 1] for c in c_ta if c and isinstance(r[c - 1], (int, float))]
            if tv:
                diffs.append(av - sum(tv) / len(tv))
        if diffs:
            mae = sum(abs(d) for d in diffs) / len(diffs)
            bias = sum(diffs) / len(diffs)
            print(f"    Q{q}: n={len(diffs)}  MAE={mae:.3f}  bias={bias:+.3f}")


if __name__ == "__main__":
    main()
