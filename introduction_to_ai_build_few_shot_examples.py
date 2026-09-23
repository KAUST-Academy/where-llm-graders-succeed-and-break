"""Generate `introduction_to_ai_dataset/few_shot_examples.json` — the example bank used by the
Stage 2 few-shot ablation (`--few-shot N` on introduction_to_ai_grade_with_gemini.py /
introduction_to_ai_grade_with_local.py).

Stage 2 counterpart of `build_few_shot_examples.py`. Same sourcing logic; the
exam differs in four ways that matter here:

  - grades ship as a CSV, not an xlsx, with plain `TA n - Qk Grade` columns
  - three questions, not four
  - all three questions carry bonus marks (Stage 3 had bonus on Q1/Q2 only)
  - the scale is ~65 points rather than 35, so the agreement threshold is
    re-derived rather than copied (see below)

## Sourcing

For each question we need K worked examples: the student's code, plus a target
`score`, `bonus` and `reasoning`.

  - Students are restricted to the **two highest-agreement TA pairs**
    (TA_3 / TA_4 at mean |TA1-TA2| = 1.90, TA_5 / TA_6 at 2.02, against an
    exam-wide floor of 5.13). Their TA-averaged totals are the most
    trustworthy human ground truth in the dataset.
  - We further restrict to students whose two TAs agreed within
    MAX_AGREEMENT_GAP on the total.
  - Per-question score/bonus/reasoning come from the best Stage 2 config
    **IG08** (`gemini-3.1-pro-preview`, overall MAE 3.40 — below this exam's
    human floor of 5.13), the direct analogue of Stage 3's D01.

## Why MAX_AGREEMENT_GAP is 2.0 and not 1.0

Stage 3 used |TA1-TA2| <= 1.0 against a floor of 2.61, i.e. 0.38x the floor.
Copying the raw 1.0 here would be a much tighter filter, because this exam's
floor is nearly twice as large. Matching the *ratio* instead gives
0.38 x 5.13 = 1.97, rounded to 2.0, which leaves 57 candidate students across
the two pairs below.

Note this is deliberately a floor-matched threshold, not a percent-of-scale
one: Stage 2 folds bonus marks into its question scores and Stage 3 keeps them
separate, so the two exams' denominators are not like-for-like and any
"percent of scale" comparison between them is invalid.

## Why examples are stratified on exam-wide percentiles, not pool quantiles

Stage 3 picked its K examples at the 40th and 75th quantile *of its eligible
pool*, which happened to land at the 41st and 73rd percentile of that exam —
i.e. representative, mid-range worked examples.

The same code applied here lands at the 16th and 32nd exam percentile,
because on this exam TAs only agree closely on *weak* submissions: every
tight-agreement pool is skewed low (the top pair averages 20.4 against an
exam-wide 29.6, and widening to more pairs makes it worse, not better). That
is a structural property of the data, not a pair-selection artefact.

Copying the raw quantiles would therefore hand the model a bank of examples
roughly 1.4 quartiles harder than Stage 3's, and any few-shot difference could
be attributed to the bank's difficulty rather than to few-shot prompting. So
we match Stage 3's *realized outcome* instead of its literal arithmetic:
examples are the pool members closest to the exam-wide 40th and 75th
percentile. With the two best pairs this reproduces Stage 3's spread (37th-41st
and 72nd), while keeping the high-agreement and small-gap constraints intact.

## Caveat: example students stay in the evaluation set

The few-shot runs grade `--all`, which includes the example students
themselves (their own worked grading appears in their prompt). With K=2 that
is ~6 of 1,038 students — negligible for headline MAE, but flagged in the
paper's limitations. Exclude the chosen students from any few-shot-vs-baseline
per-student comparison if you need it to be airtight.

## What this tests

If the designed prompt (solution + breakdown + persona) is what drives
quality, a naive few-shot prompt with the same backbone plus K worked examples
should be roughly equivalent or worse. If few-shot *beats* the designed
prompt, that is a finding worth reporting.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "introduction_to_ai_dataset"
MASTER = DATA / "Practical_AI_exam_grades.csv"
EXTRACTED = DATA / "submissions_extracted"
LABEL_XLSX = (ROOT / "introduction_to_ai_results" / "results"
              / "IG08__IG08_m-3.1-pro-preview_sol1_gd0_rs0_bd1_str-neutral_t00_n1.xlsx")
OUT = DATA / "few_shot_examples.json"

QUESTIONS = (1, 2, 3)
# The two highest-agreement pairs; one pair alone leaves too few candidates
# near the upper target percentile. See module docstring.
HIGH_AGREEMENT_PAIRS = [("TA_3", "TA_4"), ("TA_5", "TA_6")]
MAX_AGREEMENT_GAP = 2.0   # 0.38x this exam's 5.13 floor; see module docstring
MAX_CODE_CHARS = 6000     # per-example student code cap
K_PER_QUESTION = 2        # how many examples per Q to bake in
# Exam-wide percentiles the examples should sit at (Stage 3 realized 41st/73rd).
TARGET_PERCENTILES = (0.40, 0.75)


def notebook_to_text(nb_path: Path) -> str:
    """Mirror the cell-by-cell text extract used by introduction_to_ai_grade_with_*.py."""
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    out: list[str] = []
    for idx, cell in enumerate(nb.get("cells", []), start=1):
        src = "".join(cell.get("source", []))
        ctype = cell.get("cell_type", "code")
        out.append(f"--- Cell {idx} [{ctype}] ---\n{src}\n")
    return "\n".join(out)


def truncate_code(code: str, max_chars: int = MAX_CODE_CHARS) -> str:
    if len(code) <= max_chars:
        return code
    head = code[:max_chars]
    last_nl = head.rfind("\n")
    if last_nl > max_chars * 0.7:
        head = head[:last_nl]
    return head + "\n\n# … (truncated for few-shot example; original notebook continues)"


def exam_targets() -> list[float]:
    """The TA-average totals sitting at TARGET_PERCENTILES of the whole exam."""
    df = pd.read_csv(MASTER)
    t1 = pd.to_numeric(df["TA 1 - Total Grade"], errors="coerce")
    t2 = pd.to_numeric(df["TA 2 - Total Grade"], errors="coerce")
    avg = ((t1 + t2) / 2).dropna()
    return [float(avg.quantile(p)) for p in TARGET_PERCENTILES]


def candidate_pool() -> pd.DataFrame:
    """Students from the highest-agreement TA pairs within MAX_AGREEMENT_GAP."""
    df = pd.read_csv(MASTER)
    mask = pd.Series(False, index=df.index)
    for a, b in HIGH_AGREEMENT_PAIRS:
        mask = mask | (
            ((df["TA_1_ID"] == a) & (df["TA_2_ID"] == b))
            | ((df["TA_1_ID"] == b) & (df["TA_2_ID"] == a))
        )
    sub = df[mask].copy()
    t1 = pd.to_numeric(sub["TA 1 - Total Grade"], errors="coerce")
    t2 = pd.to_numeric(sub["TA 2 - Total Grade"], errors="coerce")
    sub["ta_avg"] = (t1 + t2) / 2
    sub["ta_gap"] = (t1 - t2).abs()
    sub = sub[(sub["ta_gap"] <= MAX_AGREEMENT_GAP) & sub["ta_avg"].notna()]
    sub = sub[sub["ta_avg"] > 0]
    return sub.sort_values("ta_avg").reset_index(drop=True)


def pick_for_question(pool: pd.DataFrame, q: int, labels: pd.DataFrame, k: int,
                      targets: list[float]) -> list[int]:
    """Pick K students for question q closest to the exam-wide target
    percentiles, requiring (a) Qq.ipynb exists and (b) the label run has a
    score & reasoning."""
    eligible = []
    for _, r in pool.iterrows():
        n = int(r["Number"])
        if not (EXTRACTED / str(n) / f"Q{q}.ipynb").exists():
            continue
        if n not in labels.index:
            continue
        d = labels.loc[n]
        if pd.isna(d.get(f"AI Q{q} Score")) or not isinstance(d.get(f"AI Q{q} Reasoning"), str):
            continue
        eligible.append((n, float(r["ta_avg"])))
    if not eligible:
        return []
    if len(eligible) <= k:
        return [n for n, _ in eligible]
    picks: list[int] = []
    for t in targets[:k]:
        cand = [e for e in eligible if e[0] not in picks]
        if cand:
            picks.append(min(cand, key=lambda e: abs(e[1] - t))[0])
    return picks


def main() -> None:
    if not LABEL_XLSX.exists():
        raise SystemExit(f"Missing label xlsx at {LABEL_XLSX}. "
                         f"Few-shot example labels rely on it.")
    if not EXTRACTED.is_dir():
        raise SystemExit(f"submissions_extracted not found at {EXTRACTED}")

    pool = candidate_pool()
    targets = exam_targets()
    labels = pd.read_excel(LABEL_XLSX).set_index("Number")
    print(f"Candidate pool ({HIGH_AGREEMENT_PAIRS}, |TA1-TA2|<={MAX_AGREEMENT_GAP}): "
          f"{len(pool)} students")
    print(f"Target TA-avg totals (exam percentiles {TARGET_PERCENTILES}): "
          f"{[round(t, 1) for t in targets]}")

    bank: dict[str, list[dict]] = {f"Q{q}": [] for q in QUESTIONS}
    for q in QUESTIONS:
        chosen = pick_for_question(pool, q, labels, K_PER_QUESTION, targets)
        print(f"  Q{q}: picked {chosen}")
        for n in chosen:
            code = notebook_to_text(EXTRACTED / str(n) / f"Q{q}.ipynb")
            d_row = labels.loc[n]
            score = float(d_row[f"AI Q{q} Score"])
            # The result xlsx stores reasoning with the per-task breakdown
            # appended ("...\n\nBreakdown:\n- ..."). Strip that appendix and
            # cap at the schema's 1500-char limit, so the worked examples
            # demonstrate the exact output format the live prompt demands.
            raw_reasoning = str(d_row[f"AI Q{q} Reasoning"]).split("\n\nBreakdown:", 1)[0]
            reasoning = re.sub(r"\s+", " ", raw_reasoning).strip()[:1500]
            # Unlike Stage 3, every question on this exam carries bonus marks.
            bonus_val = d_row.get(f"AI Q{q} Bonus")
            bonus = float(bonus_val) if (bonus_val is not None and not pd.isna(bonus_val)) else 0.0
            ta_avg = float(pool.loc[pool["Number"] == n, "ta_avg"].iloc[0])
            bank[f"Q{q}"].append({
                "provenance": {
                    "student_number": n,
                    "ta_avg_total": ta_avg,
                    "ta_pairs": [list(p) for p in HIGH_AGREEMENT_PAIRS],
                    "score_source": "IG08 (gemini-3.1-pro-preview, overall MAE 3.40)",
                },
                "code": truncate_code(code),
                "score": score,
                "bonus": bonus,
                "reasoning": reasoning,
            })

    for q in QUESTIONS:
        if len(bank[f"Q{q}"]) < K_PER_QUESTION:
            print(f"  WARN: Q{q} only has {len(bank[f'Q{q}'])} examples; "
                  f"requested K_PER_QUESTION={K_PER_QUESTION}")

    OUT.write_text(json.dumps(bank, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT} with "
          f"{sum(len(v) for v in bank.values())} total examples "
          f"({K_PER_QUESTION} target × {len(QUESTIONS)} questions = "
          f"{K_PER_QUESTION * len(QUESTIONS)} max).")


if __name__ == "__main__":
    main()
