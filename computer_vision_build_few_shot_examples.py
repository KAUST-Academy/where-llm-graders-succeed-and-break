"""Generate `computer_vision_dataset/few_shot_examples.json` — the example bank used by the K-series
ablations (`--few-shot N` flag on computer_vision_grade_with_gemini.py / computer_vision_grade_with_local.py).

## Sourcing

For each question we need K worked examples, each consisting of:
  - the student's code for that question (truncated to keep prompts bounded)
  - a target `score`, `bonus`, and `reasoning`

The most defensible source we have:
  - Students are restricted to the **highest-agreement TA pair** (TA_6 / TA_16,
    inter-grader MAE 0.85). Their TA-averaged totals are the most trustworthy
    human ground truth in the dataset.
  - We further restrict to students where the two TAs agreed within 1 point
    on the total (|TA1 − TA2| < 1).
  - Per-question score/bonus/reasoning come from the best AI config **D01**
    (`gemini-3-flash-preview`, overall MAE 1.64 — *below* the human floor of
    2.61). For these specifically-chosen students D01's per-question values
    are the closest signal we have to "what a top-quality grading looks like".

## Caveat: example students stay in the evaluation set

The K-series runs grade `--all`, which includes the example students
themselves (their own worked grading appears in their prompt). With K=2 that
is ~5 of 570 students — negligible for headline MAE, but flagged in the
paper's limitations. Exclude `CHOSEN` students from any K-vs-baseline
per-student comparison if you need it to be airtight.

## What the K-series tests

If our designed prompt (with solution + guidelines + breakdown + persona) is
fundamentally what's driving quality, then a naive few-shot prompt with the
same backbone + K worked examples should be roughly equivalent or worse.
If few-shot *beats* the designed prompt, that's a finding worth reporting.

## Customisation

Edit `K_PER_QUESTION` to add more examples per question. Edit `CHOSEN_NUMBERS`
to hand-pick students if you want to vet them yourself. The JSON file this
script writes can also be edited by hand — the graders only read the JSON,
they don't re-derive it.

## Token budget

Student code is truncated to `MAX_CODE_CHARS` per example. Qwen2.5-Coder-7B
has a 32k context limit; one extra example per question typically adds
~4-8k tokens of prompt, so K=2 is the sweet spot.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
MASTER = ROOT / "computer_vision_dataset" / "Practical_AI_exam_grades.xlsx"
EXTRACTED = ROOT / "computer_vision_dataset" / "submissions_extracted"
D01_XLSX = (ROOT / "computer_vision_results" / "results"
            / "D01__D01_m-3-flash-preview_sol1_gd1_rs0_bd1_str-neutral_t00_n1.xlsx")
OUT = ROOT / "computer_vision_dataset" / "few_shot_examples.json"

HIGH_AGREEMENT_PAIR = ("TA_6", "TA_16")
MAX_AGREEMENT_GAP = 1.0   # |TA1 − TA2| <= this
MAX_CODE_CHARS = 6000     # per-example student code cap
K_PER_QUESTION = 2        # how many examples per Q to bake in


def notebook_to_text(nb_path: Path) -> str:
    """Mirror the cell-by-cell text extract used by grade_with_*.py."""
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
    # Cut on a line boundary near the cap, leave a clear marker
    head = code[:max_chars]
    last_nl = head.rfind("\n")
    if last_nl > max_chars * 0.7:
        head = head[:last_nl]
    return head + "\n\n# … (truncated for few-shot example; original notebook continues)"


def candidate_pool() -> pd.DataFrame:
    """All students from the highest-agreement TA pair with |TA1−TA2| <= 1."""
    df = pd.read_excel(MASTER)
    mask = (
        ((df["TA_1_ID"] == HIGH_AGREEMENT_PAIR[0]) & (df["TA_2_ID"] == HIGH_AGREEMENT_PAIR[1]))
        | ((df["TA_1_ID"] == HIGH_AGREEMENT_PAIR[1]) & (df["TA_2_ID"] == HIGH_AGREEMENT_PAIR[0]))
    )
    sub = df[mask].copy()
    t1 = pd.to_numeric(sub["TA 1 - Total Score (out of 35)"], errors="coerce")
    t2 = pd.to_numeric(sub["TA 2 - Total Score (out of 35)"], errors="coerce")
    sub["ta_avg"] = (t1 + t2) / 2
    sub["ta_gap"] = (t1 - t2).abs()
    sub = sub[(sub["ta_gap"] <= MAX_AGREEMENT_GAP) & sub["ta_avg"].notna()]
    sub = sub[sub["ta_avg"] > 0]
    return sub.sort_values("ta_avg").reset_index(drop=True)


def pick_for_question(pool: pd.DataFrame, q: int, d01: pd.DataFrame, k: int) -> list[int]:
    """Pick K students for question q stratified across the TA-avg range,
    requiring (a) Qq.ipynb exists and (b) D01 has a non-NaN score & reasoning."""
    eligible = []
    for _, r in pool.iterrows():
        n = int(r["Number"])
        if not (EXTRACTED / str(n) / f"Q{q}.ipynb").exists():
            continue
        if n not in d01.index:
            continue
        d = d01.loc[n]
        if pd.isna(d.get(f"AI Q{q} Score")) or not isinstance(d.get(f"AI Q{q} Reasoning"), str):
            continue
        eligible.append((n, float(r["ta_avg"])))
    if not eligible:
        return []
    if len(eligible) <= k:
        return [n for n, _ in eligible]
    # Stratified pick across the eligible range (mid + high)
    eligible.sort(key=lambda x: x[1])
    quantiles = [0.40, 0.75, 0.20, 0.90][:k]
    return [eligible[int(q_ * (len(eligible) - 1))][0] for q_ in quantiles]


def main() -> None:
    if not D01_XLSX.exists():
        raise SystemExit(f"Missing D01 result xlsx at {D01_XLSX}. "
                         f"Few-shot example labels rely on it.")
    if not EXTRACTED.is_dir():
        raise SystemExit(f"submissions_extracted not found at {EXTRACTED}")

    pool = candidate_pool()
    d01 = pd.read_excel(D01_XLSX).set_index("Number")
    print(f"Candidate pool ({HIGH_AGREEMENT_PAIR}, |TA1-TA2|<={MAX_AGREEMENT_GAP}): "
          f"{len(pool)} students")

    bank: dict[str, list[dict]] = {f"Q{q}": [] for q in (1, 2, 3, 4)}
    for q in (1, 2, 3, 4):
        chosen = pick_for_question(pool, q, d01, K_PER_QUESTION)
        print(f"  Q{q}: picked {chosen}")
        for n in chosen:
            code = notebook_to_text(EXTRACTED / str(n) / f"Q{q}.ipynb")
            d_row = d01.loc[n]
            score = float(d_row[f"AI Q{q} Score"])
            # The result xlsx stores reasoning with the per-task breakdown
            # appended ("...\n\nBreakdown:\n- ..."). Strip that appendix and
            # cap at the schema's 1500-char limit, so the worked examples
            # demonstrate the exact output format the live prompt demands
            # (one short paragraph, no bullet lists).
            raw_reasoning = str(d_row[f"AI Q{q} Reasoning"]).split("\n\nBreakdown:", 1)[0]
            reasoning = re.sub(r"\s+", " ", raw_reasoning).strip()[:1500]
            bonus_val = d_row.get(f"AI Q{q} Bonus") if q in (1, 2) else None
            bonus = float(bonus_val) if (bonus_val is not None and not pd.isna(bonus_val)) else 0.0
            ta_avg = float(pool.loc[pool["Number"] == n, "ta_avg"].iloc[0])
            bank[f"Q{q}"].append({
                "provenance": {
                    "student_number": n,
                    "ta_avg_total": ta_avg,
                    "ta_pair": list(HIGH_AGREEMENT_PAIR),
                    "score_source": "D01 (gemini-3-flash-preview, overall MAE 2.25)",
                },
                "code": truncate_code(code),
                "score": score,
                "bonus": bonus,
                "reasoning": reasoning,
            })

    for q in (1, 2, 3, 4):
        if len(bank[f"Q{q}"]) < K_PER_QUESTION:
            print(f"  WARN: Q{q} only has {len(bank[f'Q{q}'])} examples; "
                  f"requested K_PER_QUESTION={K_PER_QUESTION}")

    OUT.write_text(json.dumps(bank, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT} with "
          f"{sum(len(v) for v in bank.values())} total examples "
          f"({K_PER_QUESTION} target × 4 questions = {K_PER_QUESTION * 4} max).")


if __name__ == "__main__":
    main()
