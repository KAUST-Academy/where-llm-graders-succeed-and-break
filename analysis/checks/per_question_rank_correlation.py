#!/usr/bin/env python3
"""Item-level (per-question) rank correlations, AI vs graders.

The paper reports only total-level agreement (MAE, bias, and the
`AI_TA_pearson_r` column of the master CSVs); no rank correlation is printed
anywhere, and nothing is reported per question.  We add
item-level rank correlations so that per-criterion error cancellation is
visible.  This script computes, for the same run set as per_question_signed_error.py:

  CV  (35-pt base scale, Q1-Q3; Q4 is bonus-only and has no TA Score column)
        headline : D01, D02, F02, B03, A01, C01
        matched  : the 17 open-weights neutral/strict pairs of tab:brittle
                   (sections/appendix.tex lines 47-63; the two closed
                   reference rows C01 and F03 are not part of the "17 matched
                   pairs" -- C01 is already a headline run, F03 is not in the
                   run list here and is left out)
  ML  (>=65-pt scale, Q1-Q3 incl. per-question bonus)
        headline : IG08, IG07, IG01, IG05
        matched  : the 17 pairs of
                   analysis/introduction_to_ai_make_paper_tables.py::default_pairs

per question q in {1,2,3} and at total level:

  Spearman rho  (scipy.stats.spearmanr -- MIDRANK ties, the standard
                 definition; grade vectors are heavily tied, so the
                 argsort-of-argsort helper in
                 analysis/introduction_to_ai_run_analysis.py::spearman, which
                 breaks ties ordinally by position, is NOT used.  Both are
                 printed side by side in the audit block so the difference is
                 on the record.)
  Pearson  r    (scipy.stats.pearsonr)
  n             (students actually entering the correlation)

over exactly the students the paper's total-level metric uses.

Columns / rows read
-------------------
CV  workbooks computer_vision_results/results/<RID>__*.xlsx
      AI total     "AI Total Score" (Q1-Q3 base, 35 pts); where that cell is
                   blank but "AI Q1..Q3 Score" are all present the total is
                   rebuilt as their sum -- exactly
                   computer_vision_run_analysis.build_master_table.
      AI question  "AI Q{q} Score"           (q = 1,2,3)
      TA total     mean of the two TA base totals, each REBUILT as
                   "TA n - Q1 Score" + "TA n - Q2 Score" + "TA n - Q3 Score"
                   (computer_vision_run_analysis.ta_combined; the workbooks'
                   "TA n - Total Score (out of 35)" cells are Excel formulas
                   whose cached values were dropped on rewrite).
      TA question  mean of "TA 1 - Q{q} Score" and "TA 2 - Q{q} Score"
                   (computer_vision_run_analysis.ta_per_question).
      rows used    AI total present AND TA total present  (= n_valid_vs_TA).
ML  workbooks introduction_to_ai_results/results/<RID>__*.xlsx, resolved
      through the tracker's output_xlsx column
      (introduction_to_ai_make_paper_tables.find_workbook) because the earliest
      run_ids differ from the filename stem.
      AI total     "AI Total Score" + "AI Total Bonus"
      AI question  "AI Q{q} Score" + "AI Q{q} Bonus"
      TA total     mean of "TA 1 - Total Grade" and "TA 2 - Total Grade"
      TA question  mean of "TA 1 - Q{q} Grade" and "TA 2 - Q{q} Grade"
                   (both from introduction_to_ai_dataset/Practical_AI_exam_grades.csv
                   via introduction_to_ai_run_analysis.{ta_truth,ta_per_question})
      rows used    AI total present AND student in the ground-truth CSV
                   (= n_valid_vs_TA).

A correlation is SKIPPED, not reported as zero, when either vector is constant
(n_distinct == 1) or n < 3 -- the refusal runs that award every student the
same mark have no correlation to report.  Each skip is named with its reason.

Self-checks (all must pass, else the script asserts)
  * the total-level Pearson r reproduces the `AI_TA_pearson_r` column of
    analysis/computer_vision_master_comparison.csv and
    analysis/introduction_to_ai_master_comparison.csv to 3 dp (the CSV stores
    4 dp), for every run where the CSV has a value;
  * n reproduces `n_valid_vs_TA`;
  * MAE reproduces `MAE_vs_TA_avg` to 3 dp, so the student set is the paper's;
  * the CV per-question MAE reproduces `Q{q}_MAE_vs_TA` to 3 dp on the
    per-question mask the CSV uses.

Gap rule for "the total hides a weak item"
  gap = total-level Spearman rho  minus  the weakest per-question Spearman rho
  A run is FLAGGED when gap >= 0.15 (both quantities defined).

Run:  python3 analysis/checks/per_question_rank_correlation.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

import computer_vision_run_analysis as ca            # noqa: E402
import introduction_to_ai_run_analysis as ra         # noqa: E402
import introduction_to_ai_make_paper_tables as mt    # noqa: E402

CV_CSV = ROOT / "analysis" / "computer_vision_master_comparison.csv"
ML_CSV = ROOT / "analysis" / "introduction_to_ai_master_comparison.csv"
CV_RESULTS = ROOT / "computer_vision_results" / "results"

CV_HEADLINE = ["D01", "D02", "F02", "B03", "A01", "C01"]
ML_HEADLINE = ["IG08", "IG07", "IG01", "IG05"]

GAP = 0.15          # total rho - weakest question rho, the flag threshold
TOL = 5e-4          # 3-dp agreement


# --------------------------------------------------------------------------
# correlation primitives
# --------------------------------------------------------------------------
def corr(x, y):
    """(rho, r, n, skip_reason).  rho/r are None when undefined."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[m], y[m]
    n = int(len(x))
    if n < 3:
        return None, None, n, f"n={n} < 3"
    if len(np.unique(x)) == 1:
        return None, None, n, f"AI column constant (all = {x[0]:g})"
    if len(np.unique(y)) == 1:
        return None, None, n, f"TA column constant (all = {y[0]:g})"
    return (float(stats.spearmanr(x, y).statistic),
            float(stats.pearsonr(x, y).statistic), n, None)


def ordinal_spearman(x, y):
    """The repo helper's tie-BREAKING variant, for the audit block only."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = ~np.isnan(x) & ~np.isnan(y)
    if m.sum() < 3:
        return float("nan")
    rk = lambda v: np.argsort(np.argsort(v)).astype(float)
    return float(np.corrcoef(rk(x[m]), rk(y[m]))[0, 1])


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def cv_workbook(rid: str) -> Path:
    hits = sorted(CV_RESULTS.glob(f"{rid}__*.xlsx"))
    assert len(hits) == 1, (rid, hits)
    return hits[0]


def cv_run(rid: str) -> dict:
    """AI/TA totals and per-question vectors on the paper's student set."""
    df = ca.load_result(cv_workbook(rid)).set_index("Number")
    df = ca.maybe_fill_from_variance(df, rid)
    ta_avg = pd.Series(ca.ta_combined(df.reset_index()).values, index=df.index)
    ai_total = pd.to_numeric(df["AI Total Score"], errors="coerce")
    base = pd.concat([pd.to_numeric(df[f"AI Q{q} Score"], errors="coerce")
                      for q in (1, 2, 3)], axis=1)
    rec = ai_total.isna() & base.notna().all(axis=1)
    if rec.any():
        ai_total = ai_total.where(~rec, base.sum(axis=1))
    valid = ai_total.notna() & ta_avg.notna()          # == n_valid_vs_TA
    out = {"rid": rid, "ai_total": ai_total[valid].values,
           "ta_total": ta_avg[valid].values, "q": {}, "q_csvmask": {}}
    for q in (1, 2, 3):
        ai_q = pd.to_numeric(df[f"AI Q{q} Score"], errors="coerce")
        ta_q = ca.ta_per_question(df, q)
        out["q"][q] = (ai_q[valid].values, ta_q[valid].values)
        mq = ai_q.notna() & ta_q.notna()               # the CSV's own mask
        out["q_csvmask"][q] = (ai_q[mq].values, ta_q[mq].values)
    return out


def ml_run(rid: str, TA: dict, TAQ: dict) -> dict:
    import openpyxl
    ws = openpyxl.load_workbook(mt.find_workbook(rid), data_only=True).active
    hdr = [c.value for c in ws[1]]
    c = {h: i for i, h in enumerate(hdr) if h}
    ai_t, ta_t = [], []
    qa = {q: [] for q in (1, 2, 3)}
    qt = {q: [] for q in (1, 2, 3)}
    for r in ws.iter_rows(min_row=2, values_only=True):
        num = r[c["Number"]]
        a = r[c["AI Total Score"]]
        if a is None:
            continue
        t1, t2 = TA.get(num, (None, None))
        if t1 is None or t2 is None:
            continue
        ai_t.append(float(a) + float(r[c["AI Total Bonus"]] or 0))
        ta_t.append((t1 + t2) / 2)
        tq = TAQ[num][0]
        for q in (1, 2, 3):
            v = r[c[f"AI Q{q} Score"]]
            b = r[c[f"AI Q{q} Bonus"]] if f"AI Q{q} Bonus" in c else 0
            qa[q].append(float(v) + float(b or 0) if isinstance(v, (int, float))
                         else float("nan"))
            qt[q].append(tq[q - 1])
    return {"rid": rid, "ai_total": np.array(ai_t), "ta_total": np.array(ta_t),
            "q": {q: (np.array(qa[q]), np.array(qt[q])) for q in (1, 2, 3)}}


# --------------------------------------------------------------------------
def summarise(run: dict, meta: dict) -> dict:
    tro, tr, tn, tskip = corr(run["ai_total"], run["ta_total"])
    row = {**meta, "n": tn, "tot_rho": tro, "tot_r": tr, "tot_skip": tskip,
           "mae": float(np.mean(np.abs(run["ai_total"] - run["ta_total"]))),
           "ord_rho": ordinal_spearman(run["ai_total"], run["ta_total"])}
    for q in (1, 2, 3):
        a, t = run["q"][q]
        ro, r, n, skip = corr(a, t)
        row[f"q{q}_rho"], row[f"q{q}_r"] = ro, r
        row[f"q{q}_n"], row[f"q{q}_skip"] = n, skip
        row[f"q{q}_mae"] = float(np.nanmean(np.abs(a - t)))
    rhos = [row[f"q{q}_rho"] for q in (1, 2, 3)]
    rs = [row[f"q{q}_r"] for q in (1, 2, 3)]
    if tro is not None and all(v is not None for v in rhos):
        row["min_q_rho"] = min(rhos)
        row["weak_q"] = 1 + int(np.argmin(rhos))
        row["gap_rho"] = tro - min(rhos)
        row["flag"] = row["gap_rho"] >= GAP
    else:
        row["min_q_rho"] = row["gap_rho"] = None
        row["weak_q"] = None
        row["flag"] = False
    if tr is not None and all(v is not None for v in rs):
        row["min_q_r"] = min(rs)
        row["gap_r"] = tr - min(rs)
    else:
        row["min_q_r"] = row["gap_r"] = None
    return row


def f3(v):
    return "   --  " if v is None else f"{v:+7.3f}"


def print_table(title, rows, note):
    print(f"## {title}")
    print(note)
    print()
    hdr = (f"{'run':12s} {'model':22s} {'persona':8s} {'n':>4} | "
           f"{'tot rho':>7} {'tot r':>7} | {'Q1 rho':>7} {'Q2 rho':>7} {'Q3 rho':>7} | "
           f"{'Q1 r':>7} {'Q2 r':>7} {'Q3 r':>7} | {'min q':>7} {'gap':>7} {'weak':>4} {'flag':4} "
           f"{'behaviour':14s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['rid']:12s} {r['model'][:22]:22s} {r['persona']:8s} {r['n']:>4} | "
              f"{f3(r['tot_rho'])} {f3(r['tot_r'])} | "
              f"{f3(r['q1_rho'])} {f3(r['q2_rho'])} {f3(r['q3_rho'])} | "
              f"{f3(r['q1_r'])} {f3(r['q2_r'])} {f3(r['q3_r'])} | "
              f"{f3(r['min_q_rho'])} {f3(r['gap_rho'])} "
              f"{('Q'+str(r['weak_q'])) if r['weak_q'] else '  --':>4} "
              f"{'YES ' if r['flag'] else '    ':4} {str(r['behaviour'])[:14]:14s}")
    print()


# --------------------------------------------------------------------------
def main():
    import openpyxl
    print("# Per-question (item-level) rank correlations, AI vs grader average")
    print(f"# numpy {np.__version__}, scipy {stats.__name__ and __import__('scipy').__version__}, "
          f"pandas {pd.__version__}, openpyxl {openpyxl.__version__}")
    print("# Spearman = scipy.stats.spearmanr (midrank ties); Pearson = scipy.stats.pearsonr")
    print(f"# flag rule: total-level rho minus the weakest per-question rho >= {GAP}")
    print("# students: exactly the rows the paper's total-level metric uses "
          "(AI total present AND grader totals present)")
    print()

    cv = pd.read_csv(CV_CSV)
    ml = pd.read_csv(ML_CSV)

    # ---------------------------------------------------------------- CV set
    pairs_cv = ca.matched_persona_pairs(cv, "strict", "neutral")
    pairs_cv_open = pairs_cv[~pairs_cv["family"].isin(ca.CLOSED_FAMILIES)]
    assert len(pairs_cv_open) == 17, len(pairs_cv_open)
    brittle = ["L-M05", "L-U02", "L-W02", "L-F02", "L-M07", "L-S02", "L-Z02",
               "L-Y02", "L-M01", "L-X02", "L-C01", "L-V02", "L-N02", "L-M03",
               "L-J02", "L-Q02", "L-R02"]          # tab:brittle, open-weights block
    assert set(pairs_cv_open["test_rid"]) == set(brittle), \
        set(pairs_cv_open["test_rid"]) ^ set(brittle)
    print(f"CV matched pairs: {len(pairs_cv_open)} open-weights neutral/strict pairs; "
          f"their strict run_ids are exactly the 17 open-weights rows of tab:brittle "
          f"(sections/appendix.tex)")
    cv_ids = list(CV_HEADLINE)
    for _, p in pairs_cv_open.iterrows():
        for rid in (p["baseline_rid"], p["test_rid"]):
            if rid not in cv_ids:
                cv_ids.append(rid)
    print(f"CV runs analysed: {len(cv_ids)} "
          f"({len(CV_HEADLINE)} headline + 17 neutral + 17 strict)")

    # ---------------------------------------------------------------- ML set
    pairs_ml = mt.default_pairs(ml)
    assert len(pairs_ml) == 17, len(pairs_ml)
    ml_ids = list(ML_HEADLINE)
    for _, n, s in pairs_ml:
        for rid in (n.run_id, s.run_id):
            if rid not in ml_ids:
                ml_ids.append(rid)
    print(f"ML runs analysed: {len(ml_ids)} "
          f"({len(ML_HEADLINE)} headline + 17 neutral + 17 strict from "
          f"introduction_to_ai_make_paper_tables.default_pairs)")
    print()

    TA = ra.ta_truth()
    TAQ = ra.ta_per_question()
    print(f"ML ground truth: {len(TA)} students in "
          f"introduction_to_ai_dataset/Practical_AI_exam_grades.csv")
    print()

    # ---------------------------------------------------------------- compute
    cv_rows, ml_rows = [], []
    checks = []
    for rid in cv_ids:
        m = cv[cv.run_id == rid].iloc[0]
        run = cv_run(rid)
        row = summarise(run, {"rid": rid, "model": str(m["model"]),
                              "persona": str(m["strictness"]),
                              "behaviour": str(m["behaviour"]),
                              "thinking": int(m["use_reasoning"]),
                              "exam": "CV"})
        # cross-checks against the master CSV
        checks.append(("CV", rid, "n", row["n"], float(m["n_valid_vs_TA"])))
        checks.append(("CV", rid, "MAE", row["mae"], float(m["MAE_vs_TA_avg"])))
        checks.append(("CV", rid, "pearson_r", row["tot_r"], m["AI_TA_pearson_r"]))
        for q in (1, 2, 3):
            a, t = run["q_csvmask"][q]
            checks.append(("CV", rid, f"Q{q}_MAE",
                           float(np.mean(np.abs(a - t))), float(m[f"Q{q}_MAE_vs_TA"])))
        cv_rows.append(row)

    for rid in ml_ids:
        m = ml[ml.run_id == rid].iloc[0]
        run = ml_run(rid, TA, TAQ)
        row = summarise(run, {"rid": rid, "model": str(m["model"]),
                              "persona": str(m["strictness"]),
                              "behaviour": str(m["behaviour"]),
                              "thinking": int(m["use_reasoning"]),
                              "exam": "ML"})
        checks.append(("ML", rid, "n", row["n"], float(m["n_valid_vs_TA"])))
        checks.append(("ML", rid, "MAE", row["mae"], float(m["MAE_vs_TA_avg"])))
        checks.append(("ML", rid, "pearson_r", row["tot_r"], m["AI_TA_pearson_r"]))
        for q in (1, 2, 3):
            a, t = run["q"][q]
            checks.append(("ML", rid, f"Q{q}_MAE",
                           float(np.nanmean(np.abs(a - t))), float(m[f"Q{q}_MAE_vs_TA"])))
        ml_rows.append(row)

    # ---------------------------------------------------------------- checks
    print("## Cross-checks against the master CSVs "
          "(analysis/{computer_vision,introduction_to_ai}_master_comparison.csv)")
    bad, nan_r = [], []
    for exam, rid, what, mine, theirs in checks:
        if what == "pearson_r":
            if mine is None or (isinstance(theirs, float) and np.isnan(theirs)):
                nan_r.append((exam, rid, mine, theirs))
                continue
        if what == "n":
            ok = int(mine) == int(theirs)
        else:
            ok = abs(float(mine) - float(theirs)) < TOL
        if not ok:
            bad.append((exam, rid, what, mine, theirs))
    print(f"  {len(checks)} comparisons over {len(cv_ids)} CV + {len(ml_ids)} ML runs: "
          f"n, total MAE, total Pearson r, and Q1/Q2/Q3 MAE per run")
    print(f"  tolerance {TOL} (the CSVs store 4 dp) -- mismatches: {len(bad)}")
    for b in bad:
        print(f"    MISMATCH {b}")
    print(f"  runs where the CSV's AI_TA_pearson_r is blank and this script also "
          f"declines the correlation: {len(nan_r)}")
    for exam, rid, mine, theirs in nan_r:
        print(f"    {exam} {rid}: script r = {mine}, CSV = {theirs}")
    assert not bad, bad
    assert all(m is None for _, _, m, _ in nan_r), nan_r
    print("  ALL CROSS-CHECKS PASS -- the student set and the totals are the paper's.")
    print()

    # ---------------------------------------------------------------- tables
    print_table(
        "CV exam (35-pt base scale) -- Spearman rho and Pearson r, AI vs grader average",
        cv_rows,
        "per question q: AI 'AI Q{q} Score' vs the two-TA mean of 'TA n - Q{q} Score'; "
        "total: 'AI Total Score' vs the mean of the two rebuilt TA base totals. "
        "'min q' is the weakest per-question rho, 'gap' = total rho - min q rho, "
        "'weak' names that question.")

    print_table(
        "ML exam (>=65-pt scale) -- Spearman rho and Pearson r, AI vs grader average",
        ml_rows,
        "per question q: AI 'AI Q{q} Score' + 'AI Q{q} Bonus' vs the two-TA mean of "
        "'TA n - Q{q} Grade'; total: 'AI Total Score' + 'AI Total Bonus' vs the mean of "
        "'TA n - Total Grade'.")

    # ---------------------------------------------------------------- skips
    print("## Skipped correlations (constant vector -- reported as undefined, not zero)")
    any_skip = False
    for rows in (cv_rows, ml_rows):
        for r in rows:
            parts = []
            if r["tot_skip"]:
                parts.append(f"total: {r['tot_skip']}")
            for q in (1, 2, 3):
                if r[f"q{q}_skip"]:
                    parts.append(f"Q{q}: {r[f'q{q}_skip']}")
            if parts:
                any_skip = True
                print(f"  {r['exam']} {r['rid']:12s} {r['model'][:22]:22s} "
                      f"{r['persona']:8s} ({r['behaviour']}) -- " + "; ".join(parts))
    if not any_skip:
        print("  (none)")
    print()

    # ---------------------------------------------------------------- flags
    print(f"## Runs whose total-level rank correlation hides a weak item "
          f"(gap = total rho - weakest question rho >= {GAP})")
    print()
    for label, rows in (("CV", cv_rows), ("ML", ml_rows)):
        fl = sorted([r for r in rows if r["flag"]], key=lambda r: -r["gap_rho"])
        print(f"  {label}: {len(fl)} of {len(rows)} runs flagged")
        for r in fl:
            print(f"    {r['rid']:12s} {r['model'][:22]:22s} {r['persona']:8s} "
                  f"total rho {r['tot_rho']:+.3f}  weakest Q{r['weak_q']} rho "
                  f"{r['min_q_rho']:+.3f}  gap {r['gap_rho']:.3f}  "
                  f"({r['behaviour']}, n={r['n']})")
        print()

    print("  Distribution of the gap (total rho - weakest question rho) over the runs "
          "where all four correlations are defined:")
    for label, rows in (("CV", cv_rows), ("ML", ml_rows)):
        g = np.array([r["gap_rho"] for r in rows if r["gap_rho"] is not None])
        print(f"    {label} (n={len(g)}): min {g.min():+.3f}, median {np.median(g):.3f}, "
              f"mean {g.mean():.3f}, max {g.max():.3f}; "
              f"gap > 0 in {int((g > 0).sum())} of {len(g)}")
    print()

    print("  Same rule applied to Pearson (total r - weakest question r >= "
          f"{GAP}), for reference:")
    for label, rows in (("CV", cv_rows), ("ML", ml_rows)):
        fl = sorted([r for r in rows if r["gap_r"] is not None and r["gap_r"] >= GAP],
                    key=lambda r: -r["gap_r"])
        print(f"    {label}: {len(fl)} of {len(rows)} -- "
              + (", ".join(f"{r['rid']} ({r['gap_r']:.3f})" for r in fl) or "none"))
    print()

    print("  Is the total better ranked than EVERY single question? "
          "(total rho >= max question rho)")
    for label, rows in (("CV", cv_rows), ("ML", ml_rows)):
        sub = [r for r in rows if r["tot_rho"] is not None
               and all(r[f"q{q}_rho"] is not None for q in (1, 2, 3))]
        yes = [r for r in sub
               if r["tot_rho"] >= max(r[f"q{q}_rho"] for q in (1, 2, 3))]
        no = [r for r in sub if r not in yes]
        print(f"    {label}: yes in {len(yes)} of {len(sub)}; no in {len(no)} -- "
              + (", ".join(f"{r['rid']} (total {r['tot_rho']:.3f} < "
                           f"max q {max(r[f'q{q}_rho'] for q in (1,2,3)):.3f})"
                           for r in no) or "none"))
    print()

    print("  Which question is the weakest, counted over the runs with all three defined:")
    for label, rows in (("CV", cv_rows), ("ML", ml_rows)):
        tally = {q: sum(1 for r in rows if r["weak_q"] == q) for q in (1, 2, 3)}
        tot = sum(tally.values())
        print(f"    {label} ({tot} runs): "
              + ", ".join(f"Q{q} weakest in {tally[q]}" for q in (1, 2, 3)))
    print()

    print("  Stricter readout -- a HEADLINE-GRADE total hiding a weak item "
          "(total rho >= 0.80 and some question rho < 0.70):")
    for label, rows in (("CV", cv_rows), ("ML", ml_rows)):
        sub = [r for r in rows
               if r["tot_rho"] is not None and r["min_q_rho"] is not None
               and r["tot_rho"] >= 0.80 and r["min_q_rho"] < 0.70]
        print(f"    {label}: {len(sub)} of {len(rows)} -- "
              + (", ".join(f"{r['rid']} (tot {r['tot_rho']:.3f}, "
                           f"Q{r['weak_q']} {r['min_q_rho']:.3f})" for r in sub)
                 or "none"))
    print()

    # ------------------------------------------------------- headline summary
    print("## Headline runs, in prose form")
    for label, rows, ids in (("CV", cv_rows, CV_HEADLINE), ("ML", ml_rows, ML_HEADLINE)):
        for rid in ids:
            r = next(x for x in rows if x["rid"] == rid)
            qs = ", ".join(f"Q{q} {r[f'q{q}_rho']:+.3f}" for q in (1, 2, 3))
            print(f"  {label} {rid:6s} {r['model'][:20]:20s} {r['persona']:8s} n={r['n']:4d}  "
                  f"total rho {r['tot_rho']:+.3f} (r {r['tot_r']:+.3f});  {qs};  "
                  f"gap {r['gap_rho']:.3f} at Q{r['weak_q']}")
    print()

    # ---------------------------------------------------------------- spread
    print("## Spread of the per-question rho within a run (max - min), all runs")
    for label, rows in (("CV", cv_rows), ("ML", ml_rows)):
        sp = [(r["rid"], max(r[f"q{q}_rho"] for q in (1, 2, 3))
               - min(r[f"q{q}_rho"] for q in (1, 2, 3)))
              for r in rows if all(r[f"q{q}_rho"] is not None for q in (1, 2, 3))]
        v = np.array([s for _, s in sp])
        worst = max(sp, key=lambda t: t[1])
        print(f"  {label} (n={len(sp)} runs): median {np.median(v):.3f}, "
              f"mean {v.mean():.3f}, max {worst[1]:.3f} ({worst[0]})")
    print()

    print("## rho vs r: how much the rank and linear measures differ "
          "(|rho - r|, over every defined correlation)")
    for label, rows in (("CV", cv_rows), ("ML", ml_rows)):
        d = []
        for r in rows:
            for k in ("tot", "q1", "q2", "q3"):
                a, b = r[f"{k}_rho"], r[f"{k}_r"]
                if a is not None and b is not None:
                    d.append(abs(a - b))
        d = np.array(d)
        print(f"  {label}: n={len(d)}, median {np.median(d):.3f}, max {d.max():.3f}")
    print()

    print("## Tie handling: scipy midrank Spearman vs the argsort-of-argsort helper in")
    print("   analysis/introduction_to_ai_run_analysis.py::spearman (total level)")
    for label, rows in (("CV", cv_rows), ("ML", ml_rows)):
        d = [abs(r["tot_rho"] - r["ord_rho"]) for r in rows if r["tot_rho"] is not None]
        print(f"  {label}: max |midrank - ordinal| = {max(d):.4f}, median {np.median(d):.4f} "
              f"-- the two agree closely at total level; midrank is reported.")
    print()

    # ---------------------------------------------------------------- LaTeX
    print("## LaTeX body for the proposed appendix table (headline runs only)")
    print("% ---- tab:itemcorr body ----")
    # tab:headline names the Gemini models "gemini-3-flash-preview" etc.; follow it.
    disp_cv = {"3-flash-preview": "gemini-3-flash-preview",
               "3.1-pro-preview": "gemini-3.1-pro-preview",
               "flash-lite": "gemini-flash-lite"}
    for label, rows, ids in (("CV", cv_rows, CV_HEADLINE), ("ML", ml_rows, ML_HEADLINE)):
        print(f"\\multicolumn{{8}}{{l}}{{\\emph{{{'CV exam' if label=='CV' else 'ML exam'}}}}} \\\\")
        for rid in ids:
            r = next(x for x in rows if x["rid"] == rid)
            name = disp_cv.get(r["model"], mt.DISPLAY.get(r["model"], r["model"]))
            if r["thinking"]:
                name += " + thinking"     # B03 vs A01, as tab:headline spells it
            # tab:brittle prints run ids plain inside tables (e.g. "L-M05", "C01"),
            # so no $..$ wrapping here.
            print(f"{rid} & {name} & \\emph{{{r['persona']}}} & {r['n']} & "
                  f"{r['q1_rho']:.2f} & {r['q2_rho']:.2f} & {r['q3_rho']:.2f} & "
                  f"{r['tot_rho']:.2f} \\\\")
        if label == "CV":
            print("\\midrule")
    print("% ---- end ----")
    print()
    print("## Variant body: same ten rows with Pearson r columns added "
          "(12 columns: lll r rrrr rrrr)")
    print("% ---- tab:itemcorr body, Spearman + Pearson variant ----")
    for label, rows, ids in (("CV", cv_rows, CV_HEADLINE), ("ML", ml_rows, ML_HEADLINE)):
        print(f"\\multicolumn{{12}}{{l}}{{\\emph{{{'CV exam' if label=='CV' else 'ML exam'}}}}} \\\\")
        for rid in ids:
            r = next(x for x in rows if x["rid"] == rid)
            name = disp_cv.get(r["model"], mt.DISPLAY.get(r["model"], r["model"]))
            if r["thinking"]:
                name += " + thinking"
            vals = [r[f"q{q}_rho"] for q in (1, 2, 3)] + [r["tot_rho"]] + \
                   [r[f"q{q}_r"] for q in (1, 2, 3)] + [r["tot_r"]]
            print(f"{rid} & {name} & \\emph{{{r['persona']}}} & {r['n']} & "
                  + " & ".join(f"{v:.2f}" for v in vals) + " \\\\")
        if label == "CV":
            print("\\midrule")
    print("% ---- end ----")
    print()
    print("Pearson counterparts for the same ten rows (for the caption):")
    for label, rows, ids in (("CV", cv_rows, CV_HEADLINE), ("ML", ml_rows, ML_HEADLINE)):
        for rid in ids:
            r = next(x for x in rows if x["rid"] == rid)
            print(f"  {label} {rid:6s} Q1 r {r['q1_r']:+.3f}  Q2 r {r['q2_r']:+.3f}  "
                  f"Q3 r {r['q3_r']:+.3f}  total r {r['tot_r']:+.3f}")


if __name__ == "__main__":
    main()
