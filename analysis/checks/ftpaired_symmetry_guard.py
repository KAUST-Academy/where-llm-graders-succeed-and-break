#!/usr/bin/env python3
"""Table 15 symmetry guard.

The caption of Table 15 (sections/appendix.tex line 437) claims that
|AI - G2| tracks |AI - G1| within ~0.1 in every row. Rearranging
Eq. (1) -- |AI-G2| = 2(d_bar + floor) - |AI-G1| -- gives implied gaps of
0.22-0.56 on the ML rows. This script recomputes, for the same 20 pooled
cells (5 models x 2 recipes x 2 exams, held-out students only), from the
released workbooks:

    mean |AI-G1|, mean |AI-G2|, mean |AI-G_avg|, the held-out floor
    mean |G1-G2|, d_bar with the paper's normal 95% CI (mean +/- 1.96 SE,
    as finetune/pairwise_analysis.py does) and a percentile bootstrap CI
    (2000 resamples, numpy default_rng(0)), the per-cell gap
    |AI-G1| - |AI-G2| with a paired bootstrap CI, and the maximum gap per exam.

It then checks the implied values against the direct numbers,
decomposes the ML gap (three "ungraded slot stored as zero" rows sit in the
held-out set), and inspects whether TA 1 / TA 2 on the ML sheet is a fixed
recording slot (introduction_to_ai_dataset/Practical_AI_exam_grades.csv).

Conventions (from the task brief and finetune/pairwise_analysis.py):
  CV exam : AI = "AI Total Score"; G_k = sum of "TA k - Q1..Q3 Score".
            The workbooks' "TA k - Total Score (out of 35)" cells are Excel
            formulas whose cached values were dropped when the workbook was
            rewritten (openpyxl returns None for every AI-graded row), so the
            totals are rebuilt from the per-question Score columns and
            verified against the cached totals in
            computer_vision_dataset/Practical_AI_exam_grades.xlsx.
  ML exam : AI = "AI Total Score" + "AI Total Bonus"; G_k = "TA k - Total Grade".
Rows are the held-out students (finetune/data/eval_students.json, 114;
finetune/data_intro/eval_students.json, 208); a row is used only if the AI
total is numeric and both grader totals are numeric.
"""
from __future__ import annotations

import collections
import json
import math
import statistics as st
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "finetune" / "results"
SEED, B = 0, 2000

TAGS = [("A07", "7B"), ("A14", "14B"), ("A30M", "30B"), ("GE4B", "Gemma"), ("LL8B", "Llama")]
RECIPES = [("", "marks"), ("bd", "bd")]
EXAMS = [("cv", "CV"), ("intro", "ML")]

# Table 15 as printed (sections/appendix.tex lines 445-466):
# (n, |AI-Gavg|, |AI-G1|, d_bar, lo, hi, verdict)
PAPER = {
    ("7B", "marks", "CV"): (114, 1.84, 2.35, -0.48, -0.90, -0.05, "better"),
    ("14B", "marks", "CV"): (114, 1.89, 2.38, -0.46, -0.89, -0.04, "better"),
    ("30B", "marks", "CV"): (114, 1.75, 2.38, -0.56, -0.98, -0.13, "better"),
    ("Gemma", "marks", "CV"): (114, 1.88, 2.41, -0.45, -0.89, -0.02, "better"),
    ("Llama", "marks", "CV"): (114, 2.01, 2.54, -0.39, -0.85, +0.07, "= grader"),
    ("7B", "marks", "ML"): (208, 3.40, 4.19, -1.11, -1.73, -0.49, "better"),
    ("14B", "marks", "ML"): (208, 3.29, 4.11, -1.13, -1.75, -0.50, "better"),
    ("30B", "marks", "ML"): (208, 3.45, 4.34, -1.04, -1.65, -0.43, "better"),
    ("Gemma", "marks", "ML"): (208, 3.53, 4.45, -0.94, -1.57, -0.30, "better"),
    ("Llama", "marks", "ML"): (208, 3.37, 4.33, -1.01, -1.65, -0.36, "better"),
    ("7B", "bd", "CV"): (114, 1.94, 2.41, -0.45, -0.90, -0.00, "better"),
    ("14B", "bd", "CV"): (114, 1.84, 2.34, -0.51, -0.93, -0.10, "better"),
    ("30B", "bd", "CV"): (114, 1.99, 2.57, -0.37, -0.81, +0.08, "= grader"),
    ("Gemma", "bd", "CV"): (114, 2.22, 2.53, -0.27, -0.71, +0.17, "= grader"),
    ("Llama", "bd", "CV"): (114, 2.04, 2.54, -0.37, -0.80, +0.06, "= grader"),
    ("7B", "bd", "ML"): (208, 4.19, 5.09, -0.37, -1.06, +0.31, "= grader"),
    ("14B", "bd", "ML"): (208, 4.02, 4.87, -0.51, -1.16, +0.14, "= grader"),
    ("30B", "bd", "ML"): (208, 4.07, 5.07, -0.40, -1.08, +0.28, "= grader"),
    ("Gemma", "bd", "ML"): (208, 4.22, 5.09, -0.37, -1.05, +0.31, "= grader"),
    ("Llama", "bd", "ML"): (208, 4.25, 5.05, -0.31, -0.98, +0.37, "= grader"),
}
PAPER_FLOOR = {"CV": 2.83, "ML": 5.19}
# Implied |AI-G1| - |AI-G2| gaps quoted for Table 15.
QUOTED_GAPS = {("7B", "bd", "ML"): 0.54, ("30B", "bd", "ML"): 0.56,
                 ("Gemma", "bd", "ML"): 0.54, ("30B", "marks", "ML"): 0.38}


def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def load_cell(path: Path, exam: str):
    """-> (rows, info). rows = list of dict(number, ai, g1, g2, id1, id2)."""
    ws = openpyxl.load_workbook(path, data_only=True).active
    hdr = [c.value for c in ws[1]]
    col = {n: i for i, n in enumerate(hdr) if n}
    rows, info = [], collections.Counter()
    for r in ws.iter_rows(min_row=2, values_only=True):
        ai = r[col["AI Total Score"]]
        if not num(ai):
            continue
        info["ai_rows"] += 1
        ai = float(ai)
        if exam == "cv":
            for k in (1, 2):
                if r[col[f"TA {k} - Total Score (out of 35)"]] is None:
                    info["ta_total_cell_None"] += 1
            g = {}
            for k in (1, 2):
                v = [r[col[f"TA {k} - Q{q} Score"]] for q in (1, 2, 3)]
                if all(num(x) for x in v):
                    g[k] = float(sum(v))
        else:
            b = r[col["AI Total Bonus"]]
            if num(b):
                ai += float(b)
            else:
                info["ai_bonus_missing"] += 1
            g = {k: float(r[col[f"TA {k} - Total Grade"]]) for k in (1, 2)
                 if num(r[col[f"TA {k} - Total Grade"]])}
        if len(g) < 2:
            info["ta_missing_excluded"] += 1
            continue
        rows.append(dict(number=r[col["Number"]], ai=ai, g1=g[1], g2=g[2],
                         id1=r[col["TA_1_ID"]], id2=r[col["TA_2_ID"]]))
    return rows, info


def normal_ci(x):
    x = list(x)
    m = st.fmean(x)
    se = st.stdev(x) / math.sqrt(len(x))
    return m, m - 1.96 * se, m + 1.96 * se


def boot_ci(x, seed=SEED, b=B):
    """Percentile bootstrap of the mean over students; numpy default_rng(seed)."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(b, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def verdict(lo, hi):
    if lo <= 0 <= hi:
        return "= grader"
    return "better" if hi < 0 else "worse"


def analyse(rows):
    ai = np.array([r["ai"] for r in rows]); g1 = np.array([r["g1"] for r in rows]); g2 = np.array([r["g2"] for r in rows])
    e1, e2, h = np.abs(ai - g1), np.abs(ai - g2), np.abs(g1 - g2)
    eavg = np.abs(ai - (g1 + g2) / 2)
    d = (e1 + e2) / 2 - h
    gap = e1 - e2
    out = dict(n=len(rows), m_avg=eavg.mean(), m1=e1.mean(), m2=e2.mean(), m12=((e1 + e2) / 2).mean(),
               floor=h.mean(), bias_avg=(ai - (g1 + g2) / 2).mean(), bias1=(ai - g1).mean(), bias2=(ai - g2).mean(),
               slot_diff=(g1 - g2).mean())
    out["d"], out["d_lo"], out["d_hi"] = normal_ci(d)
    out["d_blo"], out["d_bhi"] = boot_ci(d)
    out["gap"], out["gap_lo"], out["gap_hi"] = normal_ci(gap)
    out["gap_blo"], out["gap_bhi"] = boot_ci(gap)
    out["identity_resid"] = 2 * (out["d"] + out["floor"]) - out["m1"] - out["m2"]
    out["per_student_gap"] = gap
    return out


def main():
    print("=" * 100)
    print("Table 15 symmetry guard -- |AI-G1| vs |AI-G2| on the 20 pooled held-out cells")
    print("=" * 100)
    print(f"bootstrap: percentile over students, B={B}, numpy default_rng({SEED}) re-seeded per statistic")
    print("normal CI: mean +/- 1.96 * stdev(ddof=1)/sqrt(n)  (the recipe of finetune/pairwise_analysis.py = Table 15)")

    # ---------------------------------------------------------------- ground truth for cross-checks
    ev = {"cv": json.load(open(ROOT / "finetune/data/eval_students.json")),
          "intro": json.load(open(ROOT / "finetune/data_intro/eval_students.json"))}
    print(f"\nheld-out lists: CV {len(ev['cv'])} students (finetune/data/eval_students.json), "
          f"ML {len(ev['intro'])} students (finetune/data_intro/eval_students.json)")

    cvgt = {}
    ws = openpyxl.load_workbook(ROOT / "computer_vision_dataset/Practical_AI_exam_grades.xlsx", data_only=True).active
    hdr = [c.value for c in ws[1]]; col = {n: i for i, n in enumerate(hdr) if n}
    for r in ws.iter_rows(min_row=2, values_only=True):
        cvgt[r[col["Number"]]] = (float(r[col["TA 1 - Total Score (out of 35)"]]),
                                  float(r[col["TA 2 - Total Score (out of 35)"]]), r[col["TA_1_ID"]], r[col["TA_2_ID"]])
    ml = pd.read_csv(ROOT / "introduction_to_ai_dataset/Practical_AI_exam_grades.csv").set_index("Number")
    mlgt = {n: (float(ml.at[n, "TA 1 - Total Grade"]), float(ml.at[n, "TA 2 - Total Grade"]),
                ml.at[n, "TA_1_ID"], ml.at[n, "TA_2_ID"]) for n in ml.index}

    # ---------------------------------------------------------------- per-cell statistics
    cells = {}
    print("\n--- workbook loading / cross-checks against the dataset ground truth ---")
    for exam, ex in EXAMS:
        for rec, rn in RECIPES:
            for tag, mn in TAGS:
                p = RES / f"X-{tag}{rec}-both-on-{exam}.xlsx"
                rows, info = load_cell(p, exam)
                gt = cvgt if exam == "cv" else mlgt
                ids_ok = set(r["number"] for r in rows) == set(ev[exam])
                gt_ok = all(abs(r["g1"] - gt[r["number"]][0]) < 1e-9 and abs(r["g2"] - gt[r["number"]][1]) < 1e-9
                            and r["id1"] == gt[r["number"]][2] and r["id2"] == gt[r["number"]][3] for r in rows)
                cells[(mn, rn, ex)] = analyse(rows)
                cells[(mn, rn, ex)]["rows"] = rows
                print(f"{p.name:34s} AI rows={info['ai_rows']:4d} used={len(rows):4d} "
                      f"TA-total-cells-None={info['ta_total_cell_None']:4d} bonus-missing={info['ai_bonus_missing']:3d} "
                      f"excluded={info['ta_missing_excluded']:2d} rows==held-out:{ids_ok} G1/G2/ids==dataset:{gt_ok}")
    print("(CV: totals are rebuilt from Q1-Q3 Score whether or not a workbook caches 'TA k - Total Score (out of 35)';"
          " the rebuilt totals equal the cached totals of computer_vision_dataset/Practical_AI_exam_grades.xlsx.)")

    print("\n--- Table 15 recomputed, with both single-grader columns (3 decimals) ---")
    print(f"{'cell':18s} {'n':>4} {'|AI-Gavg|':>9} {'|AI-G1|':>8} {'|AI-G2|':>8} {'mean12':>7} {'floor':>6} "
          f"{'d_bar':>7} {'normal95':>17} {'boot95':>17} {'verdict':9} {'gap=G1-G2':>9} {'gap_boot95':>17}")
    for ex in ("CV", "ML"):
        for rn in ("marks", "bd"):
            for _, mn in TAGS:
                c = cells[(mn, rn, ex)]
                print(f"{mn+' '+rn+', '+ex:18s} {c['n']:4d} {c['m_avg']:9.3f} {c['m1']:8.3f} {c['m2']:8.3f} {c['m12']:7.3f} "
                      f"{c['floor']:6.3f} {c['d']:+7.3f} [{c['d_lo']:+.3f},{c['d_hi']:+.3f}] [{c['d_blo']:+.3f},{c['d_bhi']:+.3f}] "
                      f"{verdict(c['d_lo'], c['d_hi']):9s} {c['gap']:+9.3f} [{c['gap_blo']:+.3f},{c['gap_bhi']:+.3f}]")
    print("gap = mean|AI-G1| - mean|AI-G2| = mean over students of (|AI-G1|_i - |AI-G2|_i); its CI is the paired bootstrap.")
    print("held-out floors: CV %.4f (same 114 students in all 10 CV cells), ML %.4f (same 208 in all 10 ML cells)"
          % (cells[("7B", "marks", "CV")]["floor"], cells[("7B", "marks", "ML")]["floor"]))
    print("Eq.(1) identity 2(d_bar+floor) - |AI-G1| - |AI-G2| = 0 holds in every cell to %.1e"
          % max(abs(c["identity_resid"]) for c in cells.values()))

    # ---------------------------------------------------------------- symmetry-guard check
    print("\n--- symmetry guard: |gap| per cell and its maximum per exam ---")
    for ex in ("CV", "ML"):
        gaps = {k: c["gap"] for k, c in cells.items() if k[2] == ex}
        kmax = max(gaps, key=lambda k: abs(gaps[k]))
        over = [f"{k[0]} {k[1]} ({gaps[k]:+.3f})" for k in gaps if abs(gaps[k]) > 0.1]
        sig = [f"{k[0]} {k[1]}" for k in gaps if not (cells[k]["gap_blo"] <= 0 <= cells[k]["gap_bhi"])]
        print(f"{ex}: max |gap| = {abs(gaps[kmax]):.3f} ({kmax[0]} {kmax[1]}, gap {gaps[kmax]:+.3f}); "
              f"min |gap| = {min(abs(g) for g in gaps.values()):.3f}; "
              f"rows with |gap| > 0.1: {len(over)} of 10 -> {over if over else 'none'}")
        print(f"    rows whose paired-bootstrap gap CI excludes 0: {len(sig)} of 10 -> {sig if sig else 'none'}")
        print(f"    all 10 gaps have sign {'+' if all(g > 0 for g in gaps.values()) else ('-' if all(g < 0 for g in gaps.values()) else 'mixed')}"
              f" (positive = AI farther from G1 than from G2)")

    # ---------------------------------------------------------------- quoted implied values
    print("\n--- rearrangement |AI-G2| = 2(d_bar + floor) - |AI-G1| from the printed Table 15 ---")
    print("(implied from the 2-decimal table values with floor 2.83/5.19; rounding of three 2-decimal inputs moves the implied value by up to 0.025)")
    print(f"{'cell':18s} {'tbl|AI-G1|':>10} {'tbl d_bar':>9} {'implied|AI-G2|':>14} {'direct|AI-G2|':>13} {'impl-direct':>11} {'implied gap':>11} {'direct gap':>10} {'quoted':>8}")
    for ex in ("ML", "CV"):
        for rn in ("marks", "bd"):
            for _, mn in TAGS:
                k = (mn, rn, ex); c = cells[k]; n_, mavg, m1, d, lo, hi, v = PAPER[k]
                impl = 2 * (d + PAPER_FLOOR[ex]) - m1
                rv = QUOTED_GAPS.get(k)
                print(f"{mn+' '+rn+', '+ex:18s} {m1:10.2f} {d:+9.2f} {impl:14.3f} {c['m2']:13.3f} {impl-c['m2']:+11.3f} "
                      f"{m1-impl:+11.3f} {c['gap']:+10.3f} {('%.2f' % rv) if rv is not None else '':>8}")
    k = ("7B", "bd", "ML"); c = cells[k]
    print(f"7B bd ML, exact: |AI-G1| {c['m1']:.4f}, d_bar {c['d']:+.4f}, floor {c['floor']:.4f} -> "
          f"2(d_bar+floor)-|AI-G1| = {2*(c['d']+c['floor'])-c['m1']:.4f} = direct |AI-G2| {c['m2']:.4f}; "
          f"the quoted 4.55 from rounded inputs differs by {4.55-c['m2']:+.3f}")
    ml_gaps = sorted(cells[k]["gap"] for k in cells if k[2] == "ML")
    print(f"ML implied-gap range quoted: 0.22-0.56; direct gap range across the 10 ML cells: "
          f"{ml_gaps[0]:.3f}-{ml_gaps[-1]:.3f}")

    # ---------------------------------------------------------------- comparison with the printed table
    print("\n--- printed Table 15 vs recomputation (2-decimal comparison) ---")
    bad = 0
    for k, (n_, mavg, m1, d, lo, hi, v) in PAPER.items():
        c = cells[k]
        diffs = []
        for name, tv, rv in (("|AI-Gavg|", mavg, c["m_avg"]), ("|AI-G1|", m1, c["m1"]), ("d_bar", d, c["d"]),
                             ("lo", lo, c["d_lo"]), ("hi", hi, c["d_hi"])):
            if abs(round(rv, 2) - tv) > 0.0051:
                diffs.append(f"{name}: table {tv:+.2f} vs recomputed {rv:+.3f}")
        if n_ != c["n"]:
            diffs.append(f"n: table {n_} vs {c['n']}")
        if v != verdict(c["d_lo"], c["d_hi"]):
            diffs.append(f"verdict: table {v} vs {verdict(c['d_lo'], c['d_hi'])}")
        if diffs:
            bad += 1
            print(f"  {k[0]} {k[1]}, {k[2]}: " + "; ".join(diffs))
    print(f"cells whose printed values differ from the recomputation at 2 decimals: {bad} of 20"
          + ("" if bad else " (Table 15 reproduces exactly)"))
    print("bootstrap vs normal CI: verdict changes in "
          + str(sum(1 for c in cells.values() if verdict(c["d_lo"], c["d_hi"]) != verdict(c["d_blo"], c["d_bhi"]))) + " of 20 cells")

    # ---------------------------------------------------------------- slot analysis on the ML sheet
    print("\n--- is TA 1 / TA 2 on the ML sheet a fixed primary/secondary slot? (introduction_to_ai_dataset/Practical_AI_exam_grades.csv) ---")
    s1 = collections.Counter(ml.TA_1_ID); s2 = collections.Counter(ml.TA_2_ID)
    ids = sorted(set(s1) | set(s2))
    only1 = [t for t in ids if s1[t] and not s2[t]]; only2 = [t for t in ids if s2[t] and not s1[t]]
    both = [t for t in ids if s1[t] and s2[t]]
    print(f"grader ids: {len(ids)}; slot-1 only: {len(only1)}; slot-2 only: {len(only2)}; in both slots: {len(both)} -> "
          + ", ".join(f"{t} (slot1 {s1[t]}, slot2 {s2[t]})" for t in both))
    pairs = collections.Counter(zip(ml.TA_1_ID, ml.TA_2_ID))
    uo = collections.defaultdict(set)
    for a, b in pairs:
        uo[frozenset((a, b))].add((a, b))
    print(f"ordered pairings: {len(pairs)}; unordered pairings: {len(uo)}; unordered pairings recorded in both slot orders: "
          f"{sum(1 for v in uo.values() if len(v) > 1)}")
    firsts = [t for t in ids if t.startswith('TA_') and '+' not in t]
    firsts.sort(key=lambda t: int(t[3:]))
    alt = all((s1[t] > 0) == (int(t[3:]) % 2 == 1) for t in firsts[:20])
    print(f"TA_1..TA_20: odd pseudonyms all in slot 1 and even all in slot 2: {alt} "
          f"(consistent with pseudonyms numbered in sheet-reading order, slot 1 before slot 2 -- inference)")
    lower = sum(1 for a, b in zip(ml.TA_1_ID, ml.TA_2_ID) if a[3:].split('+')[0].isdigit() and b[3:].split('+')[0].isdigit()
                and int(a[3:].split('+')[0]) < int(b[3:].split('+')[0]))
    print(f"rows where the slot-1 pseudonym number is lower than slot-2's: {lower} of {len(ml)}")
    d_all = ml["TA 1 - Total Grade"] - ml["TA 2 - Total Grade"]
    print(f"whole cohort (n={len(ml)}): mean(G1-G2) = {d_all.mean():+.4f}, mean|G1-G2| = {d_all.abs().mean():.4f}")
    print("signed slot difference per stable pairing (n >= 38, the paper's 18-pair backbone), whole cohort:")
    pos = neg = 0
    stable = [(k, v) for k, v in pairs.items() if v >= 38]
    for (a, b), c_ in sorted(stable, key=lambda kv: -kv[1]):
        m = ml[(ml.TA_1_ID == a) & (ml.TA_2_ID == b)]
        dd = m["TA 1 - Total Grade"] - m["TA 2 - Total Grade"]
        pos += dd.mean() > 0; neg += dd.mean() < 0
        print(f"   {a:>6s}/{b:<6s} n={c_:3d} mean(G1-G2)={dd.mean():+7.3f} mean|G1-G2|={dd.abs().mean():6.3f}")
    print(f"stable pairings with slot 1 more generous: {pos}; with slot 2 more generous: {neg} (of {len(stable)})")
    print("README: introduction_to_ai_dataset/README.md calls TA_1_ID/TA_2_ID pseudonymised graders and refers to them as 'slots'"
          " ('One slot carries the composite id TA_49+TA_50'); computer_vision_dataset/README.md states outright that"
          " 'TA_1_ID and TA_2_ID are slots, not people' and that a pooled signed TA_1 - TA_2 difference is uninterpretable.")
    print("Verdict: TA 1 / TA 2 is a fixed recording slot per pairing (every pairing is always written in the same order,"
          " so within a pairing the columns are two fixed people), but the slot carries no primary/secondary role:"
          " which member is more generous flips from pairing to pairing, and nothing in the data or README marks slot 1"
          " as a first or moderating grader. Neither is it a randomised order.")

    # ---------------------------------------------------------------- decomposition of the ML gap
    print("\n--- why |AI-G1| > |AI-G2| on the ML held-out set: decomposition ---")
    zero_rows = ml[((ml["TA 1 - Total Grade"] == 0) & (ml["TA 2 - Total Grade"] > 0)) |
                   ((ml["TA 2 - Total Grade"] == 0) & (ml["TA 1 - Total Grade"] > 0))]
    zh = sorted(set(zero_rows.index) & set(ev["intro"]))
    print(f"cohort rows with exactly one grader at 0.0 and the partner > 0 ('ungraded slots stored as zeros', Appendix G): {len(zero_rows)}; "
          f"in the held-out 208: {zh}")
    for n_ in zh:
        print(f"   student {n_}: {ml.at[n_, 'TA_1_ID']}/{ml.at[n_, 'TA_2_ID']} G1={ml.at[n_, 'TA 1 - Total Grade']:.2f} G2={ml.at[n_, 'TA 2 - Total Grade']:.2f}")
    print(f"{'cell':18s} {'gap':>7} {'zero-row contrib':>16} {'gap excl.':>9} {'|AI-G1| excl.':>13} {'|AI-G2| excl.':>13} {'floor excl.':>11} {'d_bar excl.':>11} {'n':>4}")
    for rn in ("marks", "bd"):
        for _, mn in TAGS:
            c = cells[(mn, rn, "ML")]
            rows = c["rows"]
            zc = sum(abs(r["ai"] - r["g1"]) - abs(r["ai"] - r["g2"]) for r in rows if r["number"] in zh) / len(rows)
            keep = [r for r in rows if r["number"] not in zh]
            k = analyse(keep)
            print(f"{mn+' '+rn+', ML':18s} {c['gap']:+7.3f} {zc:+16.3f} {k['gap']:+9.3f} {k['m1']:13.3f} {k['m2']:13.3f} {k['floor']:11.3f} {k['d']:+11.3f} {k['n']:4d}")
    print("(zero-row contrib = sum of (|AI-G1|_i - |AI-G2|_i) over the listed students, divided by 208; student 634 alone"
          " contributes (|AI-0| - |AI-51|)/208.)")

    # per-pairing contributions, averaged over the 10 ML cells
    print("\nper-pairing contribution to the ML gap (held-out students, gap_i averaged over the 10 ML cells), pairings with n >= 5:")
    pp = collections.defaultdict(list)
    for rn in ("marks", "bd"):
        for _, mn in TAGS:
            for r in cells[(mn, rn, "ML")]["rows"]:
                pp[(r["id1"], r["id2"])].append(abs(r["ai"] - r["g1"]) - abs(r["ai"] - r["g2"]))
    xs, ys = [], []
    print(f"   {'pairing':16s} {'n_heldout':>9} {'mean gap_i':>10} {'sum/208':>8} {'cohort mean(G1-G2)':>19}")
    for (a, b), v in sorted(pp.items(), key=lambda kv: -abs(sum(kv[1]))):
        n_ = len(v) // 10
        if n_ < 5:
            continue
        m = ml[(ml.TA_1_ID == a) & (ml.TA_2_ID == b)]
        cd = (m["TA 1 - Total Grade"] - m["TA 2 - Total Grade"]).mean()
        xs.append(cd); ys.append(st.fmean(v))
        print(f"   {a+'/'+b:16s} {n_:9d} {st.fmean(v):+10.3f} {sum(v)/10/208:+8.3f} {cd:+19.3f}")
    r = np.corrcoef(xs, ys)[0, 1]
    print(f"Pearson r across these {len(xs)} pairings between the cohort signed slot difference mean(G1-G2) and the held-out mean gap_i: {r:+.3f}"
          " (negative = the AI sits closer to whichever slot holds the more generous grader)")

    # lenient / harsh relabelling (slot-free)
    print("\n--- slot-free relabelling: per pairing, 'lenient' = the slot with the higher cohort mean total, 'harsh' = the other ---")
    len_slot = {}
    for (a, b) in pairs:
        m = ml[(ml.TA_1_ID == a) & (ml.TA_2_ID == b)]
        len_slot[(a, b)] = 1 if (m["TA 1 - Total Grade"] - m["TA 2 - Total Grade"]).mean() >= 0 else 2
    cv_len = {}
    cvdf = pd.DataFrame([dict(id1=v[2], id2=v[3], g1=v[0], g2=v[1]) for v in cvgt.values()])
    for (a, b), m in cvdf.groupby(["id1", "id2"]):
        cv_len[(a, b)] = 1 if (m.g1 - m.g2).mean() >= 0 else 2
    print(f"{'cell':18s} {'|AI-G_len|':>10} {'|AI-G_harsh|':>12} {'len-harsh':>9} {'boot95':>17} {'bias vs Gavg':>12} {'held-out mean(G1-G2)':>20}")
    for ex in ("CV", "ML"):
        for rn in ("marks", "bd"):
            for _, mn in TAGS:
                c = cells[(mn, rn, ex)]
                ls = cv_len if ex == "CV" else len_slot
                el, eh = [], []
                for r in c["rows"]:
                    L = ls[(r["id1"], r["id2"])]
                    gl, gh = (r["g1"], r["g2"]) if L == 1 else (r["g2"], r["g1"])
                    el.append(abs(r["ai"] - gl)); eh.append(abs(r["ai"] - gh))
                el, eh = np.array(el), np.array(eh)
                lo, hi = boot_ci(el - eh)
                print(f"{mn+' '+rn+', '+ex:18s} {el.mean():10.3f} {eh.mean():12.3f} {(el-eh).mean():+9.3f} [{lo:+.3f},{hi:+.3f}] "
                      f"{c['bias_avg']:+12.3f} {c['slot_diff']:+20.3f}")
    print("(bias vs Gavg = mean(AI - G_avg); held-out mean(G1-G2) = signed slot difference on the held-out students.)")

    # ---------------------------------------------------------------- LaTeX-ready rows
    print("\n--- LaTeX-ready Table 15 rows with both single-grader columns (2 decimals, same order as the paper) ---")
    print(r"Cell & $n$ & $|\mathrm{AI}{-}\mathrm{G}_{\mathrm{avg}}|$ & $|\mathrm{AI}{-}\mathrm{G}_1|$ & $|\mathrm{AI}{-}\mathrm{G}_2|$ & $\overline{d}$ [$95\%$ CI] & Verdict \\")
    for rn in ("marks", "bd"):
        for ex in ("CV", "ML"):
            for _, mn in TAGS:
                c = cells[(mn, rn, ex)]
                v = verdict(c["d_lo"], c["d_hi"])
                vt = r"\textbf{better}" if v == "better" else ("$=$ grader" if v == "= grader" else r"\textbf{worse}")
                print(f"{mn} {rn}, {ex} & {c['n']} & {c['m_avg']:.2f} & {c['m1']:.2f} & {c['m2']:.2f} & "
                      f"${c['d']:+.2f}$ $[{c['d_lo']:+.2f}, {c['d_hi']:+.2f}]$ & {vt} \\\\")
    cvmax = max(abs(cells[k]["gap"]) for k in cells if k[2] == "CV")
    mlmax = max(abs(cells[k]["gap"]) for k in cells if k[2] == "ML")
    c634 = []
    for rn in ("marks", "bd"):
        for _, mn in TAGS:
            c = cells[(mn, rn, "ML")]
            i = [r["number"] for r in c["rows"]].index(634)
            c634.append(c["per_student_gap"][i] / c["n"])
    ml_pos = sum(1 for k in cells if k[2] == "ML" and cells[k]["gap"] > 0)
    cv_pos = sum(1 for k in cells if k[2] == "CV" and cells[k]["gap"] > 0)
    print(f"\nstudent 634 alone (G1 recorded 0.0, G2 51.0) contributes {min(c634):.3f}-{max(c634):.3f} to the ML gap across the 10 ML cells;"
          f" |AI-G1| > |AI-G2| in {ml_pos} of 10 ML rows and {cv_pos} of 10 CV rows.")
    print("\nsuggested caption sentence replacing the symmetry guard:")
    print(f"  'The two single-grader columns differ by up to {cvmax:.2f} on the CV exam and {mlmax:.2f} on the ML exam, where "
          f"$|\\mathrm{{AI}}-\\mathrm{{G}}_1|$ is the larger in every row (no paired difference is significant). "
          f"$\\mathrm{{G}}_1$/$\\mathrm{{G}}_2$ are the sheets' recording slots, fixed within a grader pairing but carrying no "
          f"primary/secondary role; the adapters' positive bias places them nearer the more generous member of each pairing, and one "
          f"held-out ML row whose ungraded slot is stored as $0$ contributes {min(c634):.2f}--{max(c634):.2f} of the ML gap.'")

if __name__ == "__main__":
    main()
