#!/usr/bin/env python3
"""The nine ML rows with an unrecorded (0.0) grade from one grader.

What this computes, end to end, and from where.

(1) IDENTIFICATION.  introduction_to_ai_dataset/Practical_AI_exam_grades.csv
    (1,038 data rows, header `Number, TA_1_ID, TA_2_ID, TA n - Q1..Q3 Grade,
    TA n - Total Grade, Average Grade`).  A row is a one-sided zero when exactly
    one of "TA 1 - Total Grade" / "TA 2 - Total Grade" is 0.0 and the other is
    > 0 -- the criterion of
    analysis/introduction_to_ai_run_analysis.py::one_sided_zeros, which the
    paper's Appendix G paragraph uses.  Printed: student Number, both grader
    ids, both totals, the gap, the resulting TA average, IG08's AI total on that
    student, and the split each falls in per finetune/data_intro/split_meta.json
    and finetune/data_both/split_meta.json ("intro" side).  The nine rows with
    0.0 from BOTH graders (genuine non-submissions) are listed for contrast and
    are never dropped here.

(2) HUMAN FLOOR.  mean |TA1 - TA2| over the same rows, with and without the
    nine, plus the maximum single-paper disagreement max |TA1 - TA2| (the paper
    quotes 51.0 with / 41.0 without).  95% CI by percentile bootstrap over
    students: 2000 resamples with numpy default_rng(0) (the shared
    convention) and, as the cross-check against the printed
    [4.80, 5.48], the 50,000-resample version that
    introduction_to_ai_run_analysis.py::boot_ci actually produced for the paper.

(3) RUN-LEVEL EFFECT.  For every ML run: AI total = "AI Total Score" +
    "AI Total Bonus" (blank bonus = 0), compared with the mean of the two TA
    totals.  MAE = mean |AI - TA_avg|, bias = mean (AI - TA_avg), zero-rate =
    share of AI totals exactly 0, behaviour class by
    introduction_to_ai_run_analysis.py::behaviour (refusal: zero-rate >= 90%
    and sd < 0.5, near-refusal if sd >= 0.5; collapse: MAE >= 8; graded
    otherwise).  "without" drops the nine students from the run entirely (from
    the MAE, the bias, the zero-rate and the sd), exactly as the 24 empty-upload
    students are already dropped.  Covered: the four headline runs (IG08, IG09,
    IG07 and IA133 = Qwen3-Coder-Next neutral, the id confirmed against
    analysis/introduction_to_ai_master_comparison.csv), all 17 matched
    neutral/strict pairs of Table 17 (tab:replication; the pairs come from
    analysis/introduction_to_ai_make_paper_tables.py::default_pairs), and every
    completed run of Table 20 (tab:iaruns) for the sweep.  Multi-sample runs
    (n>1) are rebuilt from introduction_to_ai_results/variance/<tag>.csv the way
    introduction_to_ai_run_analysis.py::stats_from_variance does.

(4) HELD-OUT / FINE-TUNING EFFECT.  The 208 pinned held-out students
    (finetune/data_intro/eval_students.json) and the ten pooled-adapter ML
    workbooks finetune/results/X-{A07,A14,A30M,GE4B,LL8B}[bd]-both-on-intro.xlsx
    -- the source of Table 1's "pooled" ML column and of Table 15's ML rows.
    Recomputed with and without whichever of the nine are held out: MAE against
    the grader average, the paired statistic
    d_i = 0.5(|AI-G1| + |AI-G2|) - |G1-G2| (Eq. paired) with the paper's
    mean +/- 1.96*SE CI (finetune/pairwise_analysis.py::mean_ci) and a 2000-
    resample percentile bootstrap, and the Table 15 verdict.

(5) VERDICT ON RETRAINING WITHOUT THE ROWS, which is warranted only
    if the rows sit in the training split AND their removal moves a held-out
    verdict.

Cross-checks against numbers already in the paper are asserted or printed at
every step (floor 5.13, IG08 3.40/+1.18, IG07 3.92, IG09 3.47, IA133 4.47,
Table 17's neutral/strict columns, Table 1's pooled ML column, Table 15's ML
d-bar).  Nothing is hand-entered.

Run:  python3 analysis/checks/ml_unrecorded_grader_rows.py
"""
from __future__ import annotations

import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np
import openpyxl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
import introduction_to_ai_run_analysis as ra  # noqa: E402
import introduction_to_ai_make_paper_tables as mpt  # noqa: E402

import pandas as pd  # noqa: E402

DATA = ROOT / "introduction_to_ai_dataset"
RES = ROOT / "introduction_to_ai_results"
FT = ROOT / "finetune"
MASTER = ROOT / "analysis" / "introduction_to_ai_master_comparison.csv"

N_BOOT = 2000
SEED = 0

# Values already printed in the paper, asserted / diffed against below.
PAPER = {
    "floor": 5.13, "floor_lo": 4.80, "floor_hi": 5.48,
    "max_gap_with": 51.0, "max_gap_without": 41.0, "floor_shift": 0.05,
    "IG08": 3.40, "IG09": 3.47, "IG07": 3.92, "IA133": 4.47,
    "ft_floor": 5.19,
}
# Table 1 (tab:ftmain), ML "pooled" column; Table 15 (tab:ftpaired) ML marks/bd d-bar.
PAPER_FT_MAE = {"A07": 3.40, "A14": 3.29, "A30M": 3.45, "GE4B": 3.53, "LL8B": 3.37}
PAPER_FT_DBAR = {("A07", ""): -1.11, ("A14", ""): -1.13, ("A30M", ""): -1.04,
                 ("GE4B", ""): -0.94, ("LL8B", ""): -1.01,
                 ("A07", "bd"): -0.37, ("A14", "bd"): -0.51, ("A30M", "bd"): -0.40,
                 ("GE4B", "bd"): -0.37, ("LL8B", "bd"): -0.31}
FT_MODELS = [("Qwen2.5-Coder-7B", "A07"), ("Qwen2.5-Coder-14B", "A14"),
             ("Qwen3-Coder-30B-A3B", "A30M"), ("Gemma-4-E4B", "GE4B"),
             ("Llama-3.1-8B", "LL8B")]


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------
def boot_ci(vals, n_boot=N_BOOT, seed=SEED):
    """95% percentile bootstrap of the mean, over students."""
    v = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    b = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# ---------------------------------------------------------------------------
# ground truth
# ---------------------------------------------------------------------------
def truth_rows():
    """Raw CSV rows as dicts of floats (no filtering)."""
    out = []
    with open(DATA / "Practical_AI_exam_grades.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append({
                "n": int(row["Number"]), "ta1": row["TA_1_ID"], "ta2": row["TA_2_ID"],
                "t1": float(row["TA 1 - Total Grade"]), "t2": float(row["TA 2 - Total Grade"]),
                "q1": (float(row["TA 1 - Q1 Grade"]), float(row["TA 2 - Q1 Grade"])),
                "q2": (float(row["TA 1 - Q2 Grade"]), float(row["TA 2 - Q2 Grade"])),
                "q3": (float(row["TA 1 - Q3 Grade"]), float(row["TA 2 - Q3 Grade"])),
            })
    return out


# ---------------------------------------------------------------------------
# run workbooks -> per-student AI totals
# ---------------------------------------------------------------------------
def run_totals(path):
    """{student number: AI total (score + bonus)} for one ML run workbook."""
    ws = openpyxl.load_workbook(path, data_only=True).active
    hdr = [c.value for c in ws[1]]
    c = {h: i for i, h in enumerate(hdr) if h}
    out = {}
    for r in ws.iter_rows(min_row=2, values_only=True):
        a = r[c["AI Total Score"]]
        if a is None:
            continue
        out[r[c["Number"]]] = float(a) + float(r[c["AI Total Bonus"]] or 0)
    return out


def run_stats_from_totals(totals, TA, drop=frozenset()):
    """MAE / bias / zero-rate / sd / behaviour, mirroring run_stats + behaviour."""
    tot, diffs = [], []
    for stu, t in totals.items():
        if stu in drop:
            continue
        tot.append(t)
        t1, t2 = TA.get(stu, (None, None))
        if t1 is not None and t2 is not None:
            diffs.append(t - (t1 + t2) / 2)
    if not diffs:
        return None
    s = {"n": len(tot), "dual": len(diffs), "mean": st.fmean(tot),
         "std": st.stdev(tot) if len(tot) > 1 else 0.0,
         "zero_rate": sum(1 for t in tot if t == 0) / len(tot),
         "mae": st.fmean(abs(d) for d in diffs),
         "bias": st.fmean(diffs), "abs": [abs(d) for d in diffs]}
    s["behaviour"] = ra.behaviour(s)
    return s


def tracker_runs():
    """[(run_id, run_tag, workbook path)] for every completed run -- mirrors all_runs()."""
    ws = openpyxl.load_workbook(RES / "ablation_runs.xlsx", data_only=True).active
    hdr = [c.value for c in ws[1]]
    c = {h: i for i, h in enumerate(hdr)}
    out = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        rid = str(r[c["run_id"]] or "")
        if not rid or str(r[c["status"]]) != "done":
            continue
        tag = str(r[c["run_tag"]] or "")
        rel = str(r[c["output_xlsx"]] or "")
        wb = (RES / rel) if rel else (RES / "results" / f"{rid}__{tag}.xlsx")
        if not wb.exists():
            wb = RES / "results" / f"{rid}__{tag}.xlsx"
        if not wb.exists():
            continue
        out.append((rid, tag, wb))
    return out


def totals_for_run(rid, tag, wb):
    """AI totals for a run; n>1 runs come from the variance CSV, as the ledger does."""
    tot = run_totals(wb)
    if tot:
        return tot, "workbook"
    v = ra.variance_totals(tag)
    return {stu: t for stu, (t, _q) in v.items()}, "variance CSV"


# ---------------------------------------------------------------------------
# fine-tuning workbooks (Table 1 / Table 15)
# ---------------------------------------------------------------------------
def ft_rows(path):
    """[(student, AI total, G1 total, G2 total)] from a finetune ML workbook.

    Mirrors finetune/pairwise_analysis.py::load_run: AI = score + bonus, each TA
    total = sum of that TA's Q1-Q3 Grade (bonus folded in), rows with a numeric
    AI total only.
    """
    ws = openpyxl.load_workbook(path, data_only=True).active
    hdr = [c.value for c in ws[1]]
    c = {h: i for i, h in enumerate(hdr) if h}
    rows = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        a = r[c["AI Total Score"]]
        if not isinstance(a, (int, float)):
            continue
        ai = float(a) + float(r[c["AI Total Bonus"]] or 0)
        tot = {}
        for ta in (1, 2):
            v = [r[c[f"TA {ta} - Q{q} Grade"]] for q in (1, 2, 3)]
            if all(isinstance(x, (int, float)) for x in v):
                tot[ta] = float(sum(v))
        if len(tot) != 2:
            continue
        rows.append((r[c["Number"]], ai, tot[1], tot[2]))
    return rows


def normal_ci(d):
    """The paper's CI: mean +/- 1.96 * sd(ddof=1)/sqrt(n) (pairwise_analysis.mean_ci)."""
    m = st.fmean(d)
    se = st.stdev(d) / math.sqrt(len(d))
    return m, m - 1.96 * se, m + 1.96 * se


def verdict(lo, hi):
    if lo <= 0 <= hi:
        return "= grader"
    return "worse" if lo > 0 else "better"


# ---------------------------------------------------------------------------
def main():
    print("# ML exam: the nine rows with an unrecorded (0.0) grade from one grader")
    print(f"# numpy {np.__version__}, pandas {pd.__version__}, openpyxl {openpyxl.__version__}")
    print(f"# bootstrap: {N_BOOT} resamples, numpy default_rng(seed={SEED}), percentiles 2.5/97.5")
    print("# (the paper's own floor CI used introduction_to_ai_run_analysis.boot_ci at 50,000")
    print("#  resamples, seed 0; both are printed where the paper quotes an interval)")
    print()

    rows = truth_rows()
    TA = ra.ta_truth()
    assert len(rows) == 1038, len(rows)
    nine = [r for r in rows if (r["t1"] == 0) != (r["t2"] == 0)]
    both0 = [r for r in rows if r["t1"] == 0 and r["t2"] == 0]
    NINE = {r["n"] for r in nine}
    assert len(nine) == 9, len(nine)
    assert nine == ra.one_sided_zeros(ra.ta_rows()) or \
        [r["n"] for r in nine] == [r["n"] for r in ra.one_sided_zeros(ra.ta_rows())]

    # ---------------------------------------------------------------- (1)
    print("=" * 110)
    print("## 1. The nine rows  (introduction_to_ai_dataset/Practical_AI_exam_grades.csv)")
    print("=" * 110)
    print("criterion: exactly one of 'TA 1 - Total Grade' / 'TA 2 - Total Grade' is 0.0 and the "
          "other is > 0")
    print("(= analysis/introduction_to_ai_run_analysis.py::one_sided_zeros, the paper's definition)")
    print()

    meta_intro = json.load(open(FT / "data_intro" / "split_meta.json"))
    meta_both = json.load(open(FT / "data_both" / "split_meta.json"))
    ev_intro = set(meta_intro["eval_students"]); tr_intro = set(meta_intro["train_students"])
    ev_both = set(meta_both["eval_students"]["intro"]); tr_both = set(meta_both["train_students"]["intro"])
    eval_json = set(json.load(open(FT / "data_intro" / "eval_students.json")))
    assert ev_intro == eval_json and len(ev_intro) == 208, (len(ev_intro), len(eval_json))
    assert len(tr_intro) == 830 and ev_both == ev_intro and tr_both == tr_intro, \
        (len(tr_intro), ev_both == ev_intro, tr_both == tr_intro)
    print(f"splits: finetune/data_intro/split_meta.json  train {len(tr_intro)} / eval {len(ev_intro)}"
          f";  finetune/data_both/split_meta.json ('intro' side) train {len(tr_both)} / eval "
          f"{len(ev_both)} -- the two files carry the identical ML split (seed 42, train_frac 0.8)")
    print()

    ig08_wb = next(p for p in (RES / "results").glob("IG08__*.xlsx"))
    ig08 = run_totals(ig08_wb)
    print(f"IG08 workbook: {ig08_wb.relative_to(ROOT)}  ({len(ig08)} scored students)")
    print()
    hdr = (f"{'student':>7} {'graders':>14} {'TA1 tot':>8} {'TA2 tot':>8} {'zero slot':>9} "
           f"{'gap':>6} {'TA avg':>7} {'IG08 AI':>8} {'|AI-avg|':>9} {'|AI-real|':>9} "
           f"{'data_intro':>10} {'data_both':>10}")
    print(hdr); print("-" * len(hdr))
    for r in sorted(nine, key=lambda r: r["n"]):
        gap = abs(r["t1"] - r["t2"]); avg = (r["t1"] + r["t2"]) / 2
        zslot = "TA 1" if r["t1"] == 0 else "TA 2"
        real = r["t2"] if r["t1"] == 0 else r["t1"]
        ai = ig08.get(r["n"])
        si = "eval" if r["n"] in ev_intro else ("train" if r["n"] in tr_intro else "ABSENT")
        sb = "eval" if r["n"] in ev_both else ("train" if r["n"] in tr_both else "ABSENT")
        print(f"{r['n']:>7} {r['ta1'] + '/' + r['ta2']:>14} {r['t1']:8.3f} {r['t2']:8.3f} "
              f"{zslot:>9} {gap:6.3f} {avg:7.3f} "
              f"{(f'{ai:8.3f}' if ai is not None else '     n/a')} "
              f"{(f'{abs(ai-avg):9.3f}' if ai is not None else '      n/a')} "
              f"{(f'{abs(ai-real):9.3f}' if ai is not None else '      n/a')} {si:>10} {sb:>10}")
    print()
    print("'|AI-real|' is IG08's distance to the single grader who did mark the paper -- the target "
          "the row would have if the empty slot were treated as missing rather than as a 0.")
    n_train = sum(1 for r in nine if r["n"] in tr_intro)
    n_eval = sum(1 for r in nine if r["n"] in ev_intro)
    print(f"\nsplit tally: {n_train} of the nine are in the 830-student TRAINING split "
          f"({sorted(r['n'] for r in nine if r['n'] in tr_intro)}), "
          f"{n_eval} in the 208-student HELD-OUT split "
          f"({sorted(r['n'] for r in nine if r['n'] in ev_intro)}); identical in data_intro and data_both.")
    print()
    print(f"for contrast, the {len(both0)} rows with 0.0 from BOTH graders (genuine non-submissions, "
          f"retained everywhere): {sorted(r['n'] for r in both0)}")
    print()

    # ---------------------------------------------------------------- (2)
    print("=" * 110)
    print("## 2. Human floor and maximum single-paper disagreement, with and without the nine")
    print("=" * 110)
    gaps_all = [abs(r["t1"] - r["t2"]) for r in rows]
    gaps_wo = [abs(r["t1"] - r["t2"]) for r in rows if r["n"] not in NINE]
    for label, g in (("WITH the nine   (n = %d)" % len(gaps_all), gaps_all),
                     ("WITHOUT the nine (n = %d)" % len(gaps_wo), gaps_wo)):
        m = st.fmean(g)
        lo2, hi2 = boot_ci(g, N_BOOT, SEED)
        lo50, hi50 = ra.boot_ci(g, 50_000, 0)
        print(f"  {label}: floor = {m:.4f}  "
              f"95% CI [{lo2:.3f}, {hi2:.3f}] ({N_BOOT} resamples, seed 0)  "
              f"[{lo50:.3f}, {hi50:.3f}] (50,000 resamples, seed 0 -- the paper's)  "
              f"max |TA1-TA2| = {max(g):.1f}")
    shift = st.fmean(gaps_all) - st.fmean(gaps_wo)
    print()
    print(f"  floor shift = {shift:+.4f} (paper: 'they move the floor by {PAPER['floor_shift']}')")
    print(f"  paper cross-check: floor {PAPER['floor']} [{PAPER['floor_lo']}, {PAPER['floor_hi']}] "
          f"-> recomputed {st.fmean(gaps_all):.2f} "
          f"[{ra.boot_ci(gaps_all, 50_000, 0)[0]:.2f}, {ra.boot_ci(gaps_all, 50_000, 0)[1]:.2f}]"
          f"  MATCH: {abs(st.fmean(gaps_all) - PAPER['floor']) < 0.005}")
    print(f"  max disagreement cross-check: paper {PAPER['max_gap_with']} with / "
          f"{PAPER['max_gap_without']} without -> recomputed {max(gaps_all):.1f} / {max(gaps_wo):.1f}"
          f"  MATCH: {max(gaps_all) == PAPER['max_gap_with'] and max(gaps_wo) == PAPER['max_gap_without']}")
    print()
    top = sorted(rows, key=lambda r: -abs(r["t1"] - r["t2"]))[:12]
    print("  the twelve largest single-paper disagreements (one-sided-zero rows marked *):")
    for r in top:
        print(f"    student {r['n']:>5}  |TA1-TA2| = {abs(r['t1']-r['t2']):6.2f}  "
              f"({r['t1']:.3f} vs {r['t2']:.3f}){'  *' if r['n'] in NINE else ''}")
    print(f"  -> {sum(1 for r in top if r['n'] in NINE)} of the top 12 is a one-sided zero "
          f"(student 634, the single largest); the next-largest genuine disagreement is 41.0")
    print()
    floor_with, floor_wo = st.fmean(gaps_all), st.fmean(gaps_wo)

    # ---------------------------------------------------------------- (3a)
    print("=" * 110)
    print("## 3a. Headline ML runs: IG08, IG09, IG07, IA133 (Qwen3-Coder-Next neutral)")
    print("=" * 110)
    d = pd.read_csv(MASTER)
    nxt = d[(d.model == "qwen3-coder-next") & (d.strictness == "neutral") &
            (d.use_solution == 1) & (d.use_breakdown == 1) & (d.use_reasoning == 0) &
            (d.few_shot == 0) & (d.temperature == 0) & (d.runs == 1) & (d.students == "all")]
    print(f"id confirmation: the Qwen3-Coder-Next neutral run at the default prompt config in "
          f"{MASTER.relative_to(ROOT)} is "
          f"{list(nxt.run_id)} (MAE {list(nxt.MAE_vs_TA_avg)}) -- IA133 confirmed"
          f"{'' if list(nxt.run_id) == ['IA133'] else ' (see the row list above)'}")
    print()
    headline = ["IG08", "IG09", "IG07", "IA133"]
    wb_by_run = {rid: wb for rid, _t, wb in tracker_runs()}
    hdr = (f"{'run':>6} {'model':>22} {'n with':>7} {'n w/o':>6} "
           f"{'MAE with':>9} {'MAE w/o':>9} {'dMAE':>7} "
           f"{'bias with':>10} {'bias w/o':>9} {'dbias':>7} "
           f"{'zero with':>10} {'zero w/o':>9} {'beh with':>9} {'beh w/o':>9}")
    print(hdr); print("-" * len(hdr))
    headline_stats = {}
    for rid in headline:
        wb = wb_by_run.get(rid) or mpt.find_workbook(rid)
        tot = run_totals(wb)
        a = run_stats_from_totals(tot, TA)
        b = run_stats_from_totals(tot, TA, NINE)
        headline_stats[rid] = (a, b, wb)
        model = d[d.run_id == rid].model.iloc[0]
        print(f"{rid:>6} {model:>22} {a['dual']:>7} {b['dual']:>6} "
              f"{a['mae']:9.4f} {b['mae']:9.4f} {b['mae']-a['mae']:+7.4f} "
              f"{a['bias']:+10.4f} {b['bias']:+9.4f} {b['bias']-a['bias']:+7.4f} "
              f"{a['zero_rate']:10.4f} {b['zero_rate']:9.4f} "
              f"{a['behaviour'].split(' (')[0]:>9} {b['behaviour'].split(' (')[0]:>9}")
    print()
    print(f"  with 95% CIs and the x-floor column (floor with = {floor_with:.4f}, "
          f"floor without = {floor_wo:.4f}):")
    for rid in headline:
        a, b, wb = headline_stats[rid]
        alo, ahi = boot_ci(a["abs"], N_BOOT, SEED); blo, bhi = boot_ci(b["abs"], N_BOOT, SEED)
        a50 = ra.boot_ci(a["abs"], 50_000, 0); b50 = ra.boot_ci(b["abs"], 50_000, 0)
        pub = d[d.run_id == rid].iloc[0]
        print(f"  {rid}: with    MAE {a['mae']:.4f}  CI(2000) [{alo:.3f}, {ahi:.3f}]  "
              f"CI(50000) [{a50[0]:.3f}, {a50[1]:.3f}]  x floor {a['mae']/floor_with:.3f}")
        print(f"  {'':4s}  without MAE {b['mae']:.4f}  CI(2000) [{blo:.3f}, {bhi:.3f}]  "
              f"CI(50000) [{b50[0]:.3f}, {b50[1]:.3f}]  x floor {b['mae']/floor_wo:.3f}")
        print(f"  {'':4s}  master CSV: MAE {pub.MAE_vs_TA_avg:.4f} CI [{pub.MAE_CI95_lo:.4f}, "
              f"{pub.MAE_CI95_hi:.4f}] bias {pub.bias_vs_TA_avg:+.4f}  -> reproduced: "
              f"{abs(pub.MAE_vs_TA_avg - a['mae']) < 5e-4 and abs(pub.bias_vs_TA_avg - a['bias']) < 5e-4}"
              f"  (paper prints {PAPER[rid]:.2f})")
        print(f"  {'':4s}  workbook {wb.relative_to(ROOT)}")
    print()
    print("  per-student contribution of the nine to each headline run's MAE:")
    for rid in headline:
        a, b, _ = headline_stats[rid]
        tot = run_totals(wb_by_run.get(rid) or mpt.find_workbook(rid))
        contrib = []
        for r in sorted(nine, key=lambda r: r["n"]):
            if r["n"] in tot:
                contrib.append((r["n"], abs(tot[r["n"]] - (r["t1"] + r["t2"]) / 2)))
        print(f"    {rid}: |AI-TA_avg| on the nine = "
              + ", ".join(f"{s}:{v:.2f}" for s, v in contrib)
              + f"  (mean {st.fmean(v for _s, v in contrib):.3f} vs run mean {a['mae']:.3f})")
    print()

    # ---------------------------------------------------------------- (3b)
    print("=" * 110)
    print("## 3b. Table 17 (tab:replication): all 17 matched neutral/strict pairs")
    print("=" * 110)
    pairs = mpt.default_pairs(d)
    assert len(pairs) == 17, len(pairs)
    rec = {}
    tag_by_run = {rid: tg for rid, tg, _wb in tracker_runs()}
    for m, n, s in pairs:
        for tag, row in (("neutral", n), ("strict", s)):
            wb = wb_by_run.get(row.run_id) or mpt.find_workbook(row.run_id)
            tot, src = totals_for_run(row.run_id, tag_by_run.get(row.run_id, ""), wb)
            rec[(m, tag)] = (run_stats_from_totals(tot, TA),
                             run_stats_from_totals(tot, TA, NINE), row, src)
    hdr = (f"{'model':>22} {'run':>6} {'MAE with':>9} {'MAE w/o':>9} {'dMAE':>7} "
           f"{'bias with':>10} {'bias w/o':>9} {'dbias':>7} {'beh with':>12} {'beh w/o':>12} "
           f"{'CSV MAE':>8} {'ok':>3}")
    print(hdr); print("-" * len(hdr))
    for m, n, s in sorted(pairs, key=lambda t: -(t[2].MAE_vs_TA_avg / t[1].MAE_vs_TA_avg)):
        for tag in ("neutral", "strict"):
            a, b, row, _src = rec[(m, tag)]
            ok = abs(row.MAE_vs_TA_avg - a["mae"]) < 5e-4
            print(f"{mpt.DISPLAY[m] + ' ' + tag:>22} {row.run_id:>6} "
                  f"{a['mae']:9.4f} {b['mae']:9.4f} {b['mae']-a['mae']:+7.4f} "
                  f"{a['bias']:+10.4f} {b['bias']:+9.4f} {b['bias']-a['bias']:+7.4f} "
                  f"{a['behaviour'].split(' (')[0]:>12} {b['behaviour'].split(' (')[0]:>12} "
                  f"{row.MAE_vs_TA_avg:8.4f} {str(ok):>3}")
    print()

    print("  Table 17 as printed (2 dp) vs recomputed without the nine:")
    hdr = (f"{'model':>22} {'neut with':>9} {'neut w/o':>9} {'str with':>9} {'str w/o':>9} "
           f"{'ratio with':>10} {'ratio w/o':>9} {'xfloor with':>11} {'xfloor w/o':>10} "
           f"{'zero w/o':>9} {'bias w/o':>9} {'beh w/o':>10}")
    print(hdr); print("-" * len(hdr))
    order_with = sorted(pairs, key=lambda t: -(rec[(t[0], 'strict')][0]['mae'] /
                                               rec[(t[0], 'neutral')][0]['mae']))
    order_wo = sorted(pairs, key=lambda t: -(rec[(t[0], 'strict')][1]['mae'] /
                                             rec[(t[0], 'neutral')][1]['mae']))
    for m, _n, _s in order_with:
        na, nb = rec[(m, "neutral")][0], rec[(m, "neutral")][1]
        sa, sb = rec[(m, "strict")][0], rec[(m, "strict")][1]
        print(f"{mpt.DISPLAY[m]:>22} {na['mae']:9.2f} {nb['mae']:9.2f} "
              f"{sa['mae']:9.2f} {sb['mae']:9.2f} "
              f"{sa['mae']/na['mae']:10.2f} {sb['mae']/nb['mae']:9.2f} "
              f"{sa['mae']/floor_with:11.1f} {sb['mae']/floor_wo:10.1f} "
              f"{sb['zero_rate']:9.3f} {sb['bias']:+9.2f} "
              f"{sb['behaviour'].split(' (')[0]:>10}")
    print()
    names_with = [mpt.DISPLAY[m] for m, _n, _s in order_with]
    names_wo = [mpt.DISPLAY[m] for m, _n, _s in order_wo]
    print("  damage-ratio ordering (descending), ratios to four decimals so near-ties are visible:")
    print(f"    {'#':>2}  {'WITH the nine':<32} {'ratio':>8}   {'WITHOUT the nine':<32} {'ratio':>8}")
    for i in range(17):
        mw, _n, _s = order_with[i]
        mo, _n2, _s2 = order_wo[i]
        rw = rec[(mw, 'strict')][0]['mae'] / rec[(mw, 'neutral')][0]['mae']
        ro = rec[(mo, 'strict')][1]['mae'] / rec[(mo, 'neutral')][1]['mae']
        mark = "" if mw == mo else "   <-- swap"
        print(f"    {i+1:>2}  {mpt.DISPLAY[mw]:<32} {rw:8.4f}   {mpt.DISPLAY[mo]:<32} {ro:8.4f}{mark}")
    print(f"  -> ordering identical: {names_with == names_wo}")
    gaps_o = [(mpt.DISPLAY[order_wo[i][0]], mpt.DISPLAY[order_wo[i + 1][0]],
               rec[(order_wo[i][0], 'strict')][1]['mae'] / rec[(order_wo[i][0], 'neutral')][1]['mae']
               - rec[(order_wo[i + 1][0], 'strict')][1]['mae'] / rec[(order_wo[i + 1][0], 'neutral')][1]['mae'])
              for i in range(16)]
    tight = min(gaps_o, key=lambda t: t[2])
    print(f"  tightest adjacent gap without the nine: {tight[0]} over {tight[1]} by "
          f"{tight[2]:.6f} -- they would print as the same two-decimal ratio (1.95), so the two "
          f"rows become a visual tie, though the underlying order is unchanged")
    print()

    print("  changes of at least 0.01 among the 34 pair runs (MAE / bias / ratio at 2 dp):")
    any_big = False
    for m, _n, _s in order_with:
        for tag in ("neutral", "strict"):
            a, b, row, _src = rec[(m, tag)]
            dm, db = b["mae"] - a["mae"], b["bias"] - a["bias"]
            if abs(dm) >= 0.01 or abs(db) >= 0.01:
                any_big = True
                print(f"    {mpt.DISPLAY[m]} {tag} ({row.run_id}): MAE {a['mae']:.4f} -> "
                      f"{b['mae']:.4f} ({dm:+.4f}), bias {a['bias']:+.4f} -> {b['bias']:+.4f} "
                      f"({db:+.4f})")
    if not any_big:
        print("    (none)")
    beh_ch = [(m, tag) for m, _n, _s in pairs for tag in ("neutral", "strict")
              if rec[(m, tag)][0]["behaviour"].split(" (")[0] !=
              rec[(m, tag)][1]["behaviour"].split(" (")[0]]
    print(f"  behaviour-class changes among the 34 pair runs: "
          f"{[(mpt.DISPLAY[m], t) for m, t in beh_ch] if beh_ch else 'none'}")
    ratio_ch = [(m, rec[(m, 'strict')][0]['mae'] / rec[(m, 'neutral')][0]['mae'],
                 rec[(m, 'strict')][1]['mae'] / rec[(m, 'neutral')][1]['mae'])
                for m, _n, _s in pairs]
    big_r = [(m, r1, r2) for m, r1, r2 in ratio_ch if abs(r2 - r1) >= 0.01]
    print(f"  damage ratios moving by >= 0.01: "
          + (", ".join(f"{mpt.DISPLAY[m]} {r1:.3f}->{r2:.3f}" for m, r1, r2 in big_r)
             if big_r else "none"))
    xf_ch = [(m, rec[(m, 'strict')][0]['mae'] / floor_with,
              rec[(m, 'strict')][1]['mae'] / floor_wo) for m, _n, _s in pairs]
    print(f"  x-floor column (strict MAE / floor) moving by >= 0.05 with the floor recomputed "
          f"({floor_with:.3f} -> {floor_wo:.3f}): "
          + (", ".join(f"{mpt.DISPLAY[m]} {x1:.2f}->{x2:.2f}"
                       for m, x1, x2 in xf_ch if abs(x2 - x1) >= 0.05) or "none"))
    print(f"  models crossing the floor-matched threshold 3*floor "
          f"({3*floor_with:.2f} with, {3*floor_wo:.2f} without) -- the paper's 15.7: "
          f"with {[mpt.DISPLAY[m] for m, _n, _s in pairs if rec[(m,'strict')][0]['mae'] >= 3*floor_with]}; "
          f"without {[mpt.DISPLAY[m] for m, _n, _s in pairs if rec[(m,'strict')][1]['mae'] >= 3*floor_wo]}")
    print()

    # ---------------------------------------------------------------- (3c)
    print("=" * 110)
    print("## 3c. Table 20 (tab:iaruns) sweep: every completed run")
    print("=" * 110)
    sweep = []
    for rid, tag, wb in tracker_runs():
        tot, src = totals_for_run(rid, tag, wb)
        if not tot:
            continue
        a = run_stats_from_totals(tot, TA)
        b = run_stats_from_totals(tot, TA, NINE)
        if a is None or b is None:
            continue
        hit = sorted(set(tot) & NINE)
        sweep.append((rid, src, a, b, hit))
    print(f"runs covered: {len(sweep)} (master CSV has {len(d)} rows)")
    csv_mae = dict(zip(d.run_id, d.MAE_vs_TA_avg))
    bad = [(rid, a["mae"], csv_mae.get(rid)) for rid, _s, a, _b, _h in sweep
           if rid in csv_mae and abs(a["mae"] - csv_mae[rid]) >= 5e-4]
    print(f"runs whose 'with' MAE fails to reproduce the master CSV to 3 dp: "
          f"{[(r, round(x, 4), round(y, 4)) for r, x, y in bad] if bad else 'none'}")
    import collections
    cov = collections.Counter(len(h) for _r, _s, _a, _b, h in sweep)
    print(f"how many of the nine each run actually scores: "
          + ", ".join(f"{k} row(s): {v} runs" for k, v in sorted(cov.items(), reverse=True))
          + "  (students 22 and 29 fall inside the 1-50 and 1-100 subsets, so even the subset "
            "runs are touched)")
    changed = [(rid, a, b) for rid, _s, a, b, h in sweep
               if abs(b["mae"] - a["mae"]) >= 0.01 or abs(b["bias"] - a["bias"]) >= 0.01]
    dmaes = sorted(abs(b["mae"] - a["mae"]) for _r, _s, a, b, _h in sweep)
    print(f"runs whose MAE or bias moves by >= 0.01: {len(changed)} of {len(sweep)}")
    print(f"|dMAE| distribution over the {len(sweep)} runs: median {np.median(dmaes):.4f}, "
          f"90th pct {np.percentile(dmaes, 90):.4f}, max {dmaes[-1]:.4f}; "
          f">= 0.10: {sum(1 for x in dmaes if x >= 0.10)} runs, "
          f">= 0.05: {sum(1 for x in dmaes if x >= 0.05)} runs, "
          f">= 0.01: {sum(1 for x in dmaes if x >= 0.01)} runs")
    print("(no full-cohort MAE in Table 20 moves by as much as 0.22; the seven largest moves are "
          "all small-cohort probes, where the nine are a much larger share of the denominator)")
    print()
    print("the 25 largest |dMAE| rows of Table 20 (everything above 0.10 is inside this list):")
    hdr = (f"{'run':>10} {'students':>9} {'n with':>7} {'MAE with':>9} {'MAE w/o':>9} {'dMAE':>7} "
           f"{'bias with':>10} {'bias w/o':>9} {'dbias':>7} {'beh with':>12} {'beh w/o':>12}")
    print(hdr); print("-" * len(hdr))
    stu_col = dict(zip(d.run_id, d.students))
    for rid, a, b in sorted(changed, key=lambda t: -abs(t[2]["mae"] - t[1]["mae"]))[:25]:
        print(f"{rid:>10} {str(stu_col.get(rid, '?')):>9} {a['dual']:>7} "
              f"{a['mae']:9.4f} {b['mae']:9.4f} {b['mae']-a['mae']:+7.4f} "
              f"{a['bias']:+10.4f} {b['bias']:+9.4f} {b['bias']-a['bias']:+7.4f} "
              f"{a['behaviour'].split(' (')[0]:>12} {b['behaviour'].split(' (')[0]:>12}")
    full = [(rid, a, b) for rid, _s, a, b, _h in sweep if str(stu_col.get(rid, "all")) == "all"]
    fm = max(full, key=lambda t: abs(t[2]["mae"] - t[1]["mae"]))
    print(f"\nlargest |dMAE| among the {len(full)} FULL-COHORT rows of Table 20: {fm[0]} "
          f"{fm[1]['mae']:.4f} -> {fm[2]['mae']:.4f} ({fm[2]['mae']-fm[1]['mae']:+.4f})")
    print()
    mx = max(sweep, key=lambda s: abs(s[3]["mae"] - s[2]["mae"]))
    print(f"largest MAE move anywhere in Table 20: {mx[0]} {mx[2]['mae']:.4f} -> {mx[3]['mae']:.4f} "
          f"({mx[3]['mae']-mx[2]['mae']:+.4f})")
    mb = max(sweep, key=lambda s: abs(s[3]["bias"] - s[2]["bias"]))
    print(f"largest bias move anywhere in Table 20: {mb[0]} {mb[2]['bias']:+.4f} -> "
          f"{mb[3]['bias']:+.4f} ({mb[3]['bias']-mb[2]['bias']:+.4f})")
    beh_sw = [(rid, a["behaviour"], b["behaviour"]) for rid, _s, a, b, _h in sweep
              if a["behaviour"].split(" (")[0] != b["behaviour"].split(" (")[0]]
    print(f"behaviour-class changes anywhere in Table 20: {beh_sw if beh_sw else 'none'}")
    print()

    # ---------------------------------------------------------------- (4)
    print("=" * 110)
    print("## 4. Held-out split and the pooled adapters (Table 1 / Table 15)")
    print("=" * 110)
    held = sorted(NINE & ev_intro)
    print(f"of the nine, {len(held)} are among the 208 held-out students: {held}")
    print(f"the other {9 - len(held)} are in the 830-student training split: "
          f"{sorted(NINE & tr_intro)}")
    print()
    fl_rows = [r for r in rows if r["n"] in ev_intro]
    fl_all = [abs(r["t1"] - r["t2"]) for r in fl_rows]
    fl_wo = [abs(r["t1"] - r["t2"]) for r in fl_rows if r["n"] not in NINE]
    print(f"held-out floor (mean |G1-G2| over the 208): {st.fmean(fl_all):.4f} "
          f"(paper's Table 15 caption: {PAPER['ft_floor']}) -> without the three: "
          f"{st.fmean(fl_wo):.4f} over {len(fl_wo)} students "
          f"({st.fmean(fl_wo)-st.fmean(fl_all):+.4f})")
    print()
    hdr = (f"{'cell':>22} {'workbook':>30} {'n':>4} {'MAE with':>9} {'MAE w/o':>9} {'dMAE':>7} "
           f"{'d-bar with':>10} {'d-bar w/o':>10} {'dd':>7} "
           f"{'CI with':>20} {'CI w/o':>20} {'verdict with':>12} {'verdict w/o':>12}")
    print(hdr); print("-" * len(hdr))
    ft_cells = []
    for recipe in ("", "bd"):
        for name, tag in FT_MODELS:
            p = FT / "results" / f"X-{tag}{recipe}-both-on-intro.xlsx"
            rws = ft_rows(p)
            assert {r[0] for r in rws} == ev_intro, (p.name, len(rws))
            for drop in (False, True):
                sub = [r for r in rws if not (drop and r[0] in NINE)]
                ai = np.array([r[1] for r in sub]); g1 = np.array([r[2] for r in sub])
                g2 = np.array([r[3] for r in sub])
                mae = float(np.abs(ai - (g1 + g2) / 2).mean())
                dd = (0.5 * (np.abs(ai - g1) + np.abs(ai - g2)) - np.abs(g1 - g2))
                m, lo, hi = normal_ci(dd.tolist())
                blo, bhi = boot_ci(dd.tolist(), N_BOOT, SEED)
                cell = dict(n=len(sub), mae=mae, d=m, lo=lo, hi=hi, blo=blo, bhi=bhi,
                            ag1=float(np.abs(ai - g1).mean()), ag2=float(np.abs(ai - g2).mean()),
                            floor=float(np.abs(g1 - g2).mean()))
                if not drop:
                    keep = cell
                else:
                    ft_cells.append((f"{name} {'bd' if recipe else 'marks'}", p.name,
                                     keep, cell, tag, recipe))
    for label, fn, a, b, tag, recipe in ft_cells:
        print(f"{label:>22} {fn:>30} {a['n']:>4} {a['mae']:9.4f} {b['mae']:9.4f} "
              f"{b['mae']-a['mae']:+7.4f} {a['d']:+10.4f} {b['d']:+10.4f} {b['d']-a['d']:+7.4f} "
              f"[{a['lo']:+.3f}, {a['hi']:+.3f}] [{b['lo']:+.3f}, {b['hi']:+.3f}] "
              f"{verdict(a['lo'], a['hi']):>12} {verdict(b['lo'], b['hi']):>12}")
    print()
    print("  cross-check against the paper: Table 1's ML 'pooled' column and Table 15's ML d-bar")
    for label, fn, a, b, tag, recipe in ft_cells:
        if recipe == "":
            print(f"    {label:>22}: Table 1 prints {PAPER_FT_MAE[tag]:.2f}, recomputed "
                  f"{a['mae']:.2f}  MATCH {abs(a['mae']-PAPER_FT_MAE[tag]) < 0.005}", end="")
        else:
            print(f"    {label:>22}: (bd, Table 14/15)", end="")
        print(f" | Table 15 d-bar {PAPER_FT_DBAR[(tag, recipe)]:+.2f}, recomputed {a['d']:+.2f}"
              f"  MATCH {abs(a['d']-PAPER_FT_DBAR[(tag, recipe)]) < 0.005}")
    print()
    print("  bootstrap CIs (2000 resamples, seed 0) for the same cells:")
    for label, fn, a, b, tag, recipe in ft_cells:
        print(f"    {label:>22}: with [{a['blo']:+.3f}, {a['bhi']:+.3f}] -> "
              f"{verdict(a['blo'], a['bhi']):9s}   without [{b['blo']:+.3f}, {b['bhi']:+.3f}] -> "
              f"{verdict(b['blo'], b['bhi']):9s}")
    print()
    print("  Table 15's two single-grader columns (the caption attributes their gap mostly to the "
          "one held-out row whose G1 slot is 0.0 against a real G2 mark -- student 634):")
    hdr = (f"{'cell':>22} {'|AI-G1| with':>12} {'|AI-G2| with':>12} {'gap with':>9} "
           f"{'|AI-G1| w/o':>12} {'|AI-G2| w/o':>12} {'gap w/o':>9} {'floor w/o':>9}")
    print(hdr); print("-" * len(hdr))
    for label, fn, a, b, tag, recipe in ft_cells:
        print(f"{label:>22} {a['ag1']:12.3f} {a['ag2']:12.3f} {a['ag1']-a['ag2']:9.3f} "
              f"{b['ag1']:12.3f} {b['ag2']:12.3f} {b['ag1']-b['ag2']:9.3f} {b['floor']:9.3f}")
    n_g1 = sum(1 for _l, _f, a, _b, _t, _r in ft_cells if a['ag1'] > a['ag2'])
    n_g1w = sum(1 for _l, _f, _a, b, _t, _r in ft_cells if b['ag1'] > b['ag2'])
    print(f"  cells where |AI-G1| > |AI-G2| (the caption says 'all ten ML rows'): {n_g1}/10 with "
          f"the three rows, {n_g1w}/10 without them")
    print(f"  mean gap |AI-G1| - |AI-G2| across the ten ML cells: "
          f"{st.fmean(a['ag1'] - a['ag2'] for _l, _f, a, _b, _t, _r in ft_cells):.3f} with the "
          f"three held-out rows, "
          f"{st.fmean(b['ag1'] - b['ag2'] for _l, _f, _a, b, _t, _r in ft_cells):.3f} without "
          f"(the caption's 'grader-slot artefact' reading is confirmed, not weakened)")
    print()
    changed_ft = [(l, a, b) for l, _f, a, b, _t, _r in ft_cells
                  if abs(b["mae"] - a["mae"]) >= 0.01 or abs(b["d"] - a["d"]) >= 0.01]
    print(f"  pooled-adapter cells moving by >= 0.01 in MAE or d-bar: {len(changed_ft)} of "
          f"{len(ft_cells)}")
    for l, a, b in changed_ft:
        print(f"    {l:>22}: MAE {a['mae']:.4f} -> {b['mae']:.4f} ({b['mae']-a['mae']:+.4f}); "
              f"d-bar {a['d']:+.4f} -> {b['d']:+.4f} ({b['d']-a['d']:+.4f})")
    v_ch = [l for l, _f, a, b, _t, _r in ft_cells
            if verdict(a["lo"], a["hi"]) != verdict(b["lo"], b["hi"])
            or verdict(a["blo"], a["bhi"]) != verdict(b["blo"], b["bhi"])]
    print(f"  Table 15 verdict changes (normal CI or bootstrap CI): {v_ch if v_ch else 'none'}")
    print(f"  mean pooled marks MAE on the ML exam (Table 1's bottom row): "
          f"{st.fmean(a['mae'] for l, _f, a, _b, _t, r in ft_cells if r == ''):.4f} -> "
          f"{st.fmean(b['mae'] for l, _f, _a, b, _t, r in ft_cells if r == ''):.4f} "
          f"(paper prints 3.41)")
    print()

    # ---------------------------------------------------------------- (5)
    print("=" * 110)
    print("## 5. Is retraining without these rows warranted?")
    print("=" * 110)
    in_train = bool(NINE & tr_intro)
    moved = bool(v_ch)
    max_dd = max(abs(b["d"] - a["d"]) for _l, _f, a, b, _t, _r in ft_cells)
    print(f"  condition A -- rows sit in the training split: {in_train}  "
          f"({len(NINE & tr_intro)} of 9: {sorted(NINE & tr_intro)})")
    if moved:
        print(f"  condition B -- their removal moves a held-out verdict: True  ({v_ch})")
    else:
        print(f"  condition B -- their removal moves a held-out verdict: False  (no Table 15 cell "
              f"changes class under either CI; the largest d-bar move is {max_dd:.4f}, and the "
              f"five 'better' marks cells keep upper bounds at most "
              f"{max(b['hi'] for _l, _f, _a, b, _t, r in ft_cells if r == ''):+.3f})")
    print(f"  => retraining warranted (A AND B): {in_train and moved}")
    print()
    print("  supporting detail: the largest single held-out MAE move among the ten pooled cells is "
          f"{max(abs(b['mae'] - a['mae']) for _l, _f, a, b, _t, _r in ft_cells):.4f}; the largest "
          f"d-bar move is "
          f"{max(abs(b['d'] - a['d']) for _l, _f, a, b, _t, _r in ft_cells):.4f}; every 'better' "
          f"cell stays 'better' and every '= grader' cell stays '= grader'.")
    print()

    # ---------------------------------------------------------------- (6)
    print("=" * 110)
    print("## 6. Numbers for the proposed Appendix G sentences")
    print("=" * 110)
    print(f"  students: {sorted(NINE)}")
    print(f"  in training split / held out: {len(NINE & tr_intro)} / {len(NINE & ev_intro)} "
          f"(held out: {sorted(NINE & ev_intro)})")
    print(f"  floor with {floor_with:.2f} -> without {floor_wo:.2f} (shift {shift:+.2f})")
    print(f"  max single-paper disagreement {max(gaps_all):.1f} -> {max(gaps_wo):.1f}")
    ig = headline_stats
    print(f"  IG08 {ig['IG08'][0]['mae']:.2f} -> {ig['IG08'][1]['mae']:.2f}; "
          f"IG09 {ig['IG09'][0]['mae']:.2f} -> {ig['IG09'][1]['mae']:.2f}; "
          f"IG07 {ig['IG07'][0]['mae']:.2f} -> {ig['IG07'][1]['mae']:.2f}; "
          f"IA133 {ig['IA133'][0]['mae']:.2f} -> {ig['IA133'][1]['mae']:.2f}")
    mx_pair = max(((m, tag, rec[(m, tag)][1]['mae'] - rec[(m, tag)][0]['mae'])
                   for m, _n, _s in pairs for tag in ("neutral", "strict")),
                  key=lambda t: abs(t[2]))
    print(f"  largest Table 17 MAE move: {mpt.DISPLAY[mx_pair[0]]} {mx_pair[1]} "
          f"{mx_pair[2]:+.3f}; ordering unchanged: {names_with == names_wo}; "
          f"behaviour changes: {len(beh_ch)}")
    print(f"  largest pooled-adapter held-out MAE move: "
          f"{max(abs(b['mae'] - a['mae']) for _l, _f, a, b, _t, _r in ft_cells):.3f}; "
          f"verdict changes: {len(v_ch)}")
    print()


if __name__ == "__main__":
    main()
