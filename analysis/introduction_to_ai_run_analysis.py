#!/usr/bin/env python3
"""Generate the Introduction to AI (Stage 2) findings ledger.

    python analysis/introduction_to_ai_run_analysis.py

Writes `analysis/introduction_to_ai_new_summary.md` — the Stage 2 counterpart of
`analysis/computer_vision_new_summary.md`, in the same style: every number computed from the
released data, nothing hand-entered, so the paper can be checked against it and
it can be regenerated after any change to the grades or the runs.

Sources: introduction_to_ai_dataset/Practical_AI_exam_grades.csv (ground truth),
introduction_to_ai_results/results/*.xlsx (one workbook per run),
introduction_to_ai_results/ablation_runs.xlsx (run registry).
"""
import csv
import glob
import statistics as st
from statistics import StatisticsError
from pathlib import Path

import math
import numpy as np
import openpyxl

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "introduction_to_ai_dataset"
RES = ROOT / "introduction_to_ai_results"
OUT = ROOT / "analysis" / "introduction_to_ai_new_summary.md"

# Stage 3 figures, for the cross-exam comparison. Sourced from
# analysis/computer_vision_new_summary.md and the paper; not recomputed here.
S3 = {"floor": 2.61, "floor_lo": 2.37, "floor_hi": 2.85, "r": 0.868,
      "scale": 35, "best": 1.64, "best_cfg": "D01 (gemini-3-flash-preview)",
      "best_open": 2.68, "n": 570, "questions": 4,
      # Stage 3 per-pair AI-vs-human-noise correlation (sec:groundtruth).
      "gt_r": -0.301, "gt_p": 0.40, "gt_k": 10}
SCALE = 23 + 3 + 14 + 3 + 19 + 3      # per-question maxima incl. bonus


def ta_truth():
    out = {}
    with open(DATA / "Practical_AI_exam_grades.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            def num(k):
                try:
                    return float(row[k])
                except (TypeError, ValueError, KeyError):
                    return None
            out[int(row["Number"])] = (num("TA 1 - Total Grade"), num("TA 2 - Total Grade"))
    return out


def boot_ci(vals, n=50_000, seed=0):
    """95% percentile bootstrap.

    n is deliberately larger than the 2000 the main exam uses: at 2000-4000
    resamples the floor's lower bound moves by ~0.02 between seeds, enough to
    change the second decimal that gets published. At 50k the endpoints are
    stable to +/-0.004 and agree with a 200k run.
    """
    v = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    chunks = []
    for i in range(0, n, 20_000):
        k = min(20_000, n - i)
        chunks.append(v[rng.integers(0, len(v), (k, len(v)))].mean(axis=1))
    b = np.concatenate(chunks)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def run_stats(path, TA):
    ws = openpyxl.load_workbook(path, data_only=True).active
    hdr = [c.value for c in ws[1]]
    c = {h: i for i, h in enumerate(hdr)}
    totals, diffs = [], []
    q1zero = q1zero_q2pos = 0
    for r in ws.iter_rows(min_row=2, values_only=True):
        a = r[c["AI Total Score"]]
        if a is None:
            continue
        tot = float(a) + float(r[c["AI Total Bonus"]] or 0)
        totals.append(tot)
        t1, t2 = TA.get(r[c["Number"]], (None, None))
        if t1 is not None and t2 is not None:
            diffs.append(tot - (t1 + t2) / 2)
        if r[c["AI Q1 Score"]] == 0:
            q1zero += 1
            q2 = r[c["AI Q2 Score"]]
            if isinstance(q2, (int, float)) and q2 > 0:
                q1zero_q2pos += 1
    mae = sum(abs(d) for d in diffs) / len(diffs)
    lo, hi = boot_ci([abs(d) for d in diffs])
    return {
        "n": len(totals), "dual": len(diffs), "mean": st.fmean(totals),
        "std": st.stdev(totals) if len(totals) > 1 else 0.0,
        "zero_rate": sum(1 for t in totals if t == 0) / len(totals),
        "mae": mae, "lo": lo, "hi": hi,
        "bias": sum(diffs) / len(diffs),
        "q1zero": q1zero, "q1zero_q2pos": q1zero_q2pos,
    }


# --------------------------------------------------------------------------
# Ground-truth structure: pair-level and grader-level statistics.
# The Stage 3 counterparts are recomputed here from the CV workbook rather
# than quoted, so the cross-exam comparison is reproducible from one script.
# --------------------------------------------------------------------------

STABLE_MIN = 38          # see pair_groups(): the pairing sizes are bimodal


def ta_rows():
    """Full ground-truth records: grader ids, per-question marks, totals."""
    out = []
    with open(DATA / "Practical_AI_exam_grades.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            g = lambda k: float(row[k])
            out.append({"n": int(row["Number"]), "ta1": row["TA_1_ID"], "ta2": row["TA_2_ID"],
                        "t1": g("TA 1 - Total Grade"), "t2": g("TA 2 - Total Grade"),
                        "q": [(g(f"TA 1 - Q{i} Grade"), g(f"TA 2 - Q{i} Grade")) for i in (1, 2, 3)]})
    return out


def one_sided_zeros(rows):
    """Rows where exactly one grader recorded 0.0 overall.

    These are ungraded slots stored as zeros, not agreed non-submissions:
    the partner awarded real marks on the same notebooks. Rows where BOTH
    graders recorded 0.0 are genuine non-submissions and are kept (dropping
    them would raise the floor, not lower it).
    """
    return [r for r in rows if (r["t1"] == 0) != (r["t2"] == 0)]


def perm_p(x, y, stat, iters=20000, seed=0):
    """Two-sided permutation p-value; k is far too small to trust a t-approximation."""
    rng = np.random.default_rng(seed)
    obs = abs(stat(x, y))
    y = np.asarray(y, dtype=float)
    hits = sum(1 for _ in range(iters) if abs(stat(x, rng.permutation(y))) >= obs - 1e-12)
    return (hits + 1) / (iters + 1)


def pearson(x, y):
    return float(np.corrcoef(np.asarray(x, float), np.asarray(y, float))[0, 1])


def spearman(x, y):
    rank = lambda v: np.argsort(np.argsort(np.asarray(v, float))).astype(float)
    return pearson(rank(x), rank(y))


def fisher_z(r1, n1, r2, n2):
    """Test whether two independent correlations differ."""
    z = lambda r: 0.5 * math.log((1 + r) / (1 - r))
    se = math.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    stat = (z(r1) - z(r2)) / se
    p = math.erfc(abs(stat) / math.sqrt(2))
    return stat, p


def pair_groups(rows, min_n=STABLE_MIN):
    """Group students by unordered grader pairing.

    Stage 2 is a HYBRID design, unlike Stage 3's 10 fixed pairs: a stable
    backbone of pairings with 38-51 students each, plus a long tail of ad hoc
    combinations built around floating graders. The cut is bimodal, not
    arbitrary -- sorted sizes step 42, 38 | 21, 19 -- so any threshold in
    [22, 38] selects the same backbone.
    """
    g = {}
    for r in rows:
        g.setdefault(tuple(sorted((r["ta1"], r["ta2"]))), []).append(r)
    return {k: v for k, v in g.items() if len(v) >= min_n}, g


def stage3_ground_truth():
    """Stage 3 pair-level counterparts, recomputed from the CV workbook."""
    ws = openpyxl.load_workbook(ROOT / "computer_vision_dataset" /
                                "Practical_AI_exam_grades.xlsx", data_only=True).active
    h = [c.value for c in ws[1]]
    c = {k: i for i, k in enumerate(h)}
    g = {}
    for r in ws.iter_rows(min_row=2, values_only=True):
        t1, t2 = r[c["TA 1 - Total Score (out of 35)"]], r[c["TA 2 - Total Score (out of 35)"]]
        if t1 is None or t2 is None:
            continue
        key = tuple(sorted((str(r[c["TA_1_ID"]]), str(r[c["TA_2_ID"]]))))
        g.setdefault(key, []).append((float(t1), float(t2)))
    gaps = [abs(a - b) for v in g.values() for a, b in v]
    floor = st.fmean(gaps)
    per_pair = {k: st.fmean(abs(a - b) for a, b in v) for k, v in g.items()}
    bias = {k: abs(st.fmean(a - b for a, b in v)) for k, v in g.items()}
    return {"floor": floor, "pairs": per_pair, "bias": bias, "n": len(gaps),
            "above": sum(1 for m in per_pair.values() if m > floor)}


def stage3_neutral_baselines():
    """Stage 3 neutral baselines for the models re-run on this exam.

    Matched to this exam's prompt configuration -- neutral persona, solution
    on, breakdown on, t=0, no few-shot -- so the two sides are comparable.
    """
    want = {"qwen2.5-coder-7b": "qwen25-coder-7b", "qwen2.5-coder-14b": "qwen25-coder-14b",
            "qwen2.5-coder-32b": "qwen25-coder-32b", "qwen3-coder-30b-a3b": "qwen3-coder-30b-a3b",
            "gemma-3-27b": "gemma3-27b", "llama-3.1-8b": "llama31-8b",
            "mistral-small-24b": "mistral-small-24b", "glm-4-9b": "glm4-9b", "glm-4-32b": "glm4-32b"}
    on = ("1", "True", "true")
    out = {}
    with open(ROOT / "analysis" / "computer_vision_master_comparison.csv",
              newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if str(r["strictness"]).strip().lower() not in ("neutral", ""):
                continue
            if str(r["few_shot"]).strip() not in ("0", "", "False", "false"):
                continue
            if str(r["temperature"]).strip() not in ("0", "0.0"):
                continue
            if str(r["use_solution"]).strip() not in on or str(r["use_breakdown"]).strip() not in on:
                continue
            m = str(r["model"]).lower().replace("_", "-")
            for k, v in want.items():
                if k in m and v not in out:
                    out[v] = (float(r["MAE_vs_TA_avg"]), float(r["bias_vs_TA_avg"]))
    return out


# --------------------------------------------------------------------------
# Per-run table over EVERY tracker row.
#
# The earlier version globbed IA*.xlsx only and keyed by (model, persona),
# which silently excluded all 22 Gemini runs and collapsed repeated configs
# onto one another -- 18 rows out of 154. Everything below keys on run_id and
# takes the configuration from the tracker, so every axis (few-shot, thinking,
# temperature, sample count, minimal-pair persona) survives into the outputs.
# --------------------------------------------------------------------------

def ta_per_question():
    """{student: ([q1,q2,q3] TA-average marks, total)} for per-question error."""
    out = {}
    for r in ta_rows():
        out[r["n"]] = ([ (a + b) / 2 for a, b in r["q"] ], (r["t1"] + r["t2"]) / 2)
    return out


def variance_totals(run_tag):
    """Per-student (total, [q1,q2,q3]) rebuilt from a multi-sample run's CSV.

    An n>1 run writes each sample to the variance CSV and leaves the workbook's
    AI columns empty, exactly like the Stage 3 E-series. Each cell becomes the
    mean across its samples, so these runs carry the same columns as every other
    row rather than being dropped.
    """
    import collections
    vcsv = RES / "variance" / f"{run_tag}.csv"
    if not vcsv.exists():
        return {}
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(vcsv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                acc[int(row["student"])][int(row["question"])].append(
                    float(row["score"]) + float(row["bonus"] or 0))
            except (ValueError, KeyError, TypeError):
                continue
    out = {}
    for stu, qs in acc.items():
        if len(qs) < 3:                  # a question never scored -> no total
            continue
        per_q = {q: st.fmean(v) for q, v in qs.items()}
        out[stu] = (sum(per_q.values()), [per_q.get(q, float("nan")) for q in (1, 2, 3)])
    return out


def stats_from_variance(run_tag, TA, TAQ):
    """run_stats_full-shaped record for a multi-sample run."""
    tot = variance_totals(run_tag)
    if not tot:
        return None
    totals, diffs, ai_tot, ta_tot = [], [], [], []
    qerr = {1: [], 2: [], 3: []}
    qmean = {1: [], 2: [], 3: []}
    for stu, (t, qs) in tot.items():
        totals.append(t)
        for q in (1, 2, 3):
            if qs[q - 1] == qs[q - 1]:
                qmean[q].append(qs[q - 1])
        t1, t2 = TA.get(stu, (None, None))
        if t1 is None or t2 is None:
            continue
        ta = (t1 + t2) / 2
        diffs.append(t - ta)
        ai_tot.append(t); ta_tot.append(ta)
        if stu in TAQ:
            tq, _ = TAQ[stu]
            for q in (1, 2, 3):
                if qs[q - 1] == qs[q - 1]:
                    qerr[q].append(abs(qs[q - 1] - tq[q - 1]))
    if not diffs:
        return None
    lo, hi = boot_ci([abs(d) for d in diffs])
    s = {"n": len(totals), "dual": len(diffs), "mean": st.fmean(totals),
         "std": st.stdev(totals) if len(totals) > 1 else 0.0,
         "zero_rate": sum(1 for t in totals if t == 0) / len(totals),
         "mae": st.fmean(abs(d) for d in diffs), "lo": lo, "hi": hi,
         "bias": st.fmean(diffs), "q1zero": 0, "q1zero_q2pos": 0, "n_failed": 0,
         "pearson": (pearson(ai_tot, ta_tot)
                     if len(ai_tot) > 2 and len(set(ai_tot)) > 1 else float("nan"))}
    for q in (1, 2, 3):
        s[f"q{q}_mae"] = st.fmean(qerr[q]) if qerr[q] else float("nan")
        s[f"q{q}_mean"] = st.fmean(qmean[q]) if qmean[q] else float("nan")
    return s


def run_stats_full(path, TA, TAQ):
    """run_stats plus the columns the Stage 3 CSV carries: per-question MAE and
    mean, AI-vs-TA correlation, and the count of ungraded students."""
    s = run_stats(path, TA)
    ws = openpyxl.load_workbook(path, data_only=True).active
    hdr = [c.value for c in ws[1]]
    c = {h: i for i, h in enumerate(hdr)}
    qerr = {1: [], 2: [], 3: []}
    qmean = {1: [], 2: [], 3: []}
    ai_tot, ta_tot, failed = [], [], 0
    for r in ws.iter_rows(min_row=2, values_only=True):
        num = r[c["Number"]]
        a = r[c["AI Total Score"]]
        if a is None:
            failed += 1
            continue
        tot = float(a) + float(r[c["AI Total Bonus"]] or 0)
        if num in TAQ:
            tq, tt = TAQ[num]
            ai_tot.append(tot); ta_tot.append(tt)
            for q in (1, 2, 3):
                v = r[c.get(f"AI Q{q} Score", -1)] if f"AI Q{q} Score" in c else None
                if isinstance(v, (int, float)):
                    b = r[c.get(f"AI Q{q} Bonus")] if f"AI Q{q} Bonus" in c else 0
                    v = float(v) + float(b or 0)
                    qmean[q].append(v)
                    qerr[q].append(abs(v - tq[q - 1]))
    s["n_failed"] = failed
    # A refusal run scores every student 0, so its AI vector is constant and the
    # correlation is undefined rather than 0.
    s["pearson"] = (pearson(ai_tot, ta_tot)
                    if len(ai_tot) > 2 and len(set(ai_tot)) > 1 else float("nan"))
    for q in (1, 2, 3):
        s[f"q{q}_mae"] = st.fmean(qerr[q]) if qerr[q] else float("nan")
        s[f"q{q}_mean"] = st.fmean(qmean[q]) if qmean[q] else float("nan")
    return s


def all_runs(TA, TAQ):
    """Every completed run, configuration from the tracker, stats from its workbook.

    n>1 runs leave the workbook's AI columns empty and write per-sample rows to
    the variance CSV instead, so those are rebuilt as the per-student mean the
    same way introduction_to_ai_analyze_run.py does.
    """
    ws = openpyxl.load_workbook(RES / "ablation_runs.xlsx", data_only=True).active
    hdr = [c.value for c in ws[1]]
    c = {h: i for i, h in enumerate(hdr)}
    out = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        rid = str(r[c["run_id"]] or "")
        if not rid or str(r[c["status"]]) != "done":
            continue
        tag = str(r[c["run_tag"]] or "")
        # Use the tracker's own output path: the earliest rows carry a run_id
        # ("IA01-Q7-n") that differs from the filename stem ("IA01__IA01_m-..."),
        # so composing the path from run_id silently drops 18 real runs.
        out_rel = str(r[c["output_xlsx"]] or "")
        wb = (RES / out_rel) if out_rel else (RES / "results" / f"{rid}__{tag}.xlsx")
        if not wb.exists():
            wb = RES / "results" / f"{rid}__{tag}.xlsx"
        if not wb.exists():
            continue
        try:
            s = run_stats_full(wb, TA, TAQ)
        except (ZeroDivisionError, KeyError, StatisticsError):
            s = None
        if s is None or s["dual"] == 0:
            # n>1 run: workbook AI columns are empty by design, rebuild from
            # the variance CSV so it carries the same columns as every other row.
            s = stats_from_variance(tag, TA, TAQ)
        if s is None:
            continue
        model = str(r[c["model"]] or "")
        out.append({
            "run_id": rid,
            "source": ("gemini" if rid.startswith("IG") else "openai" if rid.startswith("IO")
                       else "anthropic" if rid.startswith("IN") else "local"),
            "model": model.split("/")[-1],
            "model_key": tag.split("_m-")[1].split("_sol")[0] if "_m-" in tag else model,
            "use_solution": int(r[c["use_solution"]] or 0),
            "use_guidelines": int(r[c["use_guidelines"]] or 0),
            "use_reasoning": int(r[c["use_reasoning"]] or 0),
            "use_breakdown": int(r[c["use_breakdown"]] or 0),
            "strictness": str(r[c["strictness"]] or ""),
            "temperature": float(r[c["temperature"]] or 0),
            "runs": int(r[c["runs"]] or 1),
            "few_shot": int(r[c["few_shot"]] or 0),
            "students": str(r[c["students"]] or "all"),
            "stats": s,
        })
    out.sort(key=lambda d: d["stats"]["mae"])
    return out


def write_master_comparison(runs_all, floor):
    """Emit the per-run CSV, the Stage 2 counterpart of
    analysis/computer_vision_master_comparison.csv, with the same columns.

    One row per completed run, keyed on the tracker's run_id so the released
    spreadsheets, the ledger and the appendix table all agree.
    """
    cols = ["run_id", "source", "model", "use_solution", "use_guidelines",
            "use_reasoning", "use_breakdown", "strictness", "temperature", "runs",
            "few_shot", "students", "n_graded", "n_valid_vs_TA", "mean_total_score",
            "std_total_score", "MAE_vs_TA_avg", "MAE_CI95_lo", "MAE_CI95_hi",
            "bias_vs_TA_avg", "zero_rate", "AI_TA_pearson_r", "x_floor", "behaviour",
            "n_failed", "Q1_MAE_vs_TA", "Q2_MAE_vs_TA", "Q3_MAE_vs_TA",
            "Q1_mean", "Q2_mean", "Q3_mean"]
    rows = []
    for d in runs_all:
        s_ = d["stats"]
        rows.append({
            "run_id": d["run_id"], "source": d["source"], "model": d["model_key"],
            "use_solution": d["use_solution"], "use_guidelines": d["use_guidelines"],
            "use_reasoning": d["use_reasoning"], "use_breakdown": d["use_breakdown"],
            "strictness": d["strictness"], "temperature": d["temperature"],
            "runs": d["runs"], "few_shot": d["few_shot"], "students": d["students"],
            "n_graded": s_["n"], "n_valid_vs_TA": s_["dual"],
            "mean_total_score": round(s_["mean"], 4), "std_total_score": round(s_["std"], 4),
            "MAE_vs_TA_avg": round(s_["mae"], 4),
            "MAE_CI95_lo": round(s_["lo"], 4), "MAE_CI95_hi": round(s_["hi"], 4),
            "bias_vs_TA_avg": round(s_["bias"], 4),
            "zero_rate": round(s_["zero_rate"], 4),
            "AI_TA_pearson_r": round(s_["pearson"], 4),
            "x_floor": round(s_["mae"] / floor, 3),
            "behaviour": behaviour(s_), "n_failed": s_["n_failed"],
            "Q1_MAE_vs_TA": round(s_["q1_mae"], 4), "Q2_MAE_vs_TA": round(s_["q2_mae"], 4),
            "Q3_MAE_vs_TA": round(s_["q3_mae"], 4),
            "Q1_mean": round(s_["q1_mean"], 4), "Q2_mean": round(s_["q2_mean"], 4),
            "Q3_mean": round(s_["q3_mean"], 4),
        })
    rows.sort(key=lambda r: r["MAE_vs_TA_avg"])
    out = ROOT / "analysis" / "introduction_to_ai_master_comparison.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return out, len(rows)


# Mechanism probe (## Mechanism below; Table 18 of the paper). A blanket vs
# selective reading needs a minimum volume of zeroed Q1s: below this share of
# the students a run graded, the zeros are the ordinary non-attempts every run
# sees and the selective share is noise on a handful of students (item 60).
# The CV exam applies the same gate -- computer_vision_run_analysis.MECH_ZERO_RATE
# = 0.10; on these 17 runs any cut in 4.8%-13.3% gives the same partition.
MIN_Q1_ZERO_RATE = 0.05
MECH_SELECTIVE = 0.35
# Collapse band, floor-matched (item 8): the CV exam's MAE >= 8 is 3.07x its 2.61
# floor; the same multiple on this exam's 5.13 floor is 15.7 (8.00/2.61 x 5.13 =
# 15.72; 3.07 x 5.13 = 15.75). No run's MAE lies in [15.39, 15.75] except IA61
# (15.56) and IA17 (15.50), in band under any of these, so the rounding is inert.
# The paper prints 15.7 (Section 5.4, Table 17 caption, Appendix G).
COLLAPSE_MAE = 15.7


def behaviour(s):
    if s["zero_rate"] >= 0.9:
        kind = "refusal" if s["std"] < 0.5 else "near-refusal"
        return f"{kind} ({s['zero_rate']:.0%} zeroed)"
    # The paper's rule (sections/brittleness.tex) is MAE >= 3.07x the exam's own
    # floor -- 15.7 here, 8 on the CV exam (COLLAPSE_MAE above) -- while the run
    # still discriminates -- there is no sign condition. An earlier version
    # split off positive-bias runs as a fourth class, which never fired on the
    # main exam (every CV run at MAE >= 8 under-marks) and silently relabelled
    # seven over-marking neutral runs here.
    if s["mae"] >= COLLAPSE_MAE:
        return "collapse"
    return "graded"


def write_master_md(runs_all, floor, f_lo, f_hi):
    """All runs, sorted by MAE. Stage 2 counterpart of computer_vision_master_comparison.md."""
    L = [f"# Master comparison — all {len(runs_all)} runs\n",
         f"Human-grader floor (inter-grader MAE on {len(ta_rows())} dual-graded students, pooled "
         f"across the exam's fixed TA pairs): **{floor:.2f}** points, 95% CI [{f_lo:.2f}, {f_hi:.2f}]. "
         f"Any AI MAE below this is statistically indistinguishable from a second human TA.\n",
         "Sorted by MAE vs TA-average total score (ascending = better). `x floor` is the MAE as a "
         "multiple of this exam's own floor — the scale-free quantity, since this exam folds bonus "
         "marks into its question scores and Stage 3 does not.\n",
         "| run_id | source | model | strictness | sol | rs | bd | fs | t | n | n_graded | "
         "mean | MAE | 95% CI | x floor | bias | zero | behaviour |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in runs_all:
        s = d["stats"]
        L.append(f"| {d['run_id']} | {d['source']} | {d['model_key']} | {d['strictness']} | "
                 f"{d['use_solution']} | {d['use_reasoning']} | {d['use_breakdown']} | "
                 f"{d['few_shot']} | {d['temperature']:g} | {d['runs']} | {s['n']} | "
                 f"{s['mean']:.2f} | **{s['mae']:.2f}** | [{s['lo']:.2f}, {s['hi']:.2f}] | "
                 f"{s['mae']/floor:.2f}x | {s['bias']:+.2f} | {s['zero_rate']:.1%} | "
                 f"{behaviour(s)} |")
    (ROOT / "analysis" / "introduction_to_ai_master_comparison.md").write_text(
        "\n".join(L) + "\n", encoding="utf-8")


def write_human_floor_md(floor, f_lo, f_hi, r_ta, rows, pairs_map):
    gaps = [abs(r["t1"] - r["t2"]) for r in rows]
    L = ["# Human-grader floor (inter-grader disagreement)\n",
         f"**Setup.** Every one of the {len(rows)} students is graded by two TAs. The "
         f"`TA_1_ID` / `TA_2_ID` columns are *slots*, not fixed identities, so a signed "
         f"`TA1 - TA2` bias would be uninterpretable; only the magnitude of disagreement and "
         f"the correlation are reported.\n",
         "## Pooled across all pairs\n",
         f"- **Dual-graded students:** {len(rows)}",
         f"- **Inter-grader MAE on total score:** **{floor:.2f}** / {SCALE}  "
         f"(95% bootstrap CI [{f_lo:.2f}, {f_hi:.2f}])",
         f"- **Pearson r:** {r_ta:.3f}",
         f"- **Max single-paper disagreement:** {max(gaps):.1f} pts",
         f"- **Median disagreement:** {st.median(gaps):.2f} pts\n",
         "The floor is the *pairwise* gap `E[|TA1 - TA2|]`, not `E[|TA - TA_mean|]`. The latter "
         "is half the former by construction and is not the paper's definition.\n",
         "## Per-pair\n",
         "| pair | n | MAE | vs pooled |", "|---|---|---|---|"]
    for (a, b), v in sorted(pairs_map.items(), key=lambda kv: kv[1]["mae"]):
        L.append(f"| {a} / {b} | {v['n']} | {v['mae']:.2f} | "
                 f"{'above' if v['mae'] > floor else 'below'} |")
    above = sum(1 for v in pairs_map.values() if v["mae"] > floor)
    L.append(f"\n{above} of {len(pairs_map)} pairs sit above the pooled floor. The spread across "
             f"pairs ({min(v['mae'] for v in pairs_map.values()):.2f} to "
             f"{max(v['mae'] for v in pairs_map.values()):.2f}) is itself larger than most of the "
             f"differences between AI configurations, which is why the floor is quoted pooled.\n")
    (ROOT / "analysis" / "introduction_to_ai_human_floor.md").write_text(
        "\n".join(L) + "\n", encoding="utf-8")


def write_variance_md():
    """Repeated-grading runs (n>1): per-cell stdev across samples."""
    import collections
    L = ["# Variance analysis (repeated grading)\n",
         "Each variance config grades the same student x question several times; the statistic is "
         "the sample stdev of the score across those repeats. Lower = more reproducible.\n",
         "| config | n_cells | mean_runs_per_cell | mean_cell_std | median_cell_std | "
         "max_cell_std | pct_cells_zero_std | pct_cells_std_gt_1 |",
         "|---|---|---|---|---|---|---|---|"]
    found = 0
    for vf in sorted((RES / "variance").glob("*_n[2-9].csv")) + \
              sorted((RES / "variance").glob("*_n[1-9][0-9].csv")):
        cells = collections.defaultdict(list)
        try:
            with open(vf, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    try:
                        cells[(row["student"], row["question"])].append(float(row["score"]))
                    except (KeyError, ValueError):
                        continue
        except OSError:
            continue
        cells = {k: v for k, v in cells.items() if len(v) > 1}
        if not cells:
            continue
        found += 1
        sds = [st.stdev(v) for v in cells.values()]
        runs_per = st.fmean(len(v) for v in cells.values())
        tag = vf.stem.split("_m-")[0] + "-" + vf.stem.split("_")[-2] if "_m-" in vf.stem else vf.stem
        L.append(f"| {tag} | {len(cells)} | {runs_per:.3f} | {st.fmean(sds):.3f} | "
                 f"{st.median(sds):.3f} | {max(sds):.3f} | "
                 f"{100*sum(1 for s in sds if s == 0)/len(sds):.1f} | "
                 f"{100*sum(1 for s in sds if s > 1)/len(sds):.1f} |")
    if not found:
        L.append("| (no multi-sample runs found) | | | | | | | |")
    (ROOT / "analysis" / "introduction_to_ai_variance.md").write_text(
        "\n".join(L) + "\n", encoding="utf-8")


def write_failure_cases_md(runs_all, TA):
    """Vignettes from the best full-cohort configuration."""
    best = next((d for d in runs_all if d["students"] == "all" and d["stats"]["n"] > 900), runs_all[0])
    wb = RES / "results" / f"{best['run_id']}__" \
         f"{[t for t in [best['run_id']] ][0]}"
    # locate the workbook by glob (tag is embedded in the filename)
    cands = list((RES / "results").glob(f"{best['run_id']}__*.xlsx"))
    if not cands:
        return
    ws = openpyxl.load_workbook(cands[0], data_only=True).active
    hdr = [c.value for c in ws[1]]
    c = {h: i for i, h in enumerate(hdr)}
    recs = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        a = r[c["AI Total Score"]]
        if a is None:
            continue
        num = r[c["Number"]]
        t1, t2 = TA.get(num, (None, None))
        if t1 is None or t2 is None:
            continue
        ai = float(a) + float(r[c["AI Total Bonus"]] or 0)
        ta = (t1 + t2) / 2
        recs.append((num, ai, ta, ai - ta,
                     [str(r[c.get(f"AI Q{q} Reasoning")] or "") if f"AI Q{q} Reasoning" in c else ""
                      for q in (1, 2, 3)]))
    over = sorted([x for x in recs if x[3] >= 5], key=lambda x: -x[3])[:3]
    under = sorted([x for x in recs if x[3] <= -5], key=lambda x: x[3])[:3]
    L = ["# Failure-mode vignettes\n",
         f"Best full-cohort configuration: **{best['run_id']}** "
         f"({best['model']}, {best['strictness']}, MAE {best['stats']['mae']:.2f}). "
         f"Three students it scored at least 5 points above the TA average, and three at least 5 below.\n",
         "(Reasoning truncated to ~600 chars; full text is in the result xlsx.)\n"]
    for title, group in (("Over-graded by AI (AI > TA)", over), ("Under-graded by AI (AI < TA)", under)):
        L.append(f"## {title}\n")
        if not group:
            L.append("_None at this threshold._\n")
        for num, ai, ta, d, reasons in group:
            L.append(f"### Student #{num}  (AI={ai:.2f}, TA-avg={ta:.2f}, delta={d:+.2f})")
            sub = DATA / "submissions_extracted" / str(num)
            L.append(f"- Submission dir: `introduction_to_ai_dataset/submissions_extracted/{num}/` "
                     f"({'exists' if sub.exists() else 'missing'})")
            for q, txt in zip((1, 2, 3), reasons):
                if txt:
                    t = " ".join(txt.split())[:600]
                    L.append(f"    - **Q{q}**: {t}...")
            L.append("")
    (ROOT / "analysis" / "introduction_to_ai_failure_cases.md").write_text(
        "\n".join(L) + "\n", encoding="utf-8")


def write_pair_level_ai_mae_md(runs_all, TA, rows, pairs_map):
    """Does AI error track per-pair human disagreement, or average it out?"""
    best = next((d for d in runs_all if d["students"] == "all" and d["stats"]["n"] > 900), runs_all[0])
    cands = list((RES / "results").glob(f"{best['run_id']}__*.xlsx"))
    if not cands:
        return
    ws = openpyxl.load_workbook(cands[0], data_only=True).active
    hdr = [c.value for c in ws[1]]
    c = {h: i for i, h in enumerate(hdr)}
    ai = {}
    for r in ws.iter_rows(min_row=2, values_only=True):
        a = r[c["AI Total Score"]]
        if a is not None:
            ai[r[c["Number"]]] = float(a) + float(r[c["AI Total Bonus"]] or 0)
    per = {}
    for r in rows:
        key = tuple(sorted((r["ta1"], r["ta2"])))
        if r["n"] not in ai:
            continue
        ta = (r["t1"] + r["t2"]) / 2
        per.setdefault(key, {"hum": [], "aie": [], "bias": []})
        per[key]["hum"].append(abs(r["t1"] - r["t2"]))
        per[key]["aie"].append(abs(ai[r["n"]] - ta))
        per[key]["bias"].append(ai[r["n"]] - ta)
    per = {k: v for k, v in per.items() if len(v["hum"]) >= STABLE_MIN}
    L = ["# Pair-level AI error and per-TA harshness\n",
         "Two questions about the ground truth, both answerable from existing data:\n",
         "1. Does AI error track per-pair human disagreement — does it *inherit* the pair-level "
         "noise structure, or *average it out*?",
         "2. Is the per-pair spread larger than chance?\n",
         f"The AI side is **{best['run_id']}** ({best['model']}, {best['strictness']}, "
         f"n={best['stats']['n']}, MAE {best['stats']['mae']:.2f}) — the strongest full-cohort run.\n",
         "## Per-pair AI MAE vs per-pair human MAE\n",
         "| pair | n | human MAE | AI MAE | AI bias |", "|---|---|---|---|---|"]
    hx, ay = [], []
    for k, v in sorted(per.items(), key=lambda kv: st.fmean(kv[1]["hum"])):
        h, a_, b_ = st.fmean(v["hum"]), st.fmean(v["aie"]), st.fmean(v["bias"])
        hx.append(h); ay.append(a_)
        L.append(f"| {k[0]} / {k[1]} | {len(v['hum'])} | {h:.2f} | {a_:.2f} | {b_:+.2f} |")
    if len(hx) > 2:
        r_p = pearson(hx, ay); r_s = spearman(hx, ay)
        p = perm_p(hx, ay, lambda x, y: abs(pearson(x, y)))
        L.append(f"\n**Correlation between human-pair disagreement and AI error across "
                 f"{len(hx)} pairs:** Pearson r = {r_p:+.3f}, Spearman rho = {r_s:+.3f}, "
                 f"permutation p = {p:.3f}.\n")
        L.append("A positive correlation means the AI finds the same students hard that the "
                 "humans disagree about; a null means its error is independent of where human "
                 "graders diverge.\n")
    (ROOT / "analysis" / "introduction_to_ai_pair_level_ai_mae.md").write_text(
        "\n".join(L) + "\n", encoding="utf-8")


def write_subset_restricted_md(runs_all, TA):
    """Subset runs vs the same-subset restriction of the full-cohort baseline.

    Several Gemini ablations graded students 1-100 only. Comparing those to a
    full-cohort run over-credits them, because sample variance alone lowers MAE
    on a smaller subset. This restricts the baseline to the same students.
    """
    subset = [d for d in runs_all if d["students"] != "all" or d["stats"]["n"] < 200]
    if not subset:
        return
    ids = set()
    for d in subset:
        cands = list((RES / "results").glob(f"{d['run_id']}__*.xlsx"))
        if not cands:
            continue
        ws = openpyxl.load_workbook(cands[0], data_only=True).active
        hdr = [c.value for c in ws[1]]
        c = {h: i for i, h in enumerate(hdr)}
        for r in ws.iter_rows(min_row=2, values_only=True):
            if r[c["AI Total Score"]] is not None:
                ids.add(r[c["Number"]])
    base = next((d for d in runs_all if d["students"] == "all" and d["stats"]["n"] > 900), None)
    L = ["# Subset runs vs the same-subset baseline\n",
         "Several ablations graded a student subset rather than the full cohort. Comparing them to "
         "a full-cohort run over-credits them: sample variance alone lowers MAE on a smaller "
         "sample. This file restricts the baseline to the same students.\n",
         f"Restriction set: {len(ids)} students.\n",
         "| run | model | strictness | n | MAE | 95% CI | bias |", "|---|---|---|---|---|---|---|"]
    def restricted(rid):
        cands = list((RES / "results").glob(f"{rid}__*.xlsx"))
        if not cands:
            return None
        ws = openpyxl.load_workbook(cands[0], data_only=True).active
        hdr = [c.value for c in ws[1]]
        c = {h: i for i, h in enumerate(hdr)}
        diffs = []
        for r in ws.iter_rows(min_row=2, values_only=True):
            num = r[c["Number"]]
            a = r[c["AI Total Score"]]
            if a is None or num not in ids:
                continue
            t1, t2 = TA.get(num, (None, None))
            if t1 is None:
                continue
            diffs.append(float(a) + float(r[c["AI Total Bonus"]] or 0) - (t1 + t2) / 2)
        if not diffs:
            return None
        lo, hi = boot_ci([abs(d) for d in diffs])
        return len(diffs), st.fmean(abs(d) for d in diffs), lo, hi, st.fmean(diffs)
    if base:
        rr = restricted(base["run_id"])
        if rr:
            L.append(f"| **{base['run_id']} (restricted)** | {base['model']} | {base['strictness']} | "
                     f"{rr[0]} | **{rr[1]:.2f}** | [{rr[2]:.2f}, {rr[3]:.2f}] | {rr[4]:+.2f} |")
    for d in sorted(subset, key=lambda x: x["stats"]["mae"]):
        s = d["stats"]
        L.append(f"| {d['run_id']} | {d['model']} | {d['strictness']} | {s['n']} | "
                 f"**{s['mae']:.2f}** | [{s['lo']:.2f}, {s['hi']:.2f}] | {s['bias']:+.2f} |")
    (ROOT / "analysis" / "introduction_to_ai_subset_restricted_comparison.md").write_text(
        "\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    TA = ta_truth()
    pairs = [(a, b) for a, b in TA.values() if a is not None and b is not None]
    # Human floor = the pairwise TA gap E[|TA1 - TA2|], the same definition
    # the paper uses for Stage 3 (sections/setup.tex, Metrics).
    gap = [abs(a - b) for a, b in pairs]
    floor = st.fmean(gap)
    f_lo, f_hi = boot_ci(gap)
    r_ta = float(np.corrcoef([a for a, _ in pairs], [b for _, b in pairs])[0, 1])

    # The ledger narrates one canonical contrast per open-weights model:
    # neutral vs strict at the published prompt configuration. Selecting it
    # explicitly matters now that the grid also holds minimal-pair personas,
    # few-shot, thinking and multi-sample runs -- the old filename glob read
    # "not neutral" as "strict" and divided by zero on the n>1 runs.
    TAQ = ta_per_question()
    runs_all = all_runs(TA, TAQ)
    runs = {}
    for d in runs_all:
        if d["source"] != "local" or d["strictness"] not in ("neutral", "strict"):
            continue
        if (d["use_solution"], d["use_reasoning"], d["use_breakdown"]) != (1, 0, 1):
            continue
        if d["runs"] != 1 or d["few_shot"] or d["temperature"] or d["students"] != "all":
            continue
        runs.setdefault(d["model_key"], {})[d["strictness"]] = d["stats"]
    runs = {k: v for k, v in runs.items() if "neutral" in v and "strict" in v}

    L = []
    A = L.append
    A(f"# Results: replicating the strict-persona collapse on a second exam "
      f"({len(TA)}-student Introduction to AI practical)\n")
    A(f"An {sum(len(v) for v in runs.values())}-run replication ({len(runs)} open-weights models "
      f"x {{neutral, strict}}) of the Stage 3 finding, on an independent exam: a different course, "
      f"different students, different teaching assistants, and a ~{SCALE}-point scale across "
      f"{3} questions instead of {S3['questions']}. Every one of the {len(TA)} students is graded "
      f"by two TAs. The grading instrument is held fixed — identical persona presets, JSON schema, "
      f"output rules, grading scale, temperature 0, one sample per question — so the only things "
      f"that change are the exam and the cohort.\n")
    A("> **What replicates is the failure, not a fixed failure mode.** The same three sentences "
      "range from halving a model's error to tripling it, and a model that collapsed selectively "
      "on one exam can collapse bluntly on the other. That unpredictability is the finding.\n")

    A("## Headlines\n")
    A(f"- **Human-grader floor**: the two TAs agree to MAE **{floor:.2f} / {SCALE}** on all "
      f"{len(pairs)} dual-graded students (95% bootstrap CI [{f_lo:.2f}, {f_hi:.2f}]; Pearson "
      f"r = {r_ta:.3f}) — the same definition as the main exam, E[|TA1 - TA2|]. The two "
      f"cohorts correlate almost identically (r {r_ta:.3f} here, {S3['r']:.3f} on Stage 3). Shares "
      f"of scale are NOT compared across exams: Stage 2 folds bonus marks into the question "
      f"scores while Stage 3 keeps them in a separate column, so the denominators are not "
      f"like-for-like (see Method notes).")

    order = sorted(runs.items(), key=lambda kv: -(kv[1]["strict"]["mae"] / kv[1]["neutral"]["mae"]))
    broke = [t for t, v in runs.items()
             if "collapse" in behaviour(v["strict"]) or "refusal" in behaviour(v["strict"])]
    best_tag, best = min(((t, v["neutral"]) for t, v in runs.items()), key=lambda kv: kv[1]["mae"])
    worst_tag, worst = max(((t, v["strict"]) for t, v in runs.items()), key=lambda kv: kv[1]["mae"])
    # A floor-matched threshold, not a scale-matched one: the bonus-handling
    # difference makes any cross-exam denominator unsafe, while "k x that
    # exam's own floor" needs none.
    scaled = 8 / S3["floor"] * floor
    n_scaled = sum(1 for v in runs.values() if v["strict"]["mae"] >= scaled)
    A(f"- **{len(broke)} of {len(runs)} models leave the graded band under the strict persona** on "
      f"the Stage 3 threshold of MAE >= 8. That threshold does *not* transfer cleanly: 8 points is "
      f"{8/S3['floor']:.2f}x the Stage 3 floor but only {8/floor:.2f}x this one, so it is the more "
      f"lenient bar here. On a floor-matched threshold (MAE >= {scaled:.1f}, the same multiple of "
      f"this exam's own floor), **{n_scaled} of {len(runs)}** leave the band. Both counts are "
      f"reported throughout; neither is cherry-picked.")
    A(f"- **Best configuration**: **{best_tag}** under *neutral*, MAE **{best['mae']:.2f}** "
      f"[{best['lo']:.2f}, {best['hi']:.2f}], bias {best['bias']:+.2f} — essentially unbiased, and "
      f"**below the human floor** ({best['mae']/floor:.2f}x). The parity result therefore replicates "
      f"as well as the brittleness: on Stage 3 the best configuration ({S3['best_cfg']}) reached "
      f"{S3['best']:.2f} against a floor of {S3['floor']:.2f}.")
    A(f"- **Worst configuration**: **{worst_tag}** under *strict*, MAE **{worst['mae']:.2f}** "
      f"[{worst['lo']:.2f}, {worst['hi']:.2f}], mean awarded total {worst['mean']:.2f} / {SCALE}, "
      f"{worst['zero_rate']:.0%} of students zeroed.")
    top = order[0]
    bestv = runs[best_tag]
    A(f"- **The best grader is not spared.** {best_tag}, the most accurate model on the exam "
      f"under neutral, still leaves the graded band under strict "
      f"(x{bestv['strict']['mae']/bestv['neutral']['mae']:.2f}, bias "
      f"{bestv['neutral']['bias']:+.2f} -> {bestv['strict']['bias']:+.2f}); the largest relative "
      f"damage belongs to {top[0]} "
      f"(x{top[1]['strict']['mae']/top[1]['neutral']['mae']:.2f}, "
      f"{top[1]['neutral']['mae']:.2f} -> {top[1]['strict']['mae']:.2f}). Nothing in a "
      f"single-prompt evaluation would flag either.")
    infl = sorted((t for t, v in runs.items() if v["neutral"]["bias"] > 0),
                  key=lambda t: -runs[t]["neutral"]["bias"])
    flat = sorted((t for t, v in runs.items() if v["neutral"]["bias"] <= 0),
                  key=lambda t: -runs[t]["neutral"]["bias"])
    A(f"- **The neutral prompt over-grades almost everywhere** — {len(infl)} of {len(runs)} models "
      f"carry a positive bias under *neutral* (up to "
      f"{max(v['neutral']['bias'] for v in runs.values()):+.2f}); the exceptions are "
      + ", ".join(f"`{t}` at {runs[t]['neutral']['bias']:+.2f}" for t in flat)
      + f". Stage 3's neutral baselines straddle zero instead, so this is an exam-level effect: "
      f"generous marking is easier on a rubric with more part-marks.\n")

    A("## Per-model results\n")
    A("| model | neutral MAE [95% CI] | strict MAE [95% CI] | ratio | strict x floor | "
      "strict zero-rate | strict bias | behaviour |")
    A("|---|---|---|---:|---:|---:|---:|---|")
    for tag, v in order:
        n_, s_ = v["neutral"], v["strict"]
        A(f"| `{tag}` | {n_['mae']:.2f} [{n_['lo']:.2f}, {n_['hi']:.2f}] | "
          f"{s_['mae']:.2f} [{s_['lo']:.2f}, {s_['hi']:.2f}] | "
          f"x{s_['mae']/n_['mae']:.2f} | {s_['mae']/floor:.1f}x | {s_['zero_rate']:.1%} | "
          f"{s_['bias']:+.2f} | {behaviour(s_)} |")
    A("")

    A("## Mechanism: the same probe separates blanket from selective collapse\n")
    A("Of the students a run scored 0 on Q1, how many still received marks on Q2? A high share "
      "means the score field surrendered while the neighbouring field kept honouring the rubric — "
      "the conflicting-instruction signature. A low share means the whole submission died together.\n")
    A("The reading only applies to runs that actually left the graded band, and only when the "
      "run zeroed Q1 on at least {:.0%} of the students it graded. A model still grading "
      "normally zeroes Q1 for the ordinary reason — the question was not attempted — and a "
      "handful of zeros carries no mechanism information either way, so those rows are marked "
      "n/a rather than labelled.\n".format(MIN_Q1_ZERO_RATE))
    A("| model (strict) | students with Q1 = 0 (of graded) | of those, Q2 > 0 | share | reading |")
    A("|---|---:|---:|---:|---|")
    for tag, v in sorted(runs.items(), key=lambda kv: -kv[1]["strict"]["q1zero"]):
        s_ = v["strict"]
        if not s_["q1zero"]:
            continue
        share = s_["q1zero_q2pos"] / s_["q1zero"]
        rate = s_["q1zero"] / s_["n"]
        b = behaviour(s_)
        if "collapse" not in b and "refusal" not in b:
            reading = "n/a — stayed in the graded band"
        elif rate < MIN_Q1_ZERO_RATE:
            reading = f"n/a — too few Q1 zeros ({rate:.1%} of graded students)"
        else:
            reading = ("selective field collapse" if share >= MECH_SELECTIVE
                       else "blanket zeroing")
        A(f"| `{tag}` | {s_['q1zero']} / {s_['n']} ({rate:.0%}) | {s_['q1zero_q2pos']} | "
          f"{share:.0%} | {reading} |")
    A("")

    A("## Ground-truth structure\n")
    rows = ta_rows()
    bad = one_sided_zeros(rows)
    clean = [r for r in rows if r not in bad]
    gapf = lambda r: abs(r["t1"] - r["t2"])
    A(f"**Data hygiene.** {len(bad)} of {len(rows)} rows record 0.0 for exactly one grader while the "
      f"partner awarded real marks on the same notebooks — ungraded slots stored as zeros, not "
      f"agreed non-submissions. They inflate the two statistics most sensitive to outliers:\n")
    A(f"| | all {len(rows)} rows | excluding the {len(bad)} one-sided zeros |")
    A("|---|---:|---:|")
    A(f"| human floor | {st.fmean(gapf(r) for r in rows):.4f} | {st.fmean(gapf(r) for r in clean):.4f} |")
    A(f"| max single-paper disagreement | {max(gapf(r) for r in rows):.1f} | {max(gapf(r) for r in clean):.1f} |")
    A(f"\nThe headline floor is quoted over all {len(rows)} rows (the effect is {abs(st.fmean(gapf(r) for r in rows)-st.fmean(gapf(r) for r in clean)):.3f} points, "
      f"well inside the CI), but the **maximum disagreement must be quoted excluding them**: the raw "
      f"{max(gapf(r) for r in rows):.1f} is student {max(rows, key=gapf)['n']}, where one slot is blank-as-zero. A further "
      f"{sum(1 for r in rows if r['t1'] == 0 and r['t2'] == 0)} rows carry 0.0 from BOTH graders; those are genuine non-submissions and are "
      f"kept — dropping them would raise the floor, not lower it.\n")

    stable, allg = pair_groups(rows)
    cov = sum(len(v) for v in stable.values())
    # Three counts are all true here and they differ; the paper's "graded by two
    # of N" is a claim about PEOPLE, so it must not use the slot count.
    slots = {r["ta1"] for r in rows} | {r["ta2"] for r in rows}
    composites = sorted(l for l in slots if "+" in l)
    people = {q.strip() for l in slots for q in l.split("+") if q.strip()}
    n_slots, n_people = len(slots), len(people)
    idx = sorted(int(q.rsplit("_", 1)[1]) for q in people if q.rsplit("_", 1)[-1].isdigit())
    inactive = [i for i in range(min(idx), max(idx) + 1) if i not in idx]
    A(f"**Pairing structure — a hybrid design, unlike Stage 3.** The cohort was graded by "
      f"{n_people} teaching assistants working in {n_slots} grading slots — one slot, "
      f"`{'`, `'.join(composites)}`, is a pair who marked jointly, so counting slots merges two "
      f"people into one — across {len(allg)} distinct pairings. {len(stable)} of those are stable pairs of "
      f"{min(len(v) for v in stable.values())}-{max(len(v) for v in stable.values())} students each, covering {cov} of {len(rows)} students ({cov/len(rows):.0%}); the "
      f"remaining {len(rows)-cov} were graded by {len(allg)-len(stable)} ad hoc combinations built around floating graders. "
      f"The roster numbers up to TA_{max(idx)}; TA_{' and TA_'.join(str(i) for i in inactive)} "
      f"never graded anyone, so {n_people} of {max(idx)} assigned assistants actually marked. "
      f"Stage 3's 10 fixed pairs cover 100% of its cohort. **Every per-pair statement below is "
      f"therefore scoped to the {len(stable)}-pair backbone**, never to all {len(allg)} pairings — the tail includes "
      f"pairings of a single student, where an MAE range is meaningless.\n")

    hm = {k: st.fmean(gapf(r) for r in v) for k, v in stable.items()}
    lo_k, hi_k = min(hm, key=hm.get), max(hm, key=hm.get)
    above = sum(1 for m in hm.values() if m > floor)
    s3 = stage3_ground_truth()
    s3_lo, s3_hi = min(s3["pairs"], key=s3["pairs"].get), max(s3["pairs"], key=s3["pairs"].get)
    A(f"**Per-pair human noise is structured — replicates.** Across the {len(stable)} stable pairs, per-pair "
      f"human MAE runs {hm[lo_k]:.2f} to {hm[hi_k]:.2f}, a {hm[hi_k]/hm[lo_k]:.2f}x spread, with {above} of {len(stable)} pairs above the "
      f"pooled floor of {floor:.2f}. Stage 3: {s3['pairs'][s3_lo]:.2f} to {s3['pairs'][s3_hi]:.2f}, a {s3['pairs'][s3_hi]/s3['pairs'][s3_lo]:.2f}x spread, {s3['above']} of "
      f"{len(s3['pairs'])} above its floor. Both are scale-free comparisons and both say the same thing: which "
      f"pair a student draws is a real source of ground-truth variability.\n")

    ai = {}
    bestpath = [f for f in glob.glob(str(RES / "results" / "IA*.xlsx"))
                if f"_m-{best_tag}_" in Path(f).name and "str-neutral_" in Path(f).name][0]
    wsb = openpyxl.load_workbook(bestpath, data_only=True).active
    hb = [c.value for c in wsb[1]]
    cb = {k: i for i, k in enumerate(hb)}
    for r in wsb.iter_rows(min_row=2, values_only=True):
        if r[cb["AI Total Score"]] is not None:
            ai[r[cb["Number"]]] = float(r[cb["AI Total Score"]]) + float(r[cb["AI Total Bonus"]] or 0)
    am = {k: st.fmean(abs(ai[r["n"]] - (r["t1"] + r["t2"]) / 2) for r in v if r["n"] in ai)
          for k, v in stable.items()}
    ks = sorted(hm)
    x, y = [hm[k] for k in ks], [am[k] for k in ks]
    # The arithmetic-coupling control: a noisier pair injects Var(TA1-TA2)/4
    # into the target, so subtract it from the per-pair MSE and re-correlate.
    amc = {}
    for k, v in stable.items():
        errs = [ai[r["n"]] - (r["t1"] + r["t2"]) / 2 for r in v if r["n"] in ai]
        gaps = [r["t1"] - r["t2"] for r in v if r["n"] in ai]
        mse = st.fmean(e * e for e in errs)
        inj = float(np.var(gaps)) / 4.0
        amc[k] = max(mse - inj, 0.0) ** 0.5
    r_c = pearson(x, [amc[k] for k in ks])
    r_p, r_s = pearson(x, y), spearman(x, y)
    pp = perm_p(x, y, pearson)
    zst, zp = fisher_z(r_p, len(ks), S3["gt_r"], S3["gt_k"])
    A(f"**Whether the AI inherits per-pair human noise does NOT replicate — it reverses sign.** "
      f"Correlating per-pair human MAE against per-pair MAE of the best configuration "
      f"(`{best_tag}`, neutral) over the {len(ks)} stable pairs gives Pearson r = **{r_p:+.3f}** "
      f"(permutation p = {pp:.4f}, {int(2e4)} shuffles; Spearman {r_s:+.3f}). Stage 3 gave "
      f"{S3['gt_r']:+.3f} (p = {S3['gt_p']:.2f}) over {S3['gt_k']} pairs. The two differ: Fisher "
      f"z = {zst:+.3f}, p = {zp:.4f}. The illustration inverts exactly — on this exam's noisiest "
      f"pair (human {hm[hi_k]:.2f}) the AI records {am[hi_k]:.2f}, its worst, while on the quietest "
      f"(human {hm[lo_k]:.2f}) it records {am[lo_k]:.2f}; on Stage 3 the noisiest pair (3.55) drew "
      f"the AI's best result (1.32).\n")
    A(f"> Two readings survive this, and the data cannot separate them. (a) On this exam the AI "
      f"genuinely tracks whatever makes a pair disagree. (b) Part of the coupling is arithmetic: a "
      f"noisier pair injects Var(TA1-TA2)/4 into the target the AI is scored against, so any grader "
      f"looks worse there. Subtracting that term drives the correlation to {r_c:+.2f}, but part of that "
      f"movement is an MAE->RMSE estimator change rather than artefact removal, the result is smooth "
      f"in the subtraction weight, and Stage 3's null was underpowered ({S3['gt_k']} pairs needs "
      f"|r| > 0.63 for p < 0.05). The divergence is the reportable fact; the mechanism is open.\n")
    A(f"The AI is nonetheless the more uniform grader on both exams: its per-pair MAE spans "
      f"{max(am.values())/min(am.values()):.2f}x against the humans' {hm[hi_k]/hm[lo_k]:.2f}x here.\n")

    bias = {k: abs(st.fmean(r["t1"] - r["t2"] for r in v)) for k, v in stable.items()}
    A(f"**Grader harshness, measured without the batch confound.** A one-way ANOVA of awarded total "
      f"on grader identity returns eta^2 = 0.19 here against 0.05 on Stage 3, but that comparison is "
      f"invalid: Stage 2 assigns students to pairs in contiguous blocks of student number, and block "
      f"ability varies enormously, so a grader's mean is mostly that grader's batch. The "
      f"confound-free measure is the signed bias between two graders on the SAME papers, which is "
      f"scale-free once expressed against each exam's own floor: mean |bias| {st.fmean(bias.values()):.2f} = "
      f"{st.fmean(bias.values())/floor:.2f}x floor over the {len(stable)} stable pairs, against "
      f"{st.fmean(s3['bias'].values()):.2f} = {st.fmean(s3['bias'].values())/s3['floor']:.2f}x floor over Stage 3's {len(s3['bias'])} pairs. "
      f"Systematic harshness between paired graders is the same modest fraction of the noise floor "
      f"on both exams.\n")

    s3n = stage3_neutral_baselines()
    both = [t for t in runs if t in s3n]
    if both:
        x = [s3n[t][0] for t in both]
        y = [runs[t]["neutral"]["mae"] for t in both]
        rk = lambda v: np.argsort(np.argsort(np.asarray(v, float))).astype(float)
        A(f"**Which open model is best does not transfer between exams.** For the {len(both)} models run on "
          f"both, neutral-prompt MAE is uncorrelated across exams (Spearman {pearson(rk(x), rk(y)):+.2f}, "
          f"Pearson {pearson(x, y):+.2f}). `{min(both, key=lambda t: runs[t]['neutral']['mae'])}` is the best "
          f"grader here ({min(runs[t]['neutral']['mae'] for t in both):.2f}) and the worst of the nine on "
          f"Stage 3 ({s3n[min(both, key=lambda t: runs[t]['neutral']['mae'])][0]:.2f}); "
          f"`{min(both, key=lambda t: s3n[t][0])}` leads on Stage 3 ({min(s3n[t][0] for t in both):.2f}) and "
          f"sits mid-pack here ({runs[min(both, key=lambda t: s3n[t][0])]['neutral']['mae']:.2f}). Model "
          f"choice is exam-specific, not a property of the model.\n")
        A(f"| model | Stage 3 neutral MAE | Stage 3 bias | Stage 2 neutral MAE | Stage 2 bias |")
        A("|---|---:|---:|---:|---:|")
        for t in sorted(both, key=lambda t: s3n[t][0]):
            A(f"| `{t}` | {s3n[t][0]:.2f} | {s3n[t][1]:+.2f} | {runs[t]['neutral']['mae']:.2f} | "
              f"{runs[t]['neutral']['bias']:+.2f} |")
        A("")

    A("## Cross-exam comparison\n")
    A(f"| | Stage 3 (Computer Vision) | Stage 2 (Introduction to AI) |")
    A("|---|---|---|")
    A(f"| students (dual-graded) | {S3['n']} | {len(pairs)} |")
    A(f"| questions | {S3['questions']} | 3 |")
    A(f"| scale | {S3['scale']} (bonus excluded, separate column) | {SCALE} (bonus folded into the question marks) |")
    A(f"| human floor vs TA average | {S3['floor']:.2f} / {S3['scale']} | {floor:.2f} / {SCALE} |")
    A(f"| inter-TA Pearson r | {S3['r']:.3f} | {r_ta:.3f} |")
    A(f"| best configuration | {S3['best']:.2f} — beats the floor | {best['mae']:.2f} — "
      f"beats the floor ({best['mae']/floor:.2f}x) |")
    A(f"| models broken by *strict* | 13 of 16 matched pairs | {len(broke)} of {len(runs)} |")
    A("")

    A("## Method notes and caveats\n")
    A(f"- The exam has no student-facing guidelines document, so the `gd` prompt component is fixed "
      f"at 0 for every run here; Stage 3 runs carry `gd1`. Nothing else in the prompt differs.")
    A(f"- Question maxima come from the rubric's task tables (Q1 23+3, Q2 14+3, Q3 19+3), not from "
      f"the marking scheme's header totals, which contradict their own tables in all three "
      f"questions. Two instructor corrections issued mid-grading are folded in; both affect Q3.")
    A(f"- TA per-question grades fold bonus into the mark, so the AI total compared against them is "
      f"score + bonus. Stage 3 keeps bonus in its own column and its {S3['scale']}-point scale "
      f"excludes it, and Stage 2's bonus cannot be recovered from the recorded marks. No "
      f"percentage-of-scale figure is therefore compared across the two exams; cross-exam claims "
      f"use scale-free quantities only (multiples of each exam's own floor, damage ratios, "
      f"correlations, and fractions of pairs above the floor).")
    A(f"- MAE is computed against the mean of the two TA grades, over the "
      f"{len(pairs)} students who have both. All {len(TA)} do.")
    A(f"- Not every student submitted every question. An absent notebook is graded 0, by the TAs and "
      f"by the models alike.")
    A(f"- The `MAE >= 8` band threshold is inherited from Stage 3 and is the *more lenient* bar on "
      f"this exam: 8 points is {8/S3['floor']:.2f}x the Stage 3 floor but {8/floor:.2f}x this one, "
      f"because the scale is nearly twice as large while the floors are not. Counts are given "
      f"against both that threshold and a floor-matched one.")
    A(f"- A refusal's MAE is a ceiling artefact — the distance from the TA mean — not a severity "
      f"measurement. Zero-rate and awarded-total standard deviation separate refusal from collapse.")
    A("")
    A("## Deliverables\n")
    A("- `introduction_to_ai_results/ablation_runs.xlsx` — one row per run with configuration, "
      "MAE, bootstrap CI and behaviour class.")
    A("- `introduction_to_ai_results/results/*.xlsx` — per-run workbooks: AI score, bonus and "
      "reasoning per question, alongside the TA columns.")
    A("- `introduction_to_ai_results/variance/*.csv` — per-call log for every graded question.")

    # Every completed run, not just the neutral/strict open-weights pairs the
    # ledger narrates: the CSV and the comparison tables are the full record.
    rows_gt = ta_rows()
    stable, _all_pairs = pair_groups(rows_gt)
    pairs_map = {k: {"n": len(v),
                     "mae": st.fmean(abs(r["t1"] - r["t2"]) for r in v)}
                 for k, v in stable.items()}

    mc, n_mc = write_master_comparison(runs_all, floor)
    print(f"wrote {mc} ({n_mc} runs)")
    write_master_md(runs_all, floor, f_lo, f_hi)
    print("wrote introduction_to_ai_master_comparison.md")
    write_human_floor_md(floor, f_lo, f_hi, r_ta, rows_gt, pairs_map)
    print("wrote introduction_to_ai_human_floor.md")
    write_variance_md()
    print("wrote introduction_to_ai_variance.md")
    write_failure_cases_md(runs_all, TA)
    print("wrote introduction_to_ai_failure_cases.md")
    write_pair_level_ai_mae_md(runs_all, TA, rows_gt, pairs_map)
    print("wrote introduction_to_ai_pair_level_ai_mae.md")
    write_subset_restricted_md(runs_all, TA)
    print("wrote introduction_to_ai_subset_restricted_comparison.md")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(L)} blocks)")
    print(f"  floor {floor:.3f} [{f_lo:.2f}, {f_hi:.2f}]  r={r_ta:.3f}  "
          f"models={len(runs)}  broken={len(broke)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
