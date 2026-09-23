#!/usr/bin/env python3
"""Paired third-grader test (Eq. 1 / Table 15 rule) for the zero-shot
closed configurations on the FULL CV cohort.

Runs: D01, D02, F02, B03 (the four floor-clearing Gemini configurations of
Table 7), plus A01 (Flash-Lite baseline, Table 7 reference row) and the two
cross-vendor neutral runs O01 (gpt-5.5) and N01 (claude-opus-5).

Per student i:  d_i = 0.5(|AI-G1| + |AI-G2|) - |G1-G2|          (Eq. 1)
Reported per run: n, mean|AI-G_avg| (the paper's MAE, with the paper's
percentile-bootstrap CI and bias), mean|AI-G1|, mean|AI-G2|, the floor
mean|G1-G2| on the same students, d-bar with (a) a 95% percentile bootstrap CI
over students (2000 resamples, numpy default_rng seed 0) and (b) the
normal-approximation CI mean +/- 1.96*SE used by finetune/pairwise_analysis.py
for Table 15, and the verdict under Table 15's rule (better if the whole CI is
below 0, "= grader" if it contains 0, worse if the whole CI is above 0).

Conventions (paper, CV exam): AI total = workbook "AI Total Score" (Q1-Q3 base,
35 points); grader totals = "TA 1 - Total Score (out of 35)" and
"TA 2 - Total Score (out of 35)"; rows with a blank AI total or a blank grader
total are excluded.  Every number printed below is computed here.
"""
from __future__ import annotations

import glob
import math
import os
import sys

import numpy as np
import openpyxl

try:
    from scipy import stats as sps
except Exception:  # scipy is optional; only used for the paired t-test line
    sps = None

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
N_BOOT = 2000
SEED = 0
PAPER_FLOOR = 2.61  # Section 4 / Table 7, rounded as printed in the paper

RUNS = [
    # rid, glob pattern (relative to ROOT), label, paper Table-7 numbers (MAE, lo, hi, bias)
    ("D01", "computer_vision_results/results/D01__*.xlsx", "gemini-3-flash-preview, neutral", (1.64, 1.51, 1.79, +0.01)),
    ("F02", "computer_vision_results/results/F02__*.xlsx", "gemini-3.1-pro-preview + lenient", (1.79, 1.62, 1.95, +0.82)),
    ("D02", "computer_vision_results/results/D02__*.xlsx", "gemini-3.1-pro-preview, neutral", (1.86, 1.70, 2.02, -0.82)),
    ("B03", "computer_vision_results/results/B03__*.xlsx", "gemini-flash-lite + thinking, neutral", (2.05, 1.89, 2.21, -0.15)),
    ("A01", "computer_vision_results/results/A01__*.xlsx", "gemini-flash-lite baseline, neutral", (3.34, 3.15, 3.53, -1.78)),
    ("O01", "computer_vision_results/results/O01__*.xlsx", "gpt-5.5, neutral (batch API)", (2.431, 2.276, 2.594, -1.652)),
    ("O03", "computer_vision_results/results/O03__*.xlsx", "gpt-5.5 + lenient (batch API)", (2.251, 2.072, 2.439, +1.131)),
    ("N01", "computer_vision_results/results/N01__*.xlsx", "claude-opus-5, neutral (batch API)", (3.538, 3.334, 3.762, -3.242)),
]

AI_COL, T1_COL, T2_COL = "AI Total Score", "TA 1 - Total Score (out of 35)", "TA 2 - Total Score (out of 35)"


def blank(v) -> bool:
    return v is None or v == ""


def load(pattern: str):
    """Return (path, dict) with AI / G1 / G2 arrays in workbook row order plus provenance."""
    hits = sorted(glob.glob(os.path.join(ROOT, pattern)))
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one workbook for {pattern}, found {hits}")
    path = hits[0]
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb.active
    hdr = [c.value for c in ws[1]]
    col = {h: i for i, h in enumerate(hdr) if h}
    need = [AI_COL, T1_COL, T2_COL, "Number"]
    for c in need:
        if c in col:
            continue
        raise SystemExit(f"{path}: missing column {c!r}")
    ai, g1, g2, num, rows_used, rows_excl = [], [], [], [], [], []
    rebuilt_dev = 0.0          # max |cached grader total - (Q1+Q2+Q3 Score)|, both graders
    ai_dev = 0.0               # max |AI Total Score - (AI Q1+Q2+Q3 Score)|
    qcols_ok = all(f"TA {t} - Q{q} Score" in col for t in (1, 2) for q in (1, 2, 3)) and \
        all(f"AI Q{q} Score" in col for q in (1, 2, 3))
    for ridx, r in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
        a, t1, t2 = r[col[AI_COL]], r[col[T1_COL]], r[col[T2_COL]]
        if blank(a) or blank(t1) or blank(t2):
            rows_excl.append(ridx)
            continue
        ai.append(float(a)); g1.append(float(t1)); g2.append(float(t2))
        num.append(r[col["Number"]]); rows_used.append(ridx)
        if qcols_ok:
            for t, cached in ((1, t1), (2, t2)):
                qs = [r[col[f"TA {t} - Q{q} Score"]] for q in (1, 2, 3)]
                if all(isinstance(x, (int, float)) for x in qs):
                    rebuilt_dev = max(rebuilt_dev, abs(float(cached) - sum(float(x) for x in qs)))
            qa = [r[col[f"AI Q{q} Score"]] for q in (1, 2, 3)]
            if all(isinstance(x, (int, float)) for x in qa):
                ai_dev = max(ai_dev, abs(float(a) - sum(float(x) for x in qa)))
    info = dict(path=os.path.relpath(path, ROOT), sheet=ws.title, max_row=ws.max_row,
                rows_used=rows_used, rows_excl=rows_excl, numbers=num,
                rebuilt_dev=rebuilt_dev, ai_dev=ai_dev, qcols_ok=qcols_ok)
    return np.array(ai), np.array(g1), np.array(g2), info


def boot_idx(n: int):
    """The resample index matrix: numpy default_rng(SEED), 2000 x n, exactly as
    analysis/computer_vision_run_analysis.py::bootstrap_mae_ci draws it."""
    rng = np.random.default_rng(SEED)
    return rng.integers(0, n, size=(N_BOOT, n))


def pct_ci(samples):
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def normal_ci(x):
    m = float(np.mean(x))
    se = float(np.std(x, ddof=1)) / math.sqrt(len(x))   # statistics.stdev == ddof=1
    return m, m - 1.96 * se, m + 1.96 * se, se


def verdict(lo: float, hi: float) -> str:
    if hi < 0:
        return "better"
    if lo > 0:
        return "worse"
    return "= grader"


def f3(x: float) -> str:
    return f"{x:.3f}"


def s3(x: float) -> str:
    return f"{x:+.3f}"


def main() -> None:
    print("# Paired third-grader test for the zero-shot closed configurations (CV exam, full cohort)")
    print()
    print(f"Repo root: `{ROOT}`  ")
    print(f"Bootstrap: percentile, {N_BOOT} resamples over students, `numpy.random.default_rng({SEED})`, "
          f"index matrix `rng.integers(0, n, size=({N_BOOT}, n))` drawn once per run and reused for every "
          f"bootstrapped quantity of that run (so the MAE CI is the paper's own draw).  ")
    print("Normal CI: mean +/- 1.96 * sd(ddof=1)/sqrt(n), as in `finetune/pairwise_analysis.py::mean_ci` (Table 15).  ")
    print("Verdict rule (Table 15 caption): *better* = whole 95% CI of d-bar below 0; *= grader* = CI contains 0; *worse* = whole CI above 0.")
    print()

    # ---------------------------------------------------------------- load
    data = {}
    print("## 1. Inputs and row provenance")
    print()
    print("| Run | Workbook (relative to repo root) | Sheet | Data rows | n used | Rows excluded (blank AI or grader total) | Student `Number` range | max\\|cached grader total - (Q1+Q2+Q3 Score)\\| | max\\|AI Total Score - (AI Q1+Q2+Q3 Score)\\| |")
    print("|---|---|---|---|---:|---|---|---:|---:|")
    for rid, pat, label, paper in RUNS:
        ai, g1, g2, info = load(pat)
        data[rid] = (ai, g1, g2, info, label, paper)
        nums = info["numbers"]
        try:
            rng_txt = f"{min(nums)}..{max(nums)}"
        except TypeError:
            rng_txt = f"{nums[0]}..{nums[-1]}"
        print(f"| {rid} | `{info['path']}` | {info['sheet']} | 2..{info['max_row']} | {len(ai)} | "
              f"{len(info['rows_excl'])} | {rng_txt} | {info['rebuilt_dev']:.3f} | {info['ai_dev']:.3f} |")
    print()
    # identical student sets?
    ref_nums = data["D01"][3]["numbers"]
    same = all(data[r][3]["numbers"] == ref_nums for r in data)
    print(f"Student sets identical across all {len(RUNS)} runs (same `Number` values in the same row order): **{same}**  ")
    g1_ref, g2_ref = data["D01"][1], data["D01"][2]
    same_g = all(np.array_equal(data[r][1], g1_ref) and np.array_equal(data[r][2], g2_ref) for r in data)
    print(f"Grader totals G1, G2 identical across all runs: **{same_g}**  ")
    print()

    # ---------------------------------------------------------------- floor
    n = len(g1_ref)
    h = np.abs(g1_ref - g2_ref)
    idx = boot_idx(n)
    floor = float(h.mean())
    floor_lo, floor_hi = pct_ci(h[idx].mean(axis=1))
    print("## 2. Human floor on these students and the reference bounds")
    print()
    print(f"- Floor = mean |G1 - G2| over the same n = {n} students = **{floor:.4f}** "
          f"(95% percentile bootstrap CI [{floor_lo:.3f}, {floor_hi:.3f}]); the paper prints {PAPER_FLOOR:.2f} [2.37, 2.85].")
    g_avg_ref = (g1_ref + g2_ref) / 2
    e_g1_avg = float(np.abs(g1_ref - g_avg_ref).mean())
    print(f"- Single grader vs the two-grader average, measured directly: mean |G1 - G_avg| = mean |G2 - G_avg| = "
          f"**{e_g1_avg:.4f}** = 0.5 * floor exactly (an identity, since |G1 - (G1+G2)/2| = 0.5|G1 - G2|).")
    b_half_paper = 0.5 * PAPER_FLOOR
    b_half_exact = 0.5 * floor
    b_58_paper = math.sqrt(5 / 8) * PAPER_FLOOR
    b_58_exact = math.sqrt(5 / 8) * floor
    b_866_paper = math.sqrt(3 / 4) * PAPER_FLOOR
    b_866_exact = math.sqrt(3 / 4) * floor
    print(f"- Bound (i): single grader vs average = 0.5 * 2.61 = **{b_half_paper:.4f}** "
          f"(with the unrounded floor: 0.5 * {floor:.4f} = {b_half_exact:.4f}).")
    print(f"- Bound (ii): i.i.d. third human vs average = sqrt(5/8) * 2.61 = {math.sqrt(5/8):.6f} * 2.61 = "
          f"**{b_58_paper:.4f}** (with the unrounded floor: {b_58_exact:.4f}). "
          f"Derivation as written in the review: Var(G3 - G_avg) = sigma^2 + sigma^2/4 = 1.25 sigma^2 against Var(G1 - G2) = 2 sigma^2, "
          f"ratio of SDs sqrt(1.25/2) = sqrt(5/8). Note: with two i.i.d. graders Var(G_avg) = Var(G1 + G2)/4 = 2 sigma^2/4 = sigma^2/2, "
          f"not sigma^2/4, so (ii) understates the bound; the internally consistent figure is (iii).")
    print(f"- Bound (iii) (the review's Seg. 2 figure, 0.866 x 2.61 ~ 2.26): sqrt(3/4) * 2.61 = {math.sqrt(3/4):.6f} * 2.61 = "
          f"**{b_866_paper:.4f}** (with the unrounded floor: {b_866_exact:.4f}). Derivation: Var(G3 - G_avg) = sigma^2 + sigma^2/2 = 1.5 sigma^2, "
          f"ratio of SDs sqrt(1.5/2) = sqrt(3/4). Both (ii) and (iii) also assume E|X| scales with SD(X), exact for Gaussian noise; "
          f"(i) is an identity and needs no assumption.")
    print()

    # ---------------------------------------------------------------- per-run
    print("## 3. Per-run statistics (all n = 570)")
    print()
    print("### 3a. Paper-convention MAE check (reproduces Table 7 rows)")
    print()
    print("| Run | Model | n | MAE = mean\\|AI - G_avg\\| | 95% boot CI | bias = mean(AI - G_avg) | paper says (MAE [CI], bias) |")
    print("|---|---|---:|---:|---|---:|---|")
    res = {}
    for rid, pat, label, paper in RUNS:
        ai, g1, g2, info, label, paper = data[rid]
        n = len(ai)
        idx = boot_idx(n)
        gavg = (g1 + g2) / 2
        e1, e2, eavg, h = np.abs(ai - g1), np.abs(ai - g2), np.abs(ai - gavg), np.abs(g1 - g2)
        d = 0.5 * (e1 + e2) - h
        mae = float(eavg.mean()); mae_lo, mae_hi = pct_ci(eavg[idx].mean(axis=1))
        bias = float((ai - gavg).mean())
        m1 = float(e1.mean()); m1_lo, m1_hi = pct_ci(e1[idx].mean(axis=1))
        m2 = float(e2.mean()); m2_lo, m2_hi = pct_ci(e2[idx].mean(axis=1))
        mh = float(h.mean()); mh_lo, mh_hi = pct_ci(h[idx].mean(axis=1))
        m12 = float((0.5 * (e1 + e2)).mean()); m12_lo, m12_hi = pct_ci((0.5 * (e1 + e2))[idx].mean(axis=1))
        dbar = float(d.mean()); d_lo, d_hi = pct_ci(d[idx].mean(axis=1))
        dm, dn_lo, dn_hi, dse = normal_ci(d)
        ttest = None
        if sps is not None:
            t = sps.ttest_1samp(d, 0.0)
            ttest = (float(t.statistic), float(t.pvalue))
        neg, zero, pos = int((d < 0).sum()), int((d == 0).sum()), int((d > 0).sum())
        between = int(((ai >= np.minimum(g1, g2)) & (ai <= np.maximum(g1, g2))).sum())
        res[rid] = dict(n=n, mae=mae, mae_lo=mae_lo, mae_hi=mae_hi, bias=bias, m1=m1, m1_lo=m1_lo, m1_hi=m1_hi,
                        m2=m2, m2_lo=m2_lo, m2_hi=m2_hi, mh=mh, mh_lo=mh_lo, mh_hi=mh_hi, m12=m12, m12_lo=m12_lo,
                        m12_hi=m12_hi, dbar=dbar, d_lo=d_lo, d_hi=d_hi, dn_lo=dn_lo, dn_hi=dn_hi, dse=dse,
                        ttest=ttest, neg=neg, zero=zero, pos=pos, between=between, d=d, label=label,
                        med=float(np.median(d)), dsd=float(np.std(d, ddof=1)))
        pm, plo, phi, pb = paper
        print(f"| {rid} | {label} | {n} | {f3(mae)} | [{f3(mae_lo)}, {f3(mae_hi)}] | {s3(bias)} | "
              f"{pm:.3f} [{plo:.3f}, {phi:.3f}], {pb:+.3f} |")
    print()
    print("(O01 / N01 paper CIs come from `analysis/closed_vendor_table.py`, which draws its resamples with "
          "`random.Random(0)` rather than numpy, so their third decimal need not coincide; the point estimates and bias must.)")
    print()

    print("### 3b. Single-grader errors and the floor on the same students (like-for-like, all 1-vs-1)")
    print()
    print("| Run | n | mean\\|AI - G1\\| [95% boot CI] | mean\\|AI - G2\\| [95% boot CI] | 0.5(mean\\|AI - G1\\| + mean\\|AI - G2\\|) [95% boot CI] | floor mean\\|G1 - G2\\| [95% boot CI] | \\|G2 - G1 error gap\\| (symmetry guard) |")
    print("|---|---:|---|---|---|---|---:|")
    for rid, *_ in RUNS:
        r = res[rid]
        print(f"| {rid} | {r['n']} | {f3(r['m1'])} [{f3(r['m1_lo'])}, {f3(r['m1_hi'])}] | "
              f"{f3(r['m2'])} [{f3(r['m2_lo'])}, {f3(r['m2_hi'])}] | "
              f"{f3(r['m12'])} [{f3(r['m12_lo'])}, {f3(r['m12_hi'])}] | "
              f"{f3(r['mh'])} [{f3(r['mh_lo'])}, {f3(r['mh_hi'])}] | {abs(r['m2'] - r['m1']):.3f} |")
    print()

    print("### 3c. Paired third-grader statistic d-bar (Eq. 1) with both CIs and the Table 15 verdict")
    print()
    print("| Run | Model | n | \\|AI - G_avg\\| | \\|AI - G1\\| | \\|AI - G2\\| | floor | d-bar | 95% percentile bootstrap CI | verdict (bootstrap) | 95% normal CI (mean +/- 1.96 SE) | verdict (normal) | SE of d-bar | sd(d) | median d |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|---:|---:|")
    for rid, *_ in RUNS:
        r = res[rid]
        print(f"| {rid} | {r['label']} | {r['n']} | {f3(r['mae'])} | {f3(r['m1'])} | {f3(r['m2'])} | {f3(r['mh'])} | "
              f"**{s3(r['dbar'])}** | [{s3(r['d_lo'])}, {s3(r['d_hi'])}] | **{verdict(r['d_lo'], r['d_hi'])}** | "
              f"[{s3(r['dn_lo'])}, {s3(r['dn_hi'])}] | **{verdict(r['dn_lo'], r['dn_hi'])}** | {r['dse']:.4f} | {r['dsd']:.3f} | {s3(r['med'])} |")
    print()
    print("Identity check: d-bar = 0.5(mean|AI-G1| + mean|AI-G2|) - floor, i.e. column 0.5(...) of 3b minus the floor:")
    for rid, *_ in RUNS:
        r = res[rid]
        print(f"- {rid}: {f3(r['m12'])} - {f3(r['mh'])} = {s3(r['m12'] - r['mh'])} (d-bar printed above: {s3(r['dbar'])}; "
              f"difference {r['dbar'] - (r['m12'] - r['mh']):+.1e})")
    print()

    print("### 3d. Per-student sign of d and paired t-test (supplementary)")
    print()
    print("| Run | d < 0 (AI closer to the graders than they are to each other) | d = 0 | d > 0 | AI total lies within [min(G1,G2), max(G1,G2)] | paired t (H0: mean d = 0) | two-sided p |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for rid, *_ in RUNS:
        r = res[rid]
        tt = r["ttest"]
        tt_txt = (f"{tt[0]:+.3f}", f"{tt[1]:.3g}") if tt else ("n/a (scipy missing)", "n/a")
        print(f"| {rid} | {r['neg']} ({100*r['neg']/r['n']:.1f}%) | {r['zero']} ({100*r['zero']/r['n']:.1f}%) | "
              f"{r['pos']} ({100*r['pos']/r['n']:.1f}%) | {r['between']} ({100*r['between']/r['n']:.1f}%) | {tt_txt[0]} | {tt_txt[1]} |")
    print()

    # ---------------------------------------------------------------- bounds comparison
    print("## 4. Consensus MAE against the human-to-average bounds")
    print()
    print(f"Bounds: (i) 0.5 * 2.61 = {b_half_paper:.4f}; (ii) sqrt(5/8) * 2.61 = {b_58_paper:.4f}; (iii) sqrt(3/4) * 2.61 = {b_866_paper:.4f}; raw floor 2.61. "
          f"A run 'clears' a bound when its MAE 95% bootstrap CI upper end is below it; 'exceeds' when the lower end is above it; otherwise 'straddles'.")
    print()
    print("| Run | MAE [95% CI] | vs (i) 1.305 | vs (ii) 2.063 | vs (iii) 2.260 | vs floor 2.61 | MAE / floor | 0.5(\\|AI-G1\\|+\\|AI-G2\\|) / floor |")
    print("|---|---|---|---|---|---|---:|---:|")

    def cmp(lo, hi, b):
        if hi < b:
            return f"clears ({hi:.3f} < {b:.3f})"
        if lo > b:
            return f"exceeds ({lo:.3f} > {b:.3f})"
        return f"straddles ([{lo:.3f}, {hi:.3f}] contains {b:.3f})"

    for rid, *_ in RUNS:
        r = res[rid]
        print(f"| {rid} | {f3(r['mae'])} [{f3(r['mae_lo'])}, {f3(r['mae_hi'])}] | {cmp(r['mae_lo'], r['mae_hi'], b_half_paper)} | "
              f"{cmp(r['mae_lo'], r['mae_hi'], b_58_paper)} | {cmp(r['mae_lo'], r['mae_hi'], b_866_paper)} | {cmp(r['mae_lo'], r['mae_hi'], PAPER_FLOOR)} | "
              f"{r['mae']/floor:.3f} | {r['m12']/floor:.3f} |")
    print()

    # ---------------------------------------------------------------- Table 15 style summary
    print("## 5. Table 15-style summary rows (same columns as tab:ftpaired; normal CI as in Table 15, bootstrap CI added)")
    print()
    print("| Cell | n | \\|AI-G_avg\\| | \\|AI-G1\\| | d-bar [95% normal CI] | d-bar [95% bootstrap CI] | Verdict |")
    print("|---|---:|---:|---:|---|---|---|")
    for rid, *_ in RUNS:
        r = res[rid]
        v_n, v_b = verdict(r['dn_lo'], r['dn_hi']), verdict(r['d_lo'], r['d_hi'])
        v = v_n if v_n == v_b else f"{v_n} (normal) / {v_b} (bootstrap)"
        print(f"| {rid} zero-shot, CV | {r['n']} | {r['mae']:.2f} | {r['m1']:.2f} | {r['dbar']:+.2f} [{r['dn_lo']:+.2f}, {r['dn_hi']:+.2f}] | "
              f"{r['dbar']:+.2f} [{r['d_lo']:+.2f}, {r['d_hi']:+.2f}] | **{v}** |")
    print()
    print("(2-decimal rendering for drop-in use in the paper; the 3-decimal values are in Section 3c above.)")
    print()

    # ---------------------------------------------------------------- one-line headline
    print("## 6. Headline")
    print()
    parts = []
    for rid, *_ in RUNS:
        r = res[rid]
        parts.append(f"{rid} d-bar {s3(r['dbar'])} boot [{s3(r['d_lo'])}, {s3(r['d_hi'])}] normal [{s3(r['dn_lo'])}, {s3(r['dn_hi'])}] "
                     f"-> {verdict(r['d_lo'], r['d_hi'])}; |AI-G1| {f3(r['m1'])}, |AI-G2| {f3(r['m2'])}, |AI-Gavg| {f3(r['mae'])}, floor {f3(r['mh'])}, n {r['n']}")
    print("; ".join(parts) + ".")
    print(f"Reference bounds on the CV floor 2.61: 0.5 * 2.61 = {b_half_paper:.4f}; sqrt(5/8) * 2.61 = {b_58_paper:.4f}; "
          f"sqrt(3/4) * 2.61 = {b_866_paper:.4f} (unrounded floor {floor:.4f}: {b_half_exact:.4f}, {b_58_exact:.4f}, {b_866_exact:.4f}).")


if __name__ == "__main__":
    main()
